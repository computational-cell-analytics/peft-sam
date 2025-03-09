import os
import shutil
import subprocess
import itertools
from datetime import datetime

DATASETS = {"hpa": "lm", "psfhs": "medical_imaging"}
EXPERIMENT_ROOT = "/scratch/usr/nimcarot/sam/experiments/peft"
ALL_SCRIPTS = ["evaluate_instance_segmentation", "iterative_prompting"]


def write_batch_script(
    env_name,
    out_path,
    inference_setup,
    checkpoint,
    model_type,
    experiment_folder,
    dataset,
    peft_method=None,
    peft_rank=None,
    attention_layers_to_update=[],
    update_matrices=[],
    dry=False,
):
    "Writing scripts with different fold-trainings for micro-sam evaluation"
    batch_script = f"""#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -p grete:shared
#SBATCH -t 2-00:00:00
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH --job-name={inference_setup}

source ~/.bashrc
mamba activate {env_name}
"""

    # python script
    inference_script_path = f"../evaluation/{inference_setup}.py"
    python_script = f"python {inference_script_path} "

    _op = out_path[:-3] + f"_{inference_setup}.sh"

    if checkpoint is not None:  # add the finetuned checkpoint
        python_script += f"-c {checkpoint} "

    # name of the model configuration
    python_script += f"-m {model_type} "

    # experiment folder
    python_script += f"-e {experiment_folder} "

    # IMPORTANT: choice of the dataset

    if dataset == "platy_cilia":
        dataset = "platynereis/cilia"
    elif dataset == "mitolab_glycolytic_muscle":
        dataset = "mitolab/glycolytic_muscle"

    python_script += f"-d {dataset} "

    if peft_rank is not None:
        python_script += f"--peft_rank {peft_rank} "
    if peft_method is not None:
        python_script += f"--peft_module {peft_method} "

    if attention_layers_to_update:
        python_script += "--attention_layers_to_update "
        for layer in attention_layers_to_update:
            python_script += f"{layer} "
    if update_matrices:
        python_script += "--update_matrices "
        for matrix in update_matrices:
            python_script += f"{matrix} "

    # let's add the python script to the bash script
    batch_script += python_script
    with open(_op, "w") as f:
        f.write(batch_script)

    if not dry:
        cmd = ["sbatch", _op]
        subprocess.run(cmd)

    # we run the first prompt for iterative once starting with point, and then starting with box (below)
    if inference_setup == "iterative_prompting":
        batch_script += "--box "

        new_path = out_path[:-3] + f"_{inference_setup}_box.sh"
        with open(new_path, "w") as f:
            f.write(batch_script)

        if not dry:
            cmd = ["sbatch", new_path]
            subprocess.run(cmd)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "micro-sam-inference"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def run_peft_evaluations(args):
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"
    if os.path.exists(tmp_folder):
        shutil.rmtree("./gpu_jobs")

    update_matrices = {'standard': ["q", "v"], 'all_matrices': ["q", "k", "v", "mlp"]}
    attention_layers_to_update = [[6, 7, 8, 9, 10, 11], [9, 10, 11], [11]]
    peft_methods = ["lora", "ClassicalSurgery"]

    experiment_folder = EXPERIMENT_ROOT  # for Caro.
    # experiment_folder = args.experiment_folder  # for Anwai and custom usage.

    for dataset, domain in DATASETS.items():
        if args.dataset is not None and args.dataset != dataset:
            continue

        SCRIPTS = ["iterative_prompting"] if domain == "medical_imaging" else ALL_SCRIPTS

        model = f"vit_b_{domain}"
        checkpoint_path = f"/scratch/usr/nimcarot/sam/models/{model}.pt" if model == "vit_b_lm" else None
        # run generalist / vanilla
        result_path = os.path.join(experiment_folder, model, dataset)
        os.makedirs(result_path, exist_ok=True)
        for current_setup in SCRIPTS:
            write_batch_script(
                env_name="peft-sam-qlora",
                out_path=get_batch_script_names(tmp_folder),
                inference_setup=current_setup,
                checkpoint=checkpoint_path,
                model_type=model,
                experiment_folder=result_path,
                dataset=dataset,
                dry=args.dry
            )

        for method, layers, update_matrix in itertools.product(
            peft_methods, attention_layers_to_update, update_matrices.keys()
        ):
            if method == "ClassicalSurgery" and update_matrices[update_matrix] != ["q", "v"]:
                continue

            # late lora and partial freezing
            checkpoint = f"{experiment_folder}/checkpoints/{model}/late_lora/{method}/"
            checkpoint += f"{update_matrix}/start_{layers[0]}/{dataset}_sam/best.pt"
            if method == "ClassicalSurgery":
                continue

            assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"
            result_path = os.path.join(experiment_folder, method, update_matrix, f"start_{layers[0]}", dataset)
            os.makedirs(result_path, exist_ok=True)

            for current_setup in ALL_SCRIPTS:
                write_batch_script(
                    env_name="peft-sam",
                    out_path=get_batch_script_names(tmp_folder),
                    inference_setup=current_setup,
                    checkpoint=checkpoint,
                    model_type=model,
                    experiment_folder=result_path,
                    dataset=dataset,
                    peft_method=method,
                    peft_rank=32,
                    attention_layers_to_update=layers,
                    update_matrices=update_matrices[update_matrix],
                    dry=args.dry,
                )


def main(args):
    run_peft_evaluations(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--experiment_folder", type=str, help="The directory where the results from evaluation are stored.",
        default="/mnt/vast-nhr/projects/cidas/cca/experiments/peft_sam/models",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default=None,
        help="Run the experiments for a specific supported biomedical imaging dataset."
    )
    parser.add_argument("--dry", action="store_true")

    args = parser.parse_args()
    main(args)
