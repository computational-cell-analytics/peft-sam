import os
import shutil
import subprocess
from datetime import datetime

from peft_sam.dataset.preprocess_datasets import preprocess_data

from micro_sam.util import export_custom_qlora_model

from submit_peft_finetuning import ALL_DATASETS


PEFT_METHODS = {
    "lora": {"peft_rank": 32},
    "ssf": {"peft_rank": 2},
    "adaptformer": {"peft_rank": 2, "alpha": "learnable_scalar", "dropout": None, "proj_size": 64},
    "fact": {"peft_rank": 16, "dropout": 0.1},
    "AttentionSurgery": {"peft_rank": 2},
    "BiasSurgery": {"peft_rank": 2},
    "LayerNormSurgery": {"peft_rank": 2},
    "qlora": {"peft_rank": 32, "quantize": True},
}


EXPERIMENT_ROOT = "/scratch/usr/nimcarot/sam/experiments/peft"
ALL_SCRIPTS = [
    # "evaluate_instance_segmentation",
    "iterative_prompting",
]


def write_batch_script(
    env_name,
    out_path,
    inference_setup,
    checkpoint,
    model_type,
    experiment_folder,
    dataset,
    use_masks=False,
    peft_method=None,
    freeze=None,
    peft_rank=None,
    alpha=None,
    proj_size=None,
    dropout=None,
    quantize=False,
    dry=False,
):
    "Writing scripts with different fold-trainings for micro-sam evaluation"
    batch_script = f"""#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -p grete:shared
#SBATCH -t 2-00:00:00
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH --constraint=80gb
#SBATCH --job-name={inference_setup}

source ~/.bashrc
micromamba activate {env_name}
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
    if alpha is not None:
        python_script += f"--alpha {alpha} "
    if proj_size is not None:
        python_script += f"--projection_size {proj_size} "
    if dropout is not None:
        python_script += f"--dropout {dropout} "
    if quantize:
        python_script += "--quantize "

    # let's add the python script to the bash script
    batch_script += python_script
    with open(_op, "w") as f:
        f.write(batch_script)

    if not args.dry:
        cmd = ["sbatch", _op]
        subprocess.run(cmd)

    # we run the first prompt for iterative once starting with point, and then starting with box (below)
    if inference_setup == "iterative_prompting":
        batch_script += "--box "

        new_path = out_path[:-3] + f"_{inference_setup}_box.sh"
        with open(new_path, "w") as f:
            f.write(batch_script)

        if not args.dry:
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

    # experiment_folder = EXPERIMENT_ROOT  # for Caro.
    experiment_folder = args.experiment_folder  # for Anwai and custom usage.

    for dataset, domain in ALL_DATASETS.items():
        if args.dataset is not None and args.dataset != dataset:
            print(f"The chosen dataset is not supported atm: '{args.dataset}'")
            continue

        preprocess_data(dataset)

        gen_model = f"vit_b_{domain}"
        models = ["vit_b"] if dataset == "livecell" else ["vit_b", gen_model]
        for model in models:
            # run generalist / vanilla
            modality = "generalist" if model == gen_model else "vanilla"
            result_path = os.path.join(experiment_folder, modality, dataset)
            os.makedirs(result_path, exist_ok=True)
            for current_setup in ALL_SCRIPTS:
                write_batch_script(
                    env_name="super",
                    out_path=get_batch_script_names(tmp_folder),
                    inference_setup=current_setup,
                    checkpoint=None,
                    model_type=model,
                    experiment_folder=result_path,
                    dataset=dataset
                )

            # full finetuning
            checkpoint = f"{experiment_folder}/checkpoints/{model}/full_ft/{dataset}_sam/best.pt"
            assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"
            result_path = os.path.join(experiment_folder, "full_ft", model, dataset)
            os.makedirs(result_path, exist_ok=True)
            for current_setup in ALL_SCRIPTS:
                write_batch_script(
                    env_name="super",
                    out_path=get_batch_script_names(tmp_folder),
                    inference_setup=current_setup,
                    checkpoint=checkpoint,
                    model_type=model,
                    experiment_folder=result_path,
                    dataset=dataset
                )

            # run frozen encoder
            checkpoint = f"{experiment_folder}/checkpoints/{model}/freeze_encoder/{dataset}_sam/best.pt"
            assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"
            result_path = os.path.join(experiment_folder, "freeze_encoder", model, dataset)
            os.makedirs(result_path, exist_ok=True)
            for current_setup in ALL_SCRIPTS:
                write_batch_script(
                    env_name="super",
                    out_path=get_batch_script_names(tmp_folder),
                    inference_setup=current_setup,
                    checkpoint=checkpoint,
                    model_type=model,
                    experiment_folder=result_path,
                    dataset=dataset
                )

            # run peft methods
            for peft_method, peft_kwargs in PEFT_METHODS.items():
                checkpoint = f"{experiment_folder}/checkpoints/{model}/{peft_method}/{dataset}_sam/best.pt"
                if peft_method == "qlora":
                    inference_checkpoint = f"{experiment_folder}/checkpoints/{model}/lora/{dataset}_sam/for_inference/best.pt"  # noqa
                    if not os.path.exists(inference_checkpoint):
                        os.makedirs(os.path.split(inference_checkpoint)[0], exist_ok=True)
                        export_custom_qlora_model(None, checkpoint, model, inference_checkpoint)
                        checkpoint = inference_checkpoint
                    else:
                        checkpoint = inference_checkpoint

                assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"
                result_path = os.path.join(experiment_folder, peft_method, model, dataset)
                _peft_method = 'lora' if peft_method == 'qlora' else peft_method

                os.makedirs(result_path, exist_ok=True)
                for current_setup in ALL_SCRIPTS:
                    write_batch_script(
                        env_name="super",
                        out_path=get_batch_script_names(tmp_folder),
                        inference_setup=current_setup,
                        checkpoint=checkpoint,
                        model_type=model,
                        experiment_folder=result_path,
                        dataset=dataset,
                        peft_method=_peft_method,
                        **peft_kwargs
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
