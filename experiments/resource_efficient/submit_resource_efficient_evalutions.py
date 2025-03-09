import os
import shutil
import subprocess
from datetime import datetime
from micro_sam.util import export_custom_qlora_model

# ALL_DATASETS = {
#    'covid_if': 'lm', 'orgasegment': 'lm', 'gonuclear': 'lm', 'mitolab_glycolytic_muscle': 'em_organelles',
#    'platy_cilia': 'em_organelles', 'hpa': 'lm', 'livecell': 'lm',
    # medical

ALL_DATASETS = {
    'motum': 'medical_imaging', 'papila': 'medical_imaging', #'jsrt': 'medical_imaging',
    'amd_sd': 'medical_imaging', 'mice_tumseg': 'medical_imaging', 'sega': 'medical_imaging',
    'ircadb': 'medical_imaging', 'dsad': 'medical_imaging', 'psfhs': 'medical_imaging'
}

PEFT_METHODS = {
    "lora": {"peft_rank": 32},
    "qlora": {"peft_rank": 32, "quantize": True}  # QLoRA
    }

ALL_SCRIPTS = [
    "evaluate_instance_segmentation", "iterative_prompting"
]
# replace with experiment folder
EXPERIMENT_ROOT = "/scratch/usr/nimcarot/sam/experiments/resource_efficient"


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
    alpha=None,
    proj_size=None,
    dropout=None,
    quantize=False,
    dry=False
):
    "Writing scripts with different fold-trainings for micro-sam evaluation"
    batch_script = f"""#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 1-00:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH --job-name={inference_setup}

source ~/.bashrc
conda activate {env_name} \n"""

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


def run_peft_evaluations(args, datasets, n_images):
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"

    for dataset, domain in datasets.items():
        gen_model = f"vit_b_{domain}"
        models = ["vit_b"] if dataset == "livecell" else ["vit_b", gen_model]
        if domain == "medical_imaging":
            SCRIPTS = ["iterative_prompting"]
        else:
            SCRIPTS = ALL_SCRIPTS
        for model in models:

            # run generalist / vanilla
            # checkpoint = "/scratch/usr/nimcarot/sam/models/vit_b_lm.pt" if model == "vit_b_lm" else None
            # result_path = os.path.join(EXPERIMENT_ROOT, "vanilla", model, f"{n_images}_imgs", dataset)
            # if not os.path.exists(result_path):
            #     os.makedirs(result_path, exist_ok=False)
            #     for current_setup in SCRIPTS:
            #         write_batch_script(
            #             env_name="peft-sam-qlora",
            #             out_path=get_batch_script_names(tmp_folder),
            #             inference_setup=current_setup,
            #             checkpoint=checkpoint,
            #             model_type=model,
            #             experiment_folder=result_path,
            #             dataset=dataset,
            #             dry=args.dry
            #         )

            # full finetuning
            checkpoint = f"{EXPERIMENT_ROOT}/checkpoints/{model}/full_ft/{n_images}_imgs/{dataset}_sam/best.pt"
            assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"
            result_path = os.path.join(EXPERIMENT_ROOT, "full_ft", model, f"{n_images}_imgs", dataset)
            if not os.path.exists(result_path):
                os.makedirs(result_path, exist_ok=False)
                for current_setup in SCRIPTS:
                    write_batch_script(
                        env_name="peft-sam-qlora",
                        out_path=get_batch_script_names(tmp_folder),
                        inference_setup=current_setup,
                        checkpoint=checkpoint,
                        model_type=model,
                        experiment_folder=result_path,
                        dataset=dataset,
                        dry=args.dry
                    )
            # freeze the encoder
            checkpoint = f"{EXPERIMENT_ROOT}/checkpoints/{model}/freeze_encoder/{n_images}_imgs/{dataset}_sam/best.pt"
            assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"
            result_path = os.path.join(EXPERIMENT_ROOT, "freeze_encoder", model, f"{n_images}_imgs", dataset)
            if not os.path.exists(result_path):
                os.makedirs(result_path, exist_ok=False)
                for current_setup in SCRIPTS:
                    write_batch_script(
                        env_name="peft-sam-qlora",
                        out_path=get_batch_script_names(tmp_folder),
                        inference_setup=current_setup,
                        checkpoint=checkpoint,
                        model_type=model,
                        experiment_folder=result_path,
                        dataset=dataset,
                        dry=args.dry
                    )
            # run peft methods
            for peft_method, peft_kwargs in PEFT_METHODS.items():
                checkpoint = f"{EXPERIMENT_ROOT}/checkpoints/{model}/{peft_method}/{n_images}/{dataset}_sam/best.pt"

                # Export custom QLoRA model
                if peft_method == "qlora":
                    inference_checkpoint = f"{EXPERIMENT_ROOT}/checkpoints/{model}/lora/{n_images}_imgs/{dataset}_sam/for_inference/best.pt"
                    os.makedirs(os.path.split(inference_checkpoint)[0], exist_ok=True)
                    export_custom_qlora_model(None, checkpoint, model, inference_checkpoint)
                    checkpoint = inference_checkpoint

                assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"
                result_path = os.path.join(EXPERIMENT_ROOT, peft_method, model, f"{n_images}_img", dataset)
                if os.path.exists(result_path):
                    continue
                os.makedirs(result_path, exist_ok=False)

                _peft_method = 'lora' if peft_method == 'qlora' else peft_method
                for current_setup in SCRIPTS:
                    write_batch_script(
                        env_name="peft-sam-qlora",
                        out_path=get_batch_script_names(tmp_folder),
                        inference_setup=current_setup,
                        checkpoint=checkpoint,
                        model_type=model,
                        experiment_folder=result_path,
                        dataset=dataset,
                        peft_method=_peft_method,
                        **peft_kwargs,
                        dry=args.dry
                    )


def main(args):

    if args.single_img:
        run_peft_evaluations(args, ALL_DATASETS, n_images=1)
    elif args.data_scaling:
        n_images = [2, 5, 10]
        datasets = {"hpa": "lm", "psfhs": "medical_imaging"}
        for n in n_images:
            run_peft_evaluations(args, datasets, n_images=n)


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    # Set up argument parsing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single_img",
        action="store_true",
        help="Run single image finetuning."
    )
    parser.add_argument(
        "--data_scaling",
        action="store_true",
        help="Run data scaling experiments."
    )
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args)
