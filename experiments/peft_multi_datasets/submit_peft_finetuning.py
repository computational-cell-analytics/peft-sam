import os
import shutil
import subprocess
from datetime import datetime


ALL_DATASETS = {
    # LM DATASETS
    'covid_if': 'lm',
    'orgasegment': 'lm',
    'gonuclear': 'lm',
    'hpa': 'lm',
    'livecell': 'lm',

    # EM DATASETS
    'mitolab_glycolytic_muscle': 'em_organelles',
    'platy_cilia': 'em_organelles',

    # MEDICAL IMAGING DATASETS
    'papila': 'medical_imaging',
    'motum': 'medical_imaging',
    'psfhs': 'medical_imaging',
    'jsrt': 'medical_imaging',
    'amd_sd': 'medical_imaging',
    'mice_tumseg': 'medical_imaging',
    'sega': 'medical_imaging',
    'dsad': 'medical_imaging',
    'ircadb': 'medical_imaging',
}

# Dictionary with all peft methods and their peft kwargs
PEFT_METHODS = {
    "lora": {"peft_rank": 32},
    "adaptformer": {"peft_rank": 2, "alpha": "learnable_scalar", "dropout": None, "projection_size": 64},
    "ssf": {"peft_rank": 2},
    "fact": {"peft_rank": 16, "dropout": 0.1},
    "AttentionSurgery": {"peft_rank": 2},
    "BiasSurgery": {"peft_rank": 2},
    "LayerNormSurgery": {"peft_rank": 2},
    "qlora": {"peft_rank": 32, "quantize": True},
}


def write_batch_script(
    env_name,
    save_root,
    model_type,
    script_name,
    checkpoint_path,
    checkpoint_name=None,
    dataset="livecell",
    peft_rank=None,
    peft_method=None,
    alpha=None,
    dropout=None,
    learning_rate=1e-5,
    projection_size=None,
    freeze=None,
    quantize=False,
    dry=False,
):
    if model_type not in ["vit_b", "vit_l", "vit_h", "vit_b_lm", "vit_b_em_organelles", "vit_b_medical_imaging"]:
        raise ValueError(model_type)

    "Writing scripts for finetuning with and without lora on different light and electron microscopy datasets"

    batch_script = f"""#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -p grete:shared
#SBATCH -t 2-00:00:00
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH --constraint=80gb
#SBATCH -x ggpu212
#SBATCH --job-name=finetune-sam

source ~/.bashrc
micromamba activate {env_name}
"""

    python_script = "python ../finetuning.py "

    # add parameters to the python script
    python_script += f"-m {model_type} "  # choice of vit
    python_script += f"-d {dataset} "  # dataset

    if checkpoint_path is not None:
        python_script += f"-c {checkpoint_path} "
    if checkpoint_name is not None:
        python_script += f"--checkpoint_name {checkpoint_name} "

    if save_root is not None:
        python_script += f"-s {save_root} "  # path to save model checkpoints and logs

    if peft_rank is not None:
        python_script += f"--peft_rank {peft_rank} "
    if peft_method is not None:
        python_script += f"--peft_method {peft_method} "
    if alpha is not None:
        python_script += f"--alpha {alpha} "
    if learning_rate is not None:
        python_script += f"--learning_rate {learning_rate} "
    if projection_size is not None:
        python_script += f"--projection_size {projection_size} "
    if dropout is not None:
        python_script += f"--dropout {dropout} "
    if freeze is not None:
        python_script += f"--freeze {freeze} "
    if quantize:
        python_script += "--quantize "

    medical_datasets = ['papila', 'motum', 'psfhs', 'jsrt', 'amd_sd', 'mice_tumseg', 'sega', 'ircadb', 'dsad']
    if dataset in medical_datasets:
        python_script += "--medical_imaging "

    # let's add the python script to the bash script
    batch_script += python_script
    with open(script_name, "w") as f:
        f.write(batch_script)

    if not dry:
        cmd = ["sbatch", script_name]
        subprocess.run(cmd)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "micro-sam-finetuning"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def ckpt_exists(ckpt_name, args):
    checkpoint_path = os.path.join(args.save_root, "checkpoints", ckpt_name, "best.pt")
    return os.path.exists(checkpoint_path)


def run_peft_finetuning(args):
    for dataset, domain in ALL_DATASETS.items():
        if args.dataset is not None and args.dataset != dataset:
            continue

        model_type = args.model
        gen_model = f"{model_type}_{domain}"
        models = [model_type] if dataset == "livecell" else [model_type, gen_model]
        for model in models:
            # full finetuning
            checkpoint_name = f"{model}/full_ft/{dataset}_sam"
            if not ckpt_exists(checkpoint_name, args):
                script_name = get_batch_script_names("./gpu_jobs")
                write_batch_script(
                    env_name="super",
                    save_root=args.save_root,
                    model_type=model,
                    script_name=script_name,
                    checkpoint_path=None,
                    checkpoint_name=checkpoint_name,
                    dataset=dataset,
                    dry=args.dry,
                )

            # freeze the encoder
            checkpoint_name = f"{model}/freeze_encoder/{dataset}_sam"
            if not ckpt_exists(checkpoint_name, args):
                script_name = get_batch_script_names("./gpu_jobs")
                write_batch_script(
                    env_name="super",
                    save_root=args.save_root,
                    model_type=model,
                    script_name=script_name,
                    checkpoint_path=None,
                    checkpoint_name=checkpoint_name,
                    dataset=dataset,
                    freeze='image_encoder',
                    dry=args.dry,
                )
            # peft methods
            for peft_method, peft_kwargs in PEFT_METHODS.items():
                _peft_method = 'lora' if peft_method == 'qlora' else peft_method
                script_name = get_batch_script_names("./gpu_jobs")
                checkpoint_name = f"{model}/{peft_method}/{dataset}_sam"
                if ckpt_exists(checkpoint_name, args):
                    continue

                write_batch_script(
                    env_name="peft-sam" if peft_method == "qlora" else "super",
                    save_root=args.save_root,
                    model_type=model,
                    script_name=script_name,
                    checkpoint_path=None,
                    peft_method=_peft_method,
                    checkpoint_name=checkpoint_name,
                    dataset=dataset,
                    dry=args.dry,
                    **peft_kwargs
                )


def main(args):
    run_peft_finetuning(args)


if __name__ == "__main__":
    tmp_folder = "./gpu_jobs"
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

    # Set up parsing arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--save_root", type=str, default="/mnt/vast-nhr/projects/cidas/cca/experiments/peft_sam/models",
        help="Path to the directory where the model checkpoints are stored."
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default=None,
        help="Run the experiments for a specific supported biomedical imaging dataset."
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="The model type to initialize the predictor")
    parser.add_argument("--dry", action="store_true")

    args = parser.parse_args()
    main(args)
