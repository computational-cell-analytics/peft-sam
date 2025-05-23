import os
import shutil
import subprocess
from datetime import datetime
import itertools

DATASETS = {
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


def write_batch_script(
    env_name,
    save_root,
    model_type,
    script_name,
    checkpoint_path,
    checkpoint_name=None,
    dataset="livecell",
    peft_method=None,
    peft_rank=None,
    attention_layers_to_update=[],
    update_matrices=[],
    freeze=None,
    quantize=False,
    dry=False,
):
    assert model_type in ["vit_b", "vit_l", "vit_h", "vit_b_lm", "vit_b_em_organelles", "vit_b_medical_imaging"]

    "Writing scripts for finetuning with and without lora on different light and electron microscopy datasets"

    batch_script = f"""#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -p grete:shared
#SBATCH -t 1-00:00:00
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH --constraint=80gb
#SBATCH --job-name=finetune-sam
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=carolin.teuber@stud.uni-goettingen.de

source ~/.bashrc
mamba activate {env_name}
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
    if attention_layers_to_update:
        python_script += "--attention_layers_to_update "
        for layer in attention_layers_to_update:
            python_script += f"{layer} "

    if update_matrices:
        python_script += "--update_matrices "
        for matrix in update_matrices:
            python_script += f"{matrix} "
    if freeze is not None:
        python_script += f"--freeze {freeze} "
    if quantize:
        python_script += f"--quantize "

    medical_datasets = ['papila', 'motum', 'psfhs', 'jsrt', 'amd_sd', 'mice_tumseg', 'sega', 'dsad', 'ircadb']
    if dataset in medical_datasets:
        python_script += "--medical_imaging "

    python_script += "-i /scratch/usr/nimcarot/data"

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


def run_late_lora_finetuning(args):

    # get experiment settings
    datasets, peft_methods, attention_layers_to_update, update_matrices = get_experiment_setting(args)

    for dataset in datasets:
        if args.dataset is not None and dataset != args.dataset:
            continue
        domain = DATASETS[dataset]
        model = f"vit_b_{domain}"
        if model == "vit_b_lm":
            # make sure to use old sam models for consistency
            checkpoint_path = f"/scratch/usr/nimcarot/sam/models/vit_b_{domain}.pt"
        else:
            checkpoint_path = None

        if not args.specific:
            # freeze encoder
            checkpoint_name = f"{model}/late_lora/ClassicalSurgery/standard/start_12/{dataset}_sam/"
            write_batch_script(
                env_name="peft-sam-gpu",
                save_root=args.save_root,
                model_type=model,
                checkpoint_name=checkpoint_name,
                script_name=get_batch_script_names("./gpu_jobs"),
                checkpoint_path=checkpoint_path,
                dataset=dataset,
                freeze="image_encoder",
                dry=args.dry,
            )
        for method, layers, update_matrix in itertools.product(peft_methods, attention_layers_to_update,
                                                               update_matrices.keys()):

            checkpoint_name = f"{model}/late_lora/{method}/{update_matrix}/start_{layers[0]}/{dataset}_sam/"
            script_name = get_batch_script_names("./gpu_jobs")

            if ckpt_exists(checkpoint_name, args):
                continue

            if method == 'qlora':
                _method = 'lora'
                quantize = True
                env_name = "peft-sam-qlora"
            else:
                _method = method
                quantize = False
                env_name = "peft-sam-qlora"
            
            write_batch_script(
                env_name=env_name,
                save_root=args.save_root,
                model_type=model,
                script_name=script_name,
                checkpoint_path=checkpoint_path,
                checkpoint_name=checkpoint_name,
                dataset=dataset,
                peft_method=_method,
                peft_rank=32,
                attention_layers_to_update=layers,
                update_matrices=update_matrices[update_matrix],
                quantize=quantize,
                dry=args.dry,
            )

def get_experiment_setting(args):

    if args.best_setting:
        datasets = DATASETS.keys()
        methods = ["lora", "ClassicalSurgery", "qlora"]
        attention_layers_to_update = [[6, 7, 8, 9, 10, 11]]
        update_matrices = {'all_matrices': ["q", "v", "k", "mlp"]}
    elif args.qlora_only:
        datasets = ["hpa", "psfhs"]
        methods = ["qlora"]
        attention_layers_to_update = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11], [9, 10, 11], [11]]
        update_matrices = {'standard': ["q", "v"], 'all_matrices': ["q", "k", "v", "mlp"]}
    elif args.specific:
        # for table 6-7
        datasets = ["livecell"]
        methods = ["lora", "qlora"]
        attention_layers_to_update = [[6, 7, 8, 9, 10, 11]]
        update_matrices = {'all_matrices': ["q", "k", "v", "mlp"]}

    else:
        datasets = DATASETS.keys()
        update_matrices = {'standard': ["q", "v"], 'all_matrices': ["q", "k", "v", "mlp"]}
        attention_layers_to_update = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11], [9, 10, 11], [11]]
        methods = ["lora", "ClassicalSurgery"]

    return datasets, methods, attention_layers_to_update, update_matrices



def main(args):
    run_late_lora_finetuning(args)


if __name__ == "__main__":
    tmp_folder = "./gpu_jobs"
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

    # Set up parsing arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--save_root", type=str, default="/scratch/usr/nimcarot/sam/experiments/peft",
        help="Path to the directory where the model checkpoints are stored."
    )
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to use for finetuning.")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--best_setting", action="store_true", help="Use the only the best late lora setting for finetuning.")
    parser.add_argument("--qlora_only", action="store_true", help="Use only qlora for finetuning.")
    parser.add_argument("--specific", action="store_true", help="Use only the specific setting for finetuning.")

    args = parser.parse_args()
    main(args)
