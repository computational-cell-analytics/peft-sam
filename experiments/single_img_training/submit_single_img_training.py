import os
import shutil
import subprocess
from datetime import datetime

ALL_DATASETS = {'covid_if': 'lm', 'orgasegment': 'lm', 'gonuclear': 'lm', 'mitolab_glycolytic_muscle': 'em_organelles',
                'platy_cilia': 'em_organelles', 'hpa': 'lm', 'livecell': 'lm'}

# Dictionary with all peft methods and their peft kwargs
PEFT_METHODS = {
    "lora": {"peft_rank": 32},
    "qlora": {"peft_rank": 32, "quantize": True}  # QLoRA
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
    quantize=False
):
    assert model_type in ["vit_t", "vit_b", "vit_t_lm", "vit_b_lm", "vit_b_em_organelles"]

    "Writing scripts for finetuning with and without lora on different light and electron microscopy datasets"

    batch_script = f"""#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -p grete:shared
#SBATCH -t 2-00:00:00
#SBATCH -G A100:1
#SBATCH -A nim00007
source activate {env_name}
"""

    python_script = "python single_img_finetuning.py "

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
    # let's add the python script to the bash script
    batch_script += python_script
    print(batch_script)
    with open(script_name, "w") as f:
        f.write(batch_script)

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


def cpkt_exists(cpkt_name, args):
    checkpoint_path = os.path.join(args.save_root, "checkpoints", cpkt_name, "best.pt")
    return os.path.exists(checkpoint_path)


def run_peft_finetuning(args):
    for dataset, domain in ALL_DATASETS.items():
        gen_model = f"vit_b_{domain}"
        models = ["vit_b"] if dataset == "livecell" else ["vit_b", gen_model]
        for model in models:
            # full finetuning
            checkpoint_name = f"{model}/full_ft/{dataset}_sam"
            if not cpkt_exists(checkpoint_name, args):
                script_name = get_batch_script_names("./gpu_jobs")
                write_batch_script(
                    env_name="peft-sam",
                    save_root=args.save_root,
                    model_type=model,
                    script_name=script_name,
                    checkpoint_path=None,
                    checkpoint_name=checkpoint_name,
                    dataset=dataset
                )
            # freeze the encoder
            checkpoint_name = f"{model}/freeze_encoder/{dataset}_sam"
            if not cpkt_exists(checkpoint_name, args):
                script_name = get_batch_script_names("./gpu_jobs")
                write_batch_script(
                    env_name="peft-sam",
                    save_root=args.save_root,
                    model_type=model,
                    script_name=script_name,
                    checkpoint_path=None,
                    checkpoint_name=checkpoint_name,
                    dataset=dataset,
                    freeze='image_encoder',
                )
            # peft methods
            for peft_method, peft_kwargs in PEFT_METHODS.items():
                # for now: run only for lora
                script_name = get_batch_script_names("./gpu_jobs")
                checkpoint_name = f"{model}/{peft_method}/{dataset}_sam"
                _peft_method = "lora" if peft_method == "qlora" else peft_method
                if cpkt_exists(checkpoint_name, args):
                    continue
                write_batch_script(
                    env_name="peft-sam",
                    save_root=args.save_root,
                    model_type=model,
                    script_name=script_name,
                    checkpoint_path=None,
                    peft_method=_peft_method,
                    checkpoint_name=checkpoint_name,
                    dataset=dataset,
                    **peft_kwargs
                )


def main(args):
    run_peft_finetuning(args)


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    # Set up argument parsing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--save_root",
        type=str,
        default="/scratch/usr/nimcarot/sam/experiments/single_img_training",
        help="Path to save checkpoints."
    )

    args = parser.parse_args()
    main(args)
