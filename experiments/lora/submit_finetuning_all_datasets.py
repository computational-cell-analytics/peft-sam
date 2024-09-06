import os
import shutil
import subprocess
from datetime import datetime



def write_batch_sript(
    env_name, save_root, model_type, script_name, checkpoint_path, lora_rank, dataset
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
#SBATCH --constraint=80gb
source activate {env_name}
"""

    python_script = "python lora_finetuning.py "

    # add parameters to the python script
    python_script += f"-m {model_type} "  # choice of vit
    python_script += f"-d {dataset} "  # dataset

    if checkpoint_path is not None:
        python_script += f"-c {checkpoint_path} "

    if save_root is not None:
        python_script += f"-s {save_root} "  # path to save model checkpoints and logs

    if lora_rank is not None:
        python_script += f"--lora_rank {lora_rank} "
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


def main(args):
    tmp_folder = "./gpu_jobs"
    model_type = args.model_type

    datasets = ["covid_if", "orgasegment", "gonuclear", "mitolab_glycolytic_muscle", "platy_cilia"]

    
    for data in datasets:
        for lora_rank in [None, 4]:
            roi = "em_organelles" if data in ["mitolab_glycolytic_muscle", "platy_cilia"] else "lm"
            generalist_model = f"{args.model_type}_{roi}"
            write_batch_sript(
                env_name="mobilesam" if model_type[:5] == "vit_t" else "sam",
                save_root=args.save_root,
                model_type=generalist_model,
                script_name=get_batch_script_names(tmp_folder),
                checkpoint_path=None,
                lora_rank=lora_rank,
                dataset=data
            )

if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="vit_b", help="Choice of image encoder in SAM")
    parser.add_argument("-s", "--save_root", type=str, default=None, help="Path to save checkpoints.")

    args = parser.parse_args()
    main(args)
