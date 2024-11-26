import os
import shutil
import subprocess
from datetime import datetime

ROOT = "/scratch/usr/nimcarot/sam/experiments/livecell_peft"


def write_batch_script(
    env_name, save_root, model_type, script_name, checkpoint_path, peft_method, peft_rank, freeze
):
    assert model_type in ["vit_t", "vit_b", "vit_t_lm", "vit_b_lm", "vit_b_em_organelles"]

    "Writing scripts for finetuning with and without lora on different light and electron microscopy datasets"

    batch_script = f"""#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -p grete:shared
#SBATCH -t 2-00:00:00
#SBATCH -G A100:1
#SBATCH -A nimcarot
#SBATCH --constraint=80gb
source activate {env_name}
"""

    python_script = "python ../finetuning.py "

    # add parameters to the python script
    python_script += f"-m {model_type} "  # choice of vit

    if checkpoint_path is not None:
        python_script += f"-c {checkpoint_path} "

    if save_root is not None:
        python_script += f"-s {save_root} "  # path to save model checkpoints and logs

    # let's add the peft arguments
    if peft_method is not None:
        python_script += f"--peft_method {peft_method} "
    if peft_rank is not None:
        python_script += f"--peft_rank {peft_rank} "
    if freeze is not None:
        python_script += f"--freeze {freeze} "
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


def main():
    tmp_folder = "./gpu_jobs"
    model_type = "vit_b"

    # Full Finetuning
    write_batch_script(
        env_name="mobilesam" if model_type[:5] == "vit_t" else "sam",
        save_root=ROOT,
        model_type=model_type,
        script_name=get_batch_script_names(tmp_folder),
        checkpoint_path=None,
        freeze=None,
        peft_method=None,
        peft_rank=None
    )

    # Freeze Encoder
    write_batch_script(
        env_name="mobilesam" if model_type[:5] == "vit_t" else "sam",
        save_root=ROOT,
        model_type=model_type,
        script_name=get_batch_script_names(tmp_folder),
        checkpoint_path=None,
        freeze="image_encoder",
        peft_method=None,
        peft_rank=None
    )

    # LoRA
    write_batch_script(
        env_name="mobilesam" if model_type[:5] == "vit_t" else "sam",
        save_root=ROOT,
        model_type=model_type,
        script_name=get_batch_script_names(tmp_folder),
        checkpoint_path=None,
        freeze=None,
        peft_method="lora",
        peft_rank=4
    )

    # FacT
    write_batch_script(
        env_name="mobilesam" if model_type[:5] == "vit_t" else "sam",
        save_root=ROOT,
        model_type=model_type,
        script_name=get_batch_script_names(tmp_folder),
        checkpoint_path=None,
        freeze=None,
        peft_method="fact",
        peft_rank=4
    )


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    main()
