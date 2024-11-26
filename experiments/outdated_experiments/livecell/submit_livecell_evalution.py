import re
import os
import shutil
import subprocess
from glob import glob
from datetime import datetime


ALL_SCRIPTS = [
    "precompute_embeddings", "evaluate_amg", "iterative_prompting", "evaluate_instance_segmentation"
]
# replace with experiment folder
EXPERIMENT_ROOT = "/scratch/usr/nimcarot/sam/experiments/livecell_peft"


def write_batch_script(
    env_name, out_path, inference_setup, checkpoint, model_type,
    experiment_folder, delay=None, use_masks=False, peft_method=None, freeze=None, peft_rank=None
):
    "Writing scripts with different fold-trainings for micro-sam evaluation"
    batch_script = f"""#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 1-00:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH --constraint=80gb
#SBATCH --job-name={inference_setup}

source ~/.bashrc
conda activate {env_name} \n"""

    if delay is not None:
        batch_script += f"sleep {delay} \n"

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
    python_script += "-d livecell "

    if peft_rank is not None:
        python_script += f"--peft_rank {peft_rank} "
    if peft_method is not None:
        python_script += f"--peft_module {peft_method} "

    # let's add the python script to the bash script
    batch_script += python_script
    if inference_setup == "precompute_embeddings":
        print(batch_script)
    with open(_op, "w") as f:
        f.write(batch_script)

    # we run the first prompt for iterative once starting with point, and then starting with box (below)
    if inference_setup == "iterative_prompting":
        batch_script += "--box "

        new_path = out_path[:-3] + f"_{inference_setup}_box.sh"
        with open(new_path, "w") as f:
            f.write(batch_script)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "micro-sam-inference"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def submit_slurm():
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"
    make_delay = "10s"  # wait for precomputing the embeddings and later run inference scripts

    all_checkpoints = glob(os.path.join(EXPERIMENT_ROOT, "checkpoints", "**", "best.pt"), recursive=True)
    model_type = "vit_b"

    for checkpoint_path in all_checkpoints:
        # the checkpoints all have the format checkpoints/<model_type>/<training_modality>/<dataset_name>_sam/best.pt
        finetuning_method = checkpoint_path.split("/")[-3]

        result_path = os.path.join(EXPERIMENT_ROOT, finetuning_method, "lm", "livecell", "vit_b")
        os.makedirs(result_path, exist_ok=True)

        # set the peft rank and method
        # if peft is used the training modality has format <peft_method>_<rank>
        if 'lora' in finetuning_method or 'fact' in finetuning_method:
            peft_rank = finetuning_method.split("_")[-1]
            peft_method = finetuning_method.split("_")[0]
        else:
            peft_method = None
            peft_rank = None

        for current_setup in ALL_SCRIPTS:
            write_batch_script(
                env_name="sam",
                out_path=get_batch_script_names(tmp_folder),
                inference_setup=current_setup,
                checkpoint=checkpoint_path,
                model_type=model_type,
                experiment_folder=result_path,
                delay=None if current_setup == "precompute_embeddings" else make_delay,
                peft_rank=peft_rank,
                peft_method=peft_method
            )

    job_id = []
    for i, my_script in enumerate(sorted(glob(tmp_folder + "/*"))):
        cmd = ["sbatch", my_script]

        if i > 0:
            cmd.insert(1, f"--dependency=afterany:{job_id[0]}")

        cmd_out = subprocess.run(cmd, capture_output=True, text=True)
        print(cmd_out.stdout if len(cmd_out.stdout) > 1 else cmd_out.stderr)

        if i == 0:
            job_id.append(re.findall(r'\d+', cmd_out.stdout)[0])


def main():
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass
    submit_slurm()


if __name__ == "__main__":
    main()
