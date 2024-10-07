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
EXPERIMENT_ROOT = "/scratch/usr/nimcarot/sam/experiments/fact_dropout"


def write_batch_script(
    env_name, out_path, inference_setup, checkpoint, model_type,
    experiment_folder, peft_rank, peft_module, fact_dropout, delay=None,
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

    python_script += "-d livecell "

    if peft_rank is not None:
        python_script += f"--peft_rank {peft_rank} "
    if peft_module is not None:
        python_script += f"--peft_module {peft_module} "
    if fact_dropout is not None:
        python_script += f"--fact_dropout {fact_dropout} "
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


def run_batch_script(model_type, checkpoint, experiment_folder, peft_module=None, peft_rank=None, fact_dropout=None,
                     scripts=ALL_SCRIPTS):
    tmp_folder = "./gpu_jobs"
    shutil.rmtree(tmp_folder, ignore_errors=True)

    for current_setup in scripts:
        write_batch_script(
            inference_setup=current_setup,
            env_name="mobilesam" if model_type == "vit_t" else "sam",
            checkpoint=checkpoint,
            model_type=model_type,
            experiment_folder=experiment_folder,
            out_path=get_batch_script_names(tmp_folder),
            peft_module=peft_module,
            peft_rank=peft_rank,
            fact_dropout=fact_dropout
        )

    # the logic below automates the process of first running the precomputation of embeddings, and only then inference.
    job_id = []
    for i, my_script in enumerate(sorted(glob(tmp_folder + "/*"))):
        cmd = ["sbatch", my_script]

        if i > 0:
            cmd.insert(1, f"--dependency=afterany:{job_id[0]}")

        cmd_out = subprocess.run(cmd, capture_output=True, text=True)
        print(cmd_out.stdout if len(cmd_out.stdout) > 1 else cmd_out.stderr)

        if i == 0:
            job_id.append(re.findall(r'\d+', cmd_out.stdout)[0])


def main(args):
    # find all checkpoints in the experiment directory and run all evaluation scripts for each of them
    all_checkpoint_paths = glob(os.path.join(args.experiment_folder, "**", "best.pt"), recursive=True)

    for checkpoint_path in all_checkpoint_paths:
        # the checkpoints all have the format checkpoints/<model_type>/<training_modality>/<dataset_name>_sam/best.pt

        training_modality = checkpoint_path.split("/")[-3]
        model_type = checkpoint_path.split("/")[-4]
        experiment_folder = os.path.join(EXPERIMENT_ROOT, training_modality, "lm", "livecell", model_type)

        if "lora" in training_modality or "fact" in training_modality:
            peft_module = training_modality.split("_")[0]
            peft_rank = training_modality.split("_")[1]
            fact_dropout = training_modality.split("_")[2]

        else:
            peft_module = None
            peft_rank = None
            fact_dropout = None

        run_batch_script(
            model_type=model_type,
            checkpoint=checkpoint_path,
            experiment_folder=experiment_folder,
            peft_module=peft_module,
            peft_rank=peft_rank,
            fact_dropout=fact_dropout,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="/scratch/usr/nimcarot/data")
    parser.add_argument("-e", "--experiment_folder", type=str, default=EXPERIMENT_ROOT)
    args = parser.parse_args()
    main(args)
