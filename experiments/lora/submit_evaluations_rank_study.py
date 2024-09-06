import re
import os
import shutil
import subprocess
from glob import glob
from pathlib import Path
from datetime import datetime

from peft_sam.preprocess_datasets import preprocess_data

# replace with experiment folder
ROOT = "/scratch/usr/nimcarot/sam/experiments/"

ALL_SCRIPTS = [
    "precompute_embeddings", "evaluate_amg", "iterative_prompting", "evaluate_instance_segmentation"
]


def write_batch_script(
    env_name, out_path, inference_setup, checkpoint, model_type,
    experiment_folder, dataset_name, delay=None, use_masks=False, lora_rank=None
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

    if checkpoint is not None:# add the finetuned checkpoint
        python_script += f"-c {checkpoint} "

    # name of the model configuration
    python_script += f"-m {model_type} "

    # experiment folder
    python_script += f"-e {experiment_folder} "

    # IMPORTANT: choice of the dataset
    if dataset_name == "platy_cilia":
        dataset_name = "platynereis/cilia"
    elif dataset_name == "mitolab_glycolytic_muscle":
        dataset_name = "mitolab/glycolytic_muscle"
    python_script += f"-d {dataset_name} "

    if lora_rank is not None:
        python_script += f"--lora_rank {lora_rank} "

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


def get_checkpoint_path(experiment_set, dataset_name, model_type, region):
    # let's set the experiment type - either using the generalist or just using vanilla model
    if experiment_set == "generalist":
        checkpoint = None
        # set new model_type to vit_b_em_organelles or vit_b_lm
        model_type = f"{model_type}_{region}"

    elif experiment_set == "full_ft" or "lora" in experiment_set:
        
        _split = dataset_name.split("/")
        if len(_split) > 1:
            # it's the case for plantseg/root, we catch it and convert it to the expected format
            dataset_name = f"{_split[0]}_{_split[1]}"

        if dataset_name.startswith("mitolab"):
            dataset_name = "mitolab_glycolytic_muscle"

        checkpoint = f"{ROOT}/checkpoints/{model_type}/{dataset_name}_{experiment_set}/best.pt"

    elif experiment_set == "vanilla":
        checkpoint = None

    else:
        raise ValueError("Choose from generalist / vanilla")

    if checkpoint is not None:
        assert os.path.exists(checkpoint), checkpoint

    return checkpoint


def submit_slurm(model_type="vit_b"):
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"
    make_delay = "10s"  # wait for precomputing the embeddings and later run inference scripts

    # env name
    if model_type == "vit_t":
        env_name = "mobilesam"
    else:
        env_name = "sam"

    for dataset_name in ['orgasegment', 'mitolab_glycolytic_muscle']:
        preprocess_data(dataset_name)
        for base_model in ['vanilla', 'generalist']:
            for experiment_set in ["full_ft", "lora_1", "lora_2", "lora_4", "lora_8", "lora_16", "lora_32", "lora_64"]:
                # get the rank if finetuned with lora
                if "lora" in experiment_set:
                    lora_rank = experiment_set.split('-')[-1].split('_')[-1]
                else: 
                    lora_rank = None
            
                # chose the region and model_type if training was from generalist
                region = "em_organelles" if dataset_name == "mitolab_glycolytic_muscle" else "lm"
                model_type_roi = f"{model_type}_{region}" if base_model == "generalist" else model_type

                # get checkpoint path
                checkpoint = get_checkpoint_path(experiment_set, dataset_name, model_type_roi, region)

                # get experiment path
                experiment_folder = f"{ROOT}/{experiment_set}/{dataset_name}/{model_type_roi}/"

                # make specifications for scripts
                if experiment_set == "vanilla":
                    all_setups = ALL_SCRIPTS[:-1]
                else:
                    all_setups = ALL_SCRIPTS

                for current_setup in all_setups:
                    write_batch_script(
                        env_name=env_name,
                        out_path=get_batch_script_names(tmp_folder),
                        inference_setup=current_setup,
                        checkpoint=checkpoint,
                        model_type=model_type_roi,
                        experiment_folder=experiment_folder,
                        dataset_name=dataset_name,
                        delay=None if current_setup == "precompute_embeddings" else make_delay,
                        lora_rank=lora_rank
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
    


def main():
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass
    submit_slurm()



if __name__ == "__main__":
    main()

