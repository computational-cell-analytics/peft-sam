import os
import shutil
import subprocess
import itertools
from datetime import datetime
from peft_sam.dataset.preprocess_datasets import preprocess_data


ALL_SCRIPTS = [
    "evaluate_instance_segmentation", "iterative_prompting"
]
# replace with experiment folder
EXPERIMENT_ROOT = "/scratch/usr/nimcarot/sam/experiments/peft_param_search"


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
    dropout=None
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

    # let's add the python script to the bash script
    batch_script += python_script
    with open(_op, "w") as f:
        f.write(batch_script)

    cmd = ["sbatch", _op]
    subprocess.run(cmd)
    # we run the first prompt for iterative once starting with point, and then starting with box (below)
    if inference_setup == "iterative_prompting":
        batch_script += "--box "

        new_path = out_path[:-3] + f"_{inference_setup}_box.sh"
        with open(new_path, "w") as f:
            f.write(batch_script)

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


def run_lora_eval_a():
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"

    alphas = [1, 2, 4]
    ranks = [1, 2, 4, 8, 16, 32, 64]
    for alpha, rank in itertools.product(alphas, ranks):
        checkpoint_name = os.path.join('vit_b_lm', 'lora', 'lr_1e-05', f'rank_{rank}', f'alpha_{alpha}')
        checkpoint_path = os.path.join(EXPERIMENT_ROOT, checkpoint_name, 'orgasegment_sam', 'best.pt')
        result_path = os.path.join(EXPERIMENT_ROOT, "lora", "orgasegment", "vit_b_lm", "lr_1e-5", f"rank_{rank}", f"alpha_{alpha}")

        if os.path.exits(result_path):
            print("Warning: The result path already exists")

        os.makedirs(result_path, exist_ok=True)
        for current_setup in ALL_SCRIPTS:
            write_batch_script(
                env_name="sam",
                out_path=get_batch_script_names(tmp_folder),
                inference_setup=current_setup,
                checkpoint=checkpoint_path,
                model_type="vit_b_lm",
                experiment_folder=result_path,
                dataset="orgasegment",
                peft_rank=rank,
                peft_method="lora",
                alpha=alpha
            )


def run_lora_eval_b():
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"

    alphas = [1, 2, 4]
    ranks = [1, 32]
    lrs = [1e-3, 5e-4, 1e-4, 5e-5]

    for alpha, rank, lr in itertools.product(alphas, ranks, lrs):

        checkpoint_name = os.path.join('vit_b_lm', 'lora', f'lr_{lr}', f'rank_{rank}', f'alpha_{alpha}')
        checkpoint_path = os.path.join(EXPERIMENT_ROOT, checkpoint_name, 'orgasegment_sam', 'best.pt')
        result_path = os.path.join(EXPERIMENT_ROOT, "lora", "orgasegment", "vit_b_lm", f"lr_{lr}", f"rank_{rank}", f"alpha_{alpha}")

        if os.path.exists(result_path):
            print("Warning: The result path already exists")

        os.makedirs(result_path, exist_ok=True)

        for current_setup in ALL_SCRIPTS:
            write_batch_script(
                env_name="sam",
                out_path=get_batch_script_names(tmp_folder),
                inference_setup=current_setup,
                checkpoint=checkpoint_path,
                model_type="vit_b_lm",
                experiment_folder=result_path,
                datset="orgasegment",
                peft_rank=rank,
                peft_method="lora",
                alpha=alpha
            )


def run_lora_eval_c():
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"

    alphas = [0.1, 0.25, 0.5, 0.75, 8]
    rank = 32
    lr = 1e-5

    for alpha in alphas:
        checkpoint_name = os.path.join('vit_b_lm', 'lora', f'lr_{lr}', f'rank_{rank}', f'alpha_{alpha}')
        checkpoint_path = os.path.join(EXPERIMENT_ROOT, checkpoint_name, 'orgasegment_sam', 'best.pt')
        result_path = os.path.join(EXPERIMENT_ROOT, "lora", "orgasegment", "vit_b_lm", f"lr_{lr}", f"rank_{rank}", f"alpha_{alpha}")

        if os.path.exists(result_path):
            print("Warning: The result path already exists")

        os.makedirs(result_path, exist_ok=True)

        for current_setup in ALL_SCRIPTS:
            write_batch_script(
                env_name="sam",
                out_path=get_batch_script_names(tmp_folder),
                inference_setup=current_setup,
                checkpoint=checkpoint_path,
                model_type="vit_b_lm",
                experiment_folder=result_path,
                datset="orgasegment",
                peft_rank=rank,
                peft_method="lora",
                alpha=alpha
            )


def run_lora_eval_d():

    tmp_folder = "./gpu_jobs"

    alphas = [0.1, 1, 8]
    rank = 32
    lr = 1e-5
    datasets = ["orgasegment", "covid_if", "mitolab_glycolytic_muscle", "platy_cilia"]

    for dataset, alpha in itertools.product(datasets, alphas):
        preprocess_data(dataset)
        if dataset in ["orgasegment", "covid_if"]:
            generalist = "vit_b_lm"
        else:
            generalist = "vit_b_em_organelles"

        for model in ["vit_b", generalist]:
            checkpoint_name = os.path.join(f'{model}', 'lora', f'lr_{lr}', f'rank_{rank}', f'alpha_{alpha}')
            checkpoint_path = os.path.join(EXPERIMENT_ROOT, 'checkpoints', checkpoint_name, f'{dataset}_sam', 'best.pt')

            assert os.path.exists(checkpoint_path), "Wrong checkpoint path"

            result_path = os.path.join(EXPERIMENT_ROOT, "lora", f"{dataset}", f"{model}", f"lr_{lr}", f"rank_{rank}", f"alpha_{alpha}")

            if os.path.exists(result_path):
                print("Warning: The result path already exists")
                continue

            os.makedirs(result_path, exist_ok=True)

            for current_setup in ALL_SCRIPTS:
                write_batch_script(
                    env_name="sam",
                    out_path=get_batch_script_names(tmp_folder),
                    inference_setup=current_setup,
                    checkpoint=checkpoint_path,
                    model_type="vit_b_lm",
                    experiment_folder=result_path,
                    dataset=dataset,
                    peft_rank=rank,
                    peft_method="lora",
                    alpha=alpha
                )


def run_adaptformer_eval():
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"

    alphas = [0.1, 0.5, 1, 'learnable_scalar']
    proj_sizes = [64, 128, 256]
    dropouts = [0.1, 0.25, 0.5, None]

    for alpha, proj_size, dropout in itertools.product(alphas, proj_sizes, dropouts):
        model = "vit_b_lm"
        peft_method = "adaptformer"
        rank = 1
        checkpoint_name = f"{model}/{peft_method}/alpha_{alpha}/projection_size_{proj_size}/dropout_{dropout}"
        checkpoint_path = os.path.join(EXPERIMENT_ROOT, "checkpoints", checkpoint_name, 'orgasegment_sam', 'best.pt')

        assert os.path.exists(checkpoint_path), checkpoint_path

        result_path = os.path.join(
            EXPERIMENT_ROOT, "adaptformer", "orgasegment", f"alpha_{alpha}", f"projection_size_{proj_size}", f"dropout_{dropout}"
        )

        os.makedirs(result_path, exist_ok=True)

        for current_setup in ALL_SCRIPTS:
            write_batch_script(
                env_name="sam",
                out_path=get_batch_script_names(tmp_folder),
                inference_setup=current_setup,
                checkpoint=checkpoint_path,
                model_type="vit_b_lm",
                experiment_folder=result_path,
                dataset="orgasegment",
                peft_rank=rank,
                peft_method=peft_method,
                alpha=alpha,
                proj_size=proj_size,
                dropout=dropout
            )


def main(args):

    switch = {
        'lora_a': run_lora_eval_a,
        'lora_b': run_lora_eval_b,
        'lora_c': run_lora_eval_c,
        'lora_d': run_lora_eval_d,
        'adaptformer': run_adaptformer_eval,
    }

    # Iterate over the list of experiments and execute them
    for experiment in args.experiments:
        experiment_function = switch.get(experiment)
        if experiment_function:
            experiment_function()
        else:
            print(f"Experiment {experiment} not recognized.")


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiments',
        nargs='+',  # Allow multiple values to be passed as a list
        choices=['lora_a', 'lora_b', 'lora_c', 'lora_d', 'adaptformer'],
        required=True,
        help="Specify which experiments to run (space-separated list)"
    )
    args = parser.parse_args()
    main(args)
