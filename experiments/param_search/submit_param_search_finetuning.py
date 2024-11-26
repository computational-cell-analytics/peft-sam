import os
import shutil
import subprocess
from datetime import datetime
import itertools


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
    projection_size=None
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


def run_lora_a():
    """
    Submit the finetuning jobs LoRA on Orgasegment
    - alpha in [1, 2, 4]
    - rank in [1, 2, 4, 8, 16, 32, 64]
    """
    alphas = [1, 2, 4]
    ranks = [1, 2, 4, 8, 16, 32, 64]

    for alpha, rank in itertools.product(alphas, ranks):
        model = "vit_b_lm"
        script_name = get_batch_script_names("./gpu_jobs")
        peft_method = "lora"
        checkpoint_name = f"{model}/lora/lr_1e-5/rank_{rank}/alpha_{alpha}/orgasegment_sam"
        write_batch_script(
            env_name="sam",
            save_root=args.save_root,
            model_type=model,
            script_name=script_name,
            checkpoint_path=None,
            peft_rank=rank,
            peft_method=peft_method,
            alpha=alpha,
            checkpoint_name=checkpoint_name,
            dataset="orgasegment",
        )


def run_lora_b(args):
    """
    Submit the finetuning jobs LoRA on OrgaSegment
    - alpha in [1, 2, 4]
    - rank in [1, 32]
    - learning rate in [1e-3, 5e-4, 1e-4, 5e-5]
    """
    alphas = [1, 2, 4]
    ranks = [1, 32]
    lrs = [1e-3, 5e-4, 1e-4, 5e-5]

    for alpha, rank, lr in itertools.product(alphas, ranks, lrs):
        model = "vit_b_lm"
        script_name = get_batch_script_names("./gpu_jobs")
        peft_method = "lora"
        checkpoint_name = f"{model}/lora/lr_{lr}/rank_{rank}/alpha_{alpha}/orgasegment_sam"
        if os.path.exists(os.path.join(args.save_root, "checkpoints", checkpoint_name)):
            continue
        write_batch_script(
            env_name="sam",
            save_root=args.save_root,
            model_type=model,
            script_name=script_name,
            checkpoint_path=None,
            peft_rank=rank,
            peft_method=peft_method,
            alpha=alpha,
            checkpoint_name=checkpoint_name,
            dataset="orgasegment",
            learning_rate=lr
        )


def run_lora_c(args):
    """
    Submit the finetuning jobs LoRA on OrgaSegment with fixed learning rate and rank
    - alpha in [0.1, 0.25, 0.5, 0.75]
    """
    alphas = [0.1, 0.25, 0.5, 0.75]
    rank = 32
    lr = 1e-5

    for alpha in alphas:
        model = "vit_b_lm"
        script_name = get_batch_script_names("./gpu_jobs")
        peft_method = "lora"
        checkpoint_name = f"{model}/lora/lr_{lr}/rank_{rank}/alpha_{alpha}/orgasegment_sam"
        if os.path.exists(os.path.join(args.save_root, "checkpoints", checkpoint_name)):
            continue
        write_batch_script(
            env_name="sam",
            save_root=args.save_root,
            model_type=model,
            script_name=script_name,
            checkpoint_path=None,
            peft_rank=rank,
            peft_method=peft_method,
            alpha=alpha,
            checkpoint_name=checkpoint_name,
            dataset="orgasegment",
            learning_rate=lr
        )


def run_lora_d(args):
    """
    Submit finetuning for differnt scaling factors on 4 datasets
    """
    alphas = [0.1, 1, 8]
    lr = 1e-5
    rank = 32
    datasets = ["orgasegment", "covid_if", "mitolab_glycolytic_muscle", "platy_cilia"]

    for alpha, dataset in itertools.product(alphas, datasets):
        if dataset in ["orgasegment", "covid_if"]:
            generalist = "vit_b_lm"
        else:
            generalist = "vit_b_em_organelles"

        for model in ["vit_b", generalist]:
            script_name = get_batch_script_names("./gpu_jobs")
            peft_method = "lora"
            checkpoint_name = f"{model}/lora/lr_{lr}/rank_{rank}/alpha_{alpha}/{dataset}_sam"
            if os.path.exists(os.path.join(args.save_root, "checkpoints", checkpoint_name, 'best.pt')):
                print("Aborting, because checkpoint already exists.")
                continue
            write_batch_script(
                env_name="sam",
                save_root=args.save_root,
                model_type=model,
                script_name=script_name,
                checkpoint_path=None,
                peft_rank=rank,
                peft_method=peft_method,
                alpha=alpha,
                checkpoint_name=checkpoint_name,
                dataset=dataset,
                learning_rate=lr
            )


def run_adaptformer(args):
    """
    Submit grid search for AdaptFormer parameters on OrgaSegment
    alpha in [0.1, 0.25, 0.5, 0.75, 1, 'learnable_scalar']
    projection_size in [64, 128, 256]
    dropout in [0.1, 0.25, 0.5, None]
    """

    alphas = [0.1, 0.5, 1, 'learnable_scalar']
    projection_sizes = [64, 128, 256]
    dropouts = [0.1, 0.25, 0.5, None]

    for alpha, proj_size, dropout in itertools.product(alphas, projection_sizes, dropouts):
        model = "vit_b_lm"
        script_name = get_batch_script_names("./gpu_jobs")
        peft_method = "adaptformer"
        checkpoint_name = f"{peft_method}/alpha_{alpha}/projection_size_{proj_size}/dropout_{dropout}/orgasegment_sam"

        if os.path.exists(os.path.join(args.save_root, "checkpoints", f"{model}", checkpoint_name, 'best.pt')):
            print("Checkpoint already exists, aborting")
            continue

        write_batch_script(
            env_name="sam",
            save_root=args.save_root,
            model_type=model,
            script_name=script_name,
            checkpoint_path=None,
            peft_rank=1,
            peft_method=peft_method,
            alpha=alpha,
            checkpoint_name=checkpoint_name,
            dataset="orgasegment",
            learning_rate=1e-5,
            projection_size=proj_size,
            dropout=dropout
        )


def main(args):
    # Define the mapping of experiments to their corresponding functions
    switch = {
        'lora_a': run_lora_a,
        'lora_b': run_lora_b,
        'lora_c': run_lora_c,
        'lora_d': run_lora_d,
        'adaptformer': run_adaptformer,
    }

    # Iterate over the list of experiments and execute them
    for experiment in args.experiments:
        experiment_function = switch.get(experiment)
        if experiment_function:
            experiment_function(args)
        else:
            print(f"Experiment {experiment} not recognized.")


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
        default="/scratch/usr/nimcarot/sam/experiments/peft_param_search",
        help="Path to save checkpoints."
    )
    parser.add_argument(
        '--experiments',
        nargs='+',  # Allow multiple values to be passed as a list
        choices=['lora_a', 'lora_b', 'lora_c', 'lora_d', 'adaptformer'],
        required=True,
        help="Specify which experiments to run (space-separated list)"
    )

    args = parser.parse_args()
    main(args)
