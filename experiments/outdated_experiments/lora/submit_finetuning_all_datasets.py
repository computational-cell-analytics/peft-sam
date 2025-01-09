import os
import shutil
import subprocess
from datetime import datetime
import itertools


ALL_DATASETS = {'covid_if': 'lm', 'orgasegment': 'lm', 'gonuclear': 'lm', 'mitolab_glycolytic_muscle': 'em_organelles',
                'platy_cilia': 'em_organelles'}


def write_batch_script(
    env_name, save_root, model_type, script_name, checkpoint_path, checkpoint_name=None, dataset="livecell",
    peft_rank=None, peft_method=None, alpha=None, learning_rate=1e-5
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


def run_rank_study():
    """
    Submit the finetuning jobs for a rank study on mito-lab and orgasegment datasets
    - from generalist and from default SAM
    - for ranks 1, 2, 4, 8, 16, 32, 64
    """

    ranks = [None, 1, 2, 4, 8, 16, 32, 64]
    for rank in ranks:
        for dataset in ["mitolab_glycolytic_muscle", "orgasegment"]:
            region = ALL_DATASETS[dataset]
            generalist_model = f"vit_b_{region}"
            for base_model in ["vit_b", generalist_model]:
                script_name = get_batch_script_names("./gpu_jobs")
                peft_method = "lora" if rank is not None else None
                checkpoint_name = f"{base_model}/lora/rank_{rank}/{dataset}_sam" if rank is not None else None
                write_batch_script(
                    env_name="sam",
                    save_root=args.save_root,
                    model_type=base_model,
                    script_name=script_name,
                    checkpoint_path=None,
                    checkpoint_name=checkpoint_name,
                    peft_rank=rank,
                    peft_method=peft_method,
                    dataset=dataset,
                )


def run_all_dataset_ft():
    """
    Submit the finetuning jobs for all datasets
    - from generalist full finetuning
    - from generalist lora
    """

    for dataset, region in ALL_DATASETS.items():
        for rank in [None, 4]:
            generalist_model = f"vit_b_{region}"
            script_name = get_batch_script_names("./gpu_jobs")
            peft_method = "lora" if rank is not None else None
            write_batch_script(
                env_name="sam",
                save_root=args.save_root,
                model_type=generalist_model,
                script_name=script_name,
                checkpoint_path=None,
                peft_rank=rank,
                peft_method=peft_method,
                dataset=dataset,
            )


def run_scaling_factor_exp_a():
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


def run_scaling_factor_exp_b(args):
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


def run_scaling_factor_exp_c(args):
    """
    Submit the finetuning jobs LoRA on OrgaSegment
    - alpha in [0.1, 0.25, 0.5, 0.75]
    - learning rate = 1e-5
    """
    alphas = [0.1, 0.25, 0.5, 0.75]
    rank = 32
    lr = 1e-5

    for alpha in alphas:
        for model in ["vit_b", "vit_b_lm"]:
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


def run_scaling_factor_exp_d(args):
    """
    Submit the finetuning jobs LoRA on OrgaSegment, CovidIF, MitoLab, Platynereis Cilia
    """
    alphas = [0.1, 1, 8]
    rank = 32
    lr = 1e-5
    datasets = ["orgasegment", "covid_if", "mitolab_glycolytic_muscle", "platy_cilia"]

    for alpha, dataset in itertools.product(alphas, datasets):
        if dataset in ["orgasegment", "covid_if"]:
            generalist_model = "vit_b_lm"
        else:
            generalist_model = "vit_b_em_organelles"  
        for model in ["vit_b", generalist_model]:
            script_name = get_batch_script_names("./gpu_jobs")
            peft_method = "lora"
            checkpoint_name = f"{model}/lora/lr_{lr}/rank_{rank}/alpha_{alpha}/{dataset}_sam"
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
                dataset=dataset,
                learning_rate=lr
            )


def main(args):

    switch = {
        'ft_all_data': run_all_dataset_ft,
        'rank_study': run_rank_study,
        'scaling_factor_a': run_scaling_factor_exp_a,
        'scaling_factor_b': run_scaling_factor_exp_b,
        'scaling_factor_c': run_scaling_factor_exp_c,
        'scaling_factor_d': run_scaling_factor_exp_d
    }

    # Get the corresponding experiment function based on the argument and execute it
    experiment_function = switch.get(args.experiment)

    # Run the selected experiment
    experiment_function(args)


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_root", type=str, default=None, help="Path to save checkpoints.")
    parser.add_argument(
        '--experiment',
        choices=['ft_all_data', 'rank_study', 'scaling_factor_a', 'scaling_factor_b', 'scaling_factor_c', 'scaling_factor_d'],
        required=True,
        help="Specify which experiment to run"
    )

    args = parser.parse_args()
    main(args)
