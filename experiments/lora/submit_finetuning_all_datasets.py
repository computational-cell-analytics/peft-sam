import os
import shutil
import subprocess
from datetime import datetime


ALL_DATASETS = {'covid_if': 'lm', 'orgasegment': 'lm', 'gonuclear': 'lm', 'mitolab_glycolytic_muscle': 'em_organelles',
                'platy_cilia': 'em_organelles'}


def write_batch_script(
    env_name, save_root, model_type, script_name, checkpoint_path, checkpoint_name=None, dataset="livecell",
    peft_rank=None, peft_method=None, alpha=None
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

    if save_root is not None:
        python_script += f"-s {save_root} "  # path to save model checkpoints and logs

    if peft_rank is not None:
        python_script += f"--peft_rank {peft_rank} "
    if peft_method is not None:
        python_script += f"--peft_method {peft_method} "
    if alpha is not None:
        python_script += f"--alpha {alpha} "
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


def run_scaling_factor_exp(args):
    """
    Submit the finetuning jobs LoRA on LIVECell
    - alpha in [1, 2, 4, 8, 16, 32, 64]
    - rank in [1, 2, 4, 8, 16, 32, 64]
    """
    alphas = [1, 2, 4, 8, 16, 32, 64]
    ranks = [1, 2, 4, 8, 16, 32, 64]

    for alpha in alphas:
        for rank in ranks:
            # only consider those combinations where the ratio is within [0.25,8]
            ratio = alpha / rank
            if not (0.25 <= ratio <= 8):
                continue
            model = "vit_b"
            script_name = get_batch_script_names("./gpu_jobs")
            peft_method = "lora"
            checkpoint_name = f"{model}/lora/rank_{rank}/alpha_{alpha}/livecell_sam"
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
                dataset="livecell",
            )


def main(args):

    switch = {
        'ft_all_data': run_all_dataset_ft,
        'rank_study': run_rank_study,
        'scaling_factor': run_scaling_factor_exp
    }

    # Get the corresponding experiment function based on the argument and execute it
    experiment_function = switch.get(args.experiment)

    # Run the selected experiment
    experiment_function()


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
        choices=['ft_all_data', 'rank_study', 'scaling_factor'],
        required=True,
        help="Specify which experiment to run"
    )

    args = parser.parse_args()
    main(args)
