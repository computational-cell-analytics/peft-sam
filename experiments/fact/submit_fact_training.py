import os
import shutil
import subprocess
from datetime import datetime

from peft_sam.util import EXPERIMENT_ROOT


def write_batch_script(
    env_name, save_root, model_type, script_name, checkpoint_path, peft_rank, peft_method, dropout,
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
    python_script += "-d livecell "  # dataset

    if checkpoint_path is not None:
        python_script += f"-c {checkpoint_path} "

    if save_root is not None:
        python_script += f"-s {save_root} "  # path to save model checkpoints and logs

    if peft_rank is not None:
        python_script += f"--peft_rank {peft_rank} "
    if peft_method is not None:
        python_script += f"--peft_method {peft_method} "
    if dropout is not None:
        python_script += f"--fact_dropout {dropout} "
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


def run_dropout_study():
    """
    Submit the finetuning jobs for a study on the dropout rate with rates
    [None, 0.1, 0.25, 0.5]
    """

    dropout_rates = [None, 0.1, 0.25, 0.5]
    for dropout in dropout_rates:
        script_name = get_batch_script_names("./gpu_jobs")
        write_batch_script(
            env_name="sam",
            save_root=args.save_root,
            model_type="vit_b",
            script_name=script_name,
            checkpoint_path=None,
            peft_rank=4,
            peft_method="fact",
            dropout=dropout,
            )


def run_rank_study():
    """
    Submit the finetuning scripts for a rank study on livecell dataset
    """
    ranks = [1, 2, 4, 8, 16, 32]
    for rank in ranks:
        script_name = get_batch_script_names("./gpu_jobs")
        write_batch_script(
            env_name="sam",
            save_root=args.save_root,
            model_type="vit_b",
            script_name=script_name,
            checkpoint_path=None,
            peft_rank=rank,
            peft_method="fact",
            dropout=0.1,
        )


def main(args):
    switch = {
        'dropout_study': run_dropout_study,
        'rank_study': run_rank_study
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
    parser.add_argument("-s", "--save_root", type=str, default=os.path.join(EXPERIMENT_ROOT, "fact_dropout"),
                        help="Path to save checkpoints.")
    parser.add_argument(
        '--experiment',
        choices=['dropout_study', 'rank_study'],
        required=True,
        help="Specify which experiment to run"
    )

    args = parser.parse_args()
    main(args)
