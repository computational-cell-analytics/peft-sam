import os
import shutil
from glob import glob
from tqdm import tqdm

import pandas as pd

import torch


ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/peft_sam/models"


def get_peft_train_times_for_mi(model_type):
    # Get path to all "best" checkpoints.
    checkpoint_paths = glob(os.path.join(ROOT, "checkpoints", model_type, "**", "best.pt"), recursive=True)

    time_per_checkpoint = []
    for checkpoint_path in tqdm(checkpoint_paths):
        psplit = checkpoint_path.rsplit("/")

        # Get stuff important to store information for.
        data_name = psplit[-2][:-4]
        peft_method = psplit[-3]

        # Get train times.
        train_time = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["train_time"]

        time_per_checkpoint.append(
            pd.DataFrame.from_dict([{"dataset": data_name, "peft": peft_method, "best_train_time": train_time}])
        )

    # Store all times locally.
    times = pd.concat(time_per_checkpoint, ignore_index=True)
    print(times)
    times.to_csv(f"./medical_imaging_peft_best_times_{model_type}.csv")


def assort_quantitative_results():
    # template for storing this below:
    # psfhs_LayerNormSurgery-vit_b_medical_imaging_point.csv

    # I forgot how I did it before, but I have all results in one place. I will do the filename mapping in
    # a simple fashion so that it's easy to plot stuff and reproduce them later, if we need it.
    fres_dir = "./results"
    os.makedirs(fres_dir, exist_ok=True)

    for res_path in glob(os.path.join(ROOT, "**", "results", "iterative_prompting*", "*.csv*"), recursive=True):

        if "logs" in res_path or "checkpoints" in res_path:
            # We only want result dirs, the others dirs are not relevant for us.
            continue

        # Get the expected extension from the filename above
        fext = res_path.split("_")[-1]

        # Let's get relevant stuff to do the mapping
        _split = res_path.rsplit("/")
        if "vanilla" in res_path or "generalist" in res_path:  # We need to treat this a bit differently
            target_fpath = f"{_split[-4]}_{_split[-5]}_{fext}"
        else:
            target_fpath = f"{_split[-4]}_{_split[-6]}-{_split[-5]}_{fext}"

        target_fpath = os.path.join(fres_dir, target_fpath)

        # Now, copy it in another location.
        shutil.copy(src=res_path, dst=target_fpath)


def validate_results(dataset_name, model_name, prompt):
    res_dir = "./results"

    generalist_res = pd.read_csv(os.path.join(res_dir, f"{dataset_name}_generalist_{prompt}.csv"))
    vanilla_res = pd.read_csv(os.path.join(res_dir, f"{dataset_name}_vanilla_{prompt}.csv"))

    print(vanilla_res)
    print(generalist_res)

    for res_path in glob(os.path.join(res_dir, f"{dataset_name}*{model_name}_{prompt}.csv")):
        print(res_path)
        print(pd.read_csv(res_path))


def main():
    get_peft_train_times_for_mi("vit_b")
    get_peft_train_times_for_mi("vit_b_medical_imaging")

    # assort_quantitative_results()
    # validate_results("dsad", "vit_b", "point")


if __name__ == "__main__":
    main()
