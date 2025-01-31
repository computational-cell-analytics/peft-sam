import os
from glob import glob
from tqdm import tqdm

import pandas as pd

import torch


ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/peft_sam/models/checkpoints"


def get_peft_train_times_for_mi(model_type):
    # Get path to all "best" checkpoints.
    checkpoint_paths = glob(os.path.join(ROOT, model_type, "**", "best.pt"), recursive=True)

    time_per_checkpoint = []
    for checkpoint_path in tqdm(checkpoint_paths):
        psplit = checkpoint_path.rsplit("/")

        # Get stuff important to store information for.
        data_name = psplit[-2][:-4]
        peft_method = psplit[-3]

        # Get train times.
        train_time = torch.load(checkpoint_path, map_location="cpu")["train_time"]

        time_per_checkpoint.append(
            pd.DataFrame.from_dict([{"dataset": data_name, "peft": peft_method, "best_train_time": train_time}])
        )

    # Store all times locally.
    times = pd.concat(time_per_checkpoint, ignore_index=True)
    print(times)
    times.to_csv(f"./medical_imaging_peft_best_times_{model_type}.csv")


def main():
    get_peft_train_times_for_mi("vit_b")
    get_peft_train_times_for_mi("vit_b_medical_imaging")


if __name__ == "__main__":
    main()
