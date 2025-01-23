import os
from glob import glob
from natsort import natsorted

import numpy as np
import pandas as pd

from tukra.io import read_image

from medico_sam.evaluation.evaluation import calculate_dice_score

from util import get_paths


EXPERIMENT_ROOT = "/media/anwai/ANWAI/peft_sam"


def run_evaluation_over_images_per_class(gt_paths, pred_folder):
    pred_paths = natsorted(glob(os.path.join(pred_folder, "*")))

    assert len(gt_paths) == len(pred_paths) and gt_paths

    per_image_dice = []
    for gt_path, pred_path in zip(gt_paths, pred_paths):
        gt = read_image(gt_path)
        pred = read_image(pred_path)

        gt_ids = np.unique(gt)[1:]  # remove background id

        # Get mean over per class instance segmentations.
        per_image_dice.append(
            np.mean([calculate_dice_score(gt == id, pred == id) for id in gt_ids])
        )

    return np.mean(per_image_dice)


def main():
    os.makedirs("./results", exist_ok=True)
    for dataset in ["mice_tumseg"]:
        _, gt_paths = get_paths(dataset, split="test")
        experiment_folders = glob(os.path.join(EXPERIMENT_ROOT, "**", dataset), recursive=True)

        for exp_folder in experiment_folders:
            experiment = exp_folder[len(EXPERIMENT_ROOT)+1:]
            comps = experiment.rsplit("/")

            # Define per experiment name to store average dice score.
            if len(comps) == 2:  # these are base FMs.
                name = f"{comps[0]}"
            else:  # these are our FT models.
                name = f"{comps[0]}-{comps[1]}"

            # For box prompts
            box_res = []
            for pred_folder in natsorted(glob(os.path.join(exp_folder, "start_with_box", "iteration*"))):
                box_res.append(
                    pd.DataFrame.from_dict(
                        [{
                            "iteration": os.path.basename(pred_folder),
                            "dice_score": run_evaluation_over_images_per_class(gt_paths, pred_folder)
                        }]
                    )
                )
            box_res = pd.concat(box_res, ignore_index=True)
            box_res.to_csv(f"./results/{dataset}_{name}_box.csv")

            # For point prompts
            point_res = []
            for pred_folder in natsorted(glob(os.path.join(exp_folder, "start_with_box", "iteration*"))):
                point_res.append(
                    pd.DataFrame.from_dict(
                        [{
                            "iteration": os.path.basename(pred_folder),
                            "dice_score": run_evaluation_over_images_per_class(gt_paths, pred_folder)
                        }]
                    )
                )
            point_res = pd.concat(point_res, ignore_index=True)
            point_res.to_csv(f"./results/{dataset}_{name}_point.csv")

            print(name)
            print(box_res)
            print(point_res)

        breakpoint()


if __name__ == "__main__":
    main()
