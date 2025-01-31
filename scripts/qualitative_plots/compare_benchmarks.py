import os
from glob import glob
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from matplotlib import colors
import matplotlib.pyplot as plt
import cv2

from elf.evaluation import mean_segmentation_accuracy

from micro_sam.evaluation.livecell import _get_livecell_paths
from micro_sam.evaluation.model_comparison import _enhance_image, _overlay_outline, _overlay_mask


ROOT = "/scratch/usr/nimcarot/sam/experiments"
DATA_ROOT = "/scratch/usr/nimcarot/data"


# a function to generate a random color map for a label image
def get_random_colors(labels):
    n_labels = len(np.unique(labels)) - 1
    cmap = [[0, 0, 0]] + np.random.rand(n_labels, 3).tolist()
    cmap = colors.ListedColormap(cmap)
    return cmap


def compare_cellseg1_vs_ais(all_images, all_gt, dataset_name):
    if dataset_name == "mitolab_glycolytic_muscle":
        model = "vit_b_em_organelles"
    else:
        model = "vit_b"
    ais_base_root = os.path.join(
        ROOT, f"single_img_training/generalist/{dataset_name}/instance_segmentation_with_decoder/inference"
    )
    ais_lora_root = os.path.join(
        ROOT,
        f"single_img_training/lora/{model}/{dataset_name}/instance_segmentation_with_decoder/inference"
    )
    assert os.path.exists(ais_base_root), ais_base_root
    assert os.path.exists(ais_lora_root), ais_lora_root

    cellseg1 = os.path.join(ROOT, f"peft/cellseg1/{model}/{dataset_name}/pred_masks/")

    all_res = []
    for gt_path in tqdm(all_gt):
        image_id = os.path.split(gt_path)[-1]

        gt = imageio.imread(gt_path)
        ais_seg = imageio.imread(os.path.join(ais_lora_root, image_id))

        score = {
            "name": image_id.split(".")[0],
            "score": mean_segmentation_accuracy(ais_seg, gt)
        }
        all_res.append(pd.DataFrame.from_dict([score]))

    all_res = pd.concat(all_res, ignore_index=True)

    sscores = np.array(all_res["score"]).argsort()[::-1]
    best_image_ids = [all_res.iloc[sscore]["name"] for sscore in sscores]

    for image_path, gt_path in zip(all_images, all_gt):
        image_id = os.path.split(image_path)[-1]
        if Path(image_id).stem not in best_image_ids:
            continue

        gt = imageio.imread(gt_path)

        image = imageio.imread(image_path)
        if dataset_name != "hpa":
            image = _enhance_image(image, do_norm=True if dataset_name == "covid_if" else False)

        image = _overlay_mask(image, gt, alpha=0.95)
        image = _overlay_outline(image, gt, 0)

        ais_base = imageio.imread(os.path.join(ais_base_root, image_id))
        ais_lora = imageio.imread(os.path.join(ais_lora_root, image_id))

        cellseg1_seg = imageio.imread(
            os.path.join(
                cellseg1, image_id.replace(".tif", ".png")
            )
        )

        fig, ax = plt.subplots(1, 4, figsize=(30, 20), sharex=True, sharey=True)
        if dataset_name == "mitolab_glycolytic_muscle":
            cellseg1_seg = cv2.resize(cellseg1_seg, (765, 383), interpolation=cv2.INTER_LINEAR_EXACT)
            # show a crop of the image
            image = image[50:250, 50:250]
            ais_base = ais_base[50:250, 50:250]
            ais_lora = ais_lora[50:250, 50:250]
            cellseg1_seg = cellseg1_seg[50:250, 50:250]

        ax[0].imshow(image, cmap="gray")
        ax[0].axis("off")

        ax[1].imshow(ais_base, cmap=get_random_colors(ais_base), interpolation="nearest")
        ax[1].axis("off")

        ax[2].imshow(ais_lora, cmap=get_random_colors(ais_lora), interpolation="nearest")
        ax[2].axis("off")

        ax[3].imshow(cellseg1_seg, cmap=get_random_colors(cellseg1_seg), interpolation="nearest")
        ax[3].axis("off")

        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f"./figs/{dataset_name}-cellseg1/{Path(image_id).stem}.svg", bbox_inches="tight")
        plt.close()


def get_paths(dataset_name):
    if dataset_name == "livecell":
        image_paths, gt_paths = _get_livecell_paths(
            input_folder=os.path.join(DATA_ROOT, "livecell"), split="test"
        )

    elif dataset_name == "covid_if":
        root_dir = os.path.join(DATA_ROOT, dataset_name, "slices", "test")
        image_paths = [_path for _path in glob(os.path.join(root_dir, "raw", "*.tif"))]
        gt_paths = [_path for _path in glob(os.path.join(root_dir, "labels", "*.tif"))]

    elif dataset_name == "hpa":
        root_dir = os.path.join(DATA_ROOT, dataset_name, "slices", "test")
        image_paths = [_path for _path in glob(os.path.join(root_dir, "raw", "*.tif"))]
        gt_paths = [_path for _path in glob(os.path.join(root_dir, "labels", "*.tif"))]

    elif dataset_name.startswith("mitolab"):
        root_dir = os.path.join(DATA_ROOT, "mitolab", "slices", "glycolytic_muscle", "test")
        image_paths = [_path for _path in glob(os.path.join(root_dir, "raw", "*"))]
        gt_paths = [_path for _path in glob(os.path.join(root_dir, "labels", "*"))]

    else:
        raise ValueError
    assert len(image_paths) == len(gt_paths)

    return sorted(image_paths), sorted(gt_paths)


def compare_experiments(dataset_name):
    all_images, all_gt = get_paths(dataset_name=dataset_name)

    compare_cellseg1_vs_ais(all_images, all_gt, dataset_name=dataset_name)


def main():

    # compare_experiments("hpa")
    compare_experiments("mitolab_glycolytic_muscle")
    # compare_experiments("covid_if")


if __name__ == "__main__":
    main()
