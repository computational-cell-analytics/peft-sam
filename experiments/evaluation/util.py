import os
import argparse
from glob import glob

from torch_em.data import datasets

from micro_sam.evaluation.livecell import _get_livecell_paths


ROOT = "/scratch/usr/nimcarot/data/"

EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models"

FILE_SPECS = {
    "platynereis/cilia": {"val": "platy_cilia_val_*", "test": "platy_cilia_test_*"},
}

# good spot to track all datasets we use atm
DATASETS = [
    # in-domain (LM)
    "livecell",
    # out-of-domain (LM)
    "covid_if", "orgasegment", "gonuclear",
    # organelles (EM)
    #   - out-of-domain
    "mitolab/glycolytic_muscle", "platynereis/cilia"
]


def get_dataset_paths(dataset_name, split_choice):
    # let's check if we have a particular naming logic to save the images
    try:
        file_search_specs = FILE_SPECS[dataset_name][split_choice]
        is_explicit_split = False
    except KeyError:
        file_search_specs = "*"
        is_explicit_split = True

    # if the datasets have different modalities/species, let's make use of it
    split_names = dataset_name.split("/")
    if len(split_names) > 1:
        assert len(split_names) <= 2
        dataset_name = [split_names[0], "slices", split_names[1]]
    else:
        dataset_name = [*split_names, "slices"]

    # if there is an explicit val/test split made, let's look at them
    if is_explicit_split:
        dataset_name.append(split_choice)

    raw_dir = os.path.join(ROOT, *dataset_name, "raw", file_search_specs)
    labels_dir = os.path.join(ROOT, *dataset_name, "labels", file_search_specs)

    return raw_dir, labels_dir


def get_paths(dataset_name, split):
    assert dataset_name in DATASETS, dataset_name

    if dataset_name == "livecell":
        image_paths, gt_paths = _get_livecell_paths(input_folder=os.path.join(ROOT, "livecell"), split=split)
        return sorted(image_paths), sorted(gt_paths)

    image_dir, gt_dir = get_dataset_paths(dataset_name, split)
    image_paths = sorted(glob(os.path.join(image_dir)))
    gt_paths = sorted(glob(os.path.join(gt_dir)))
    return image_paths, gt_paths


def get_pred_paths(prediction_folder):
    pred_paths = sorted(glob(os.path.join(prediction_folder, "*")))
    return pred_paths


def download_all_datasets(path):

    # platy-cilia
    datasets.get_platynereis_cilia_dataset(os.path.join(path, "platynereis"), patch_shape=(1, 512, 512), download=True)

    # mitolab
    print("MitoLab benchmark datasets need to downloaded separately. See `datasets.cem.get_benchmark_datasets`")

    # covid-if
    datasets.get_covid_if_dataset(os.path.join(path, "covid_if"), patch_shape=(1, 512, 512), download=True)

    # orgasegment
    datasets.get_orgasegment_dataset(os.path.join(path, "orgasegment"), patch_shape=(1, 512, 512), download=True)

    # gonuclear
    datasets.get_gonuclear_dataset(os.path.join(path, "gonuclear"), patch_shape=(1, 512, 512), download=True)

#
# PARSER FOR ALL THE REQUIRED ARGUMENTS
#


def get_default_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Provide the model type to initialize the predictor"
    )
    parser.add_argument("-c", "--checkpoint", type=none_or_str, default=None)
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument("--box", action="store_true", help="If passed, starts with first prompt as box")
    parser.add_argument(
        "--use_masks", action="store_true", help="To use logits masks for iterative prompting."
    )
    parser.add_argument("--peft_rank", default=None, type=int, help="The rank for peft method.")
    parser.add_argument("--peft_module", default=None, type=str, help="The module for peft method. (e.g. LoRA or FacT)")
    parser.add_argument("--dropout", default=None, type=float, help="The dropout factor for FacT finetuning")
    parser.add_argument(
        "--alpha", default=None, type=float, help="Scaling factor for peft method. (e.g. LoRA or AdaptFormer)"
    )

    args = parser.parse_args()
    return args


def none_or_str(value):
    if value == 'None':
        return None
    return value
