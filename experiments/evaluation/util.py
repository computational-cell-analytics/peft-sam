import os
import argparse
from glob import glob

from torch_em.data import datasets

from micro_sam.evaluation.livecell import _get_livecell_paths


ROOT = "/scratch/usr/nimcarot/data/"  # for Caro

#if os.path.exists("/media/anwai/ANWAI"):  # for Anwai
#    ROOT = "/media/anwai/ANWAI/data"
#else:
#    ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/peft_sam/data"

# EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models"

FILE_SPECS = {
    "platynereis/cilia": {"val": "platy_cilia_val_*", "test": "platy_cilia_test_*"},
}

# Good spot to track all datasets we use atm
DATASETS = [
    # in-domain (LM)
    "livecell",
    # out-of-domain (LM)
    "covid_if", "orgasegment", "gonuclear", "hpa",
    # organelles (EM)
    # - out-of-domain
    "mitolab/glycolytic_muscle", "platynereis/cilia",

    # out-of-domain (MI)
    "papila", "motum", "psfhs", "jsrt", "amd_sd", "mice_tumseg",
    # and some new datasets
    "sega", "ircadb", "dsad",
]


def get_dataset_paths(dataset_name, split_choice):
    # Let's check if we have a particular naming logic to save the images
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


def download_all_datasets(path, for_microscopy=True):
    if for_microscopy:
        # microscopy datasets
        datasets.get_livecell_dataset(
            path=os.path.join(path, "livecell"), split="test", patch_shape=(512, 512), download=True
        )
        datasets.get_platynereis_cilia_dataset(
            os.path.join(path, "platynereis"), patch_shape=(1, 512, 512), download=True
        )
        print(
            "MitoLab benchmark datasets need to downloaded separately. "
            "See `torch_em.data.datasets.light_microscopy.cem.get_benchmark_datasets` for details."
        )
        datasets.get_covid_if_dataset(
            path=os.path.join(path, "covid_if"), patch_shape=(1, 512, 512), download=True
        )
        datasets.get_orgasegment_dataset(
            path=os.path.join(path, "orgasegment"), patch_shape=(1, 512, 512), download=True
        )
        datasets.get_gonuclear_dataset(
            path=os.path.join(path, "gonuclear"), patch_shape=(1, 512, 512), download=True
        )
        datasets.get_hpa_segmentation_dataset(
            path=os.path.join(path, "hpa"), patch_shape=(1, 512, 512), split="test", download=True
        )

    else:
        # medical imaging datasets
        datasets.get_papila_dataset(
            path=os.path.join(path, "papila"), patch_shape=(1, 512, 512), split="test", task="cup", download=True
        )
        datasets.get_motum_dataset(
            path=os.path.join(path, "motum"), patch_shape=(1, 512, 512), split="test", modality="flair", download=True,
        )
        datasets.get_psfhs_dataset(
            path=os.path.join(path, "psfhs"), patch_shape=(1, 512, 512), split="test", download=True,
        )
        datasets.get_jsrt_dataset(
            path=os.path.join(path, "jsrt"), patch_shape=(1, 512, 512), split="test", download=True,
        )
        datasets.get_amd_sd_dataset(
            path=os.path.join(path, "amd_sd"), patch_shape=(1, 512, 512), split="test", download=True,
        )
        datasets.get_mice_tumseg_dataset(
            path=os.path.join(path, "mice_tumseg"), patch_shape=(1, 512, 512), split="test", download=True,
        )
        # NEW DATASETS
        datasets.get_sega_dataset(
            path=os.path.join(path, "sega"), patch_shape=(1, 512, 512), data_choice="Rider", download=True,
        )
        datasets.get_dsad_dataset(
            path=os.path.join(path, "dsad"), patch_shape=(512, 512), download=True,
        )
        datasets.get_ircadb_dataset(
            path=os.path.join(path, "ircadb"), patch_shape=(1, 512, 512), label_choice="liver", download=True,
        )

#
# PARSER FOR ALL THE REQUIRED ARGUMENTS
#


def get_default_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Provide the model type to initialize the predictor"
    )
    parser.add_argument(
        "-c", "--checkpoint", type=None, default=None, help="The filepath to the model checkpoint."
    )
    parser.add_argument(
        "-e", "--experiment_folder", type=str, default="./experiments", help="The path to cache experiment files."
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="The name of dataset."
    )
    parser.add_argument(
        "--box", action="store_true", help="If passed, starts interactive segmentation with first prompt as box"
    )
    parser.add_argument(
        "--use_masks", action="store_true", help="To use logits masks for iterative prompting."
    )
    parser.add_argument(
        "--peft_rank", type=int, default=None, help="The rank for peft method."
    )
    parser.add_argument(
        "--peft_module", default=None, type=str, help="The module for peft method. (e.g. LoRA or FacT)"
    )
    parser.add_argument(
        "--dropout", default=None, type=float, help="The dropout factor for FacT finetuning"
    )
    parser.add_argument(
        "--alpha", default=None, help="Scaling factor for peft method. (e.g. LoRA or AdaptFormer)"
    )
    parser.add_argument(
        "--projection_size", default=None, type=int, help="Projection size for Adaptformer module."
    )
    parser.add_argument(
        "--attention_layers_to_update", default=None, type=int, nargs="+", help="The attention layers to update."
    )
    parser.add_argument(
        "--update_matrices", default=None, type=str, nargs="+", help="The matrices to update."
    )
    parser.add_argument("--quantize", action="store_true", help="Whether to quantize the model.")

    args = parser.parse_args()
    return args
