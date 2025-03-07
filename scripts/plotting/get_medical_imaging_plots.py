import os

import numpy as np
from skimage.measure import label as connected_components

from torch_em.data.datasets import medical
from torch_em.data import MinInstanceSampler

import micro_sam.training as sam_training
from micro_sam.evaluation.model_comparison import generate_data_for_model_comparison, model_comparison


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/peft_sam/data"
MODEL_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/peft_sam/models/checkpoints/vit_b_medical_imaging/"


# Avoid any transformations.
def _transform_identity(raw, labels):
    return raw, labels


# Maps labels to expected instance structure (to train for interactive segmentation).
def _jsrt_label_trafo(labels):
    neu_label = np.zeros_like(labels)
    lungs = (labels == 255)  # Labels for lungs
    lungs = connected_components(lungs)  # Ensure both lung volumes unique
    neu_label[lungs > 0] = lungs[lungs > 0]  # Map both lungs to new label.
    neu_label[labels == 85] = np.max(neu_label) + 1   # Belongs to heart labels.
    return neu_label


# Ensures all labels are unique.
def _amd_sd_label_trafo(labels):
    labels = connected_components(labels).astype(labels.dtype)
    return labels


# Adjusting the data alignment with switching axes.
def _mice_tumseg_raw_trafo(raw):
    raw = sam_training.util.normalize_to_8bit(raw)
    raw = raw.transpose(0, 2, 1)
    return raw


# Adjusting the data alignment with switching axes.
def _mice_tumseg_label_trafo(labels):
    labels = connected_components(labels).astype(labels.dtype)
    labels = labels.transpose(0, 2, 1)
    return labels


def get_mi_qualitative_plots(dataset_name, experiment_folder, view_napari=False, n_samples=20):
    get_loaders = {
        "psfhs": lambda: medical.get_psfhs_loader(
            path=os.path.join(DATA_ROOT, "psfhs"), batch_size=1, patch_shape=(1, 512, 512), split="test",
            raw_transform=sam_training.identity, transform=_transform_identity,
            sampler=MinInstanceSampler(), resize_inputs=True,
        ),
        "amd_sd": lambda: medical.get_amd_sd_loader(
            path=os.path.join(DATA_ROOT, "amd_sd"), batch_size=1, patch_shape=(1, 512, 512), split="test",
            raw_transform=sam_training.identity, transform=_transform_identity, resize_inputs=True,
            label_transform=_amd_sd_label_trafo, sampler=MinInstanceSampler(min_size=10),
        ),
        "mice_tumseg": lambda: medical.get_mice_tumseg_loader(
            path=os.path.join(DATA_ROOT, "mice_tumseg"), batch_size=1, patch_shape=(1, 512, 512), ndim=2, split="test",
            raw_transform=_mice_tumseg_raw_trafo, label_transform=_mice_tumseg_label_trafo,
            transform=_transform_identity, sampler=MinInstanceSampler(min_size=25), resize_inputs=True,
        ),
        "papila": lambda: medical.get_papila_loader(
            path=os.path.join(DATA_ROOT, "papila"), batch_size=1, patch_shape=(1, 512, 512), split="test",
            task="cup", raw_transform=sam_training.identity, transform=_transform_identity,
            sampler=MinInstanceSampler(), resize_inputs=True,
        ),
        "motum": lambda: medical.get_motum_loader(
            path=os.path.join(DATA_ROOT, "motum"), patch_shape=(1, 512, 512), ndim=2, batch_size=1, split="test",
            modality="flair", raw_transform=sam_training.util.normalize_to_8bit, transform=_transform_identity,
            sampler=MinInstanceSampler(min_size=50), resize_inputs=True,
        ),
        "jsrt": lambda: medical.get_jsrt_loader(
            path=os.path.join(DATA_ROOT, "jsrt"), batch_size=1, patch_shape=(512, 512), split="test",
            choice="Segmentation02", raw_transform=sam_training.identity, transform=_transform_identity,
            label_transform=_jsrt_label_trafo, sampler=MinInstanceSampler(), resize_inputs=True,
        ),
        # NEW DATASETS
        "sega": lambda: medical.get_sega_loader(
            path=os.path.join(DATA_ROOT, "sega"), batch_size=1, patch_shape=(1, 512, 512), data_choice="Rider",
            resize_inputs=True, raw_transform=sam_training.util.normalize_to_8bit, transform=_transform_identity,
            label_transform=_amd_sd_label_trafo,  # I'm too lazy to change fn.names, but it's just connected components.
            sampler=MinInstanceSampler(), shuffle=True,
        ),
        "dsad": lambda: medical.get_dsad_loader(
            path=os.path.join(DATA_ROOT, "dsad"), patch_shape=(1, 512, 512), batch_size=1, organ="liver",
            resize_inputs=True, raw_transform=sam_training.identity, transform=_transform_identity,
            label_transform=_amd_sd_label_trafo,  # lazy here as well.
            sampler=MinInstanceSampler(min_size=25), shuffle=True,
        ),
        "ircadb": lambda: medical.ircadb.get_ircadb_loader(
            path=os.path.join(DATA_ROOT, "ircadb"), batch_size=1, patch_shape=(1, 512, 512), label_choice="liver",
            split="test", resize_inputs=True, sampler=MinInstanceSampler(min_size=50), n_samples=100,
            raw_transform=sam_training.identity, transform=_transform_identity, label_transform=_amd_sd_label_trafo,
        ),
    }

    # Get the dataloader
    loader = get_loaders[dataset_name]()

    # Get the desired paths.
    standard_model = "vit_b_medical_imaging"
    finetuned_model = "vit_b_medical_imaging_lora"
    intermediate_model = "vit_b_medical_imaging_full_ft"

    # Get the checkpoints paths per dataset.
    checkpoint1 = None  # Use the MedicoSAM FM as it is.
    checkpoint2 = os.path.join(MODEL_ROOT, "lora", f"{dataset_name}_sam", "best.pt")
    checkpoint3 = os.path.join(MODEL_ROOT, "full_ft", f"{dataset_name}_sam", "best.pt")

    # Run evaluation for qualitative comparison.
    output_folder = os.path.join(
        experiment_folder, "model_comparison", dataset_name, f"{standard_model}-{finetuned_model}"
    )
    plot_folder = os.path.join(experiment_folder, "candidates", dataset_name)
    if not os.path.exists(output_folder):
        generate_data_for_model_comparison(
            loader=loader,
            output_folder=output_folder,
            model_type1=standard_model,
            model_type2=finetuned_model[:5],
            model_type3=intermediate_model[:5],
            n_samples=n_samples,
            checkpoint1=checkpoint1,
            checkpoint2=checkpoint2,
            checkpoint3=checkpoint3,
            peft_kwargs2={"rank": 32},  # for the LoRA model, i.e. the reference model to compare with.
        )

    model_comparison(
        output_folder=output_folder,
        n_images_per_sample=10,
        min_size=100,
        plot_folder=plot_folder,
        point_radius=3,
        outline_dilation=0,
        have_model3=intermediate_model is not None,
        enhance_image=True,
    )
    if view_napari:
        from micro_sam.evaluation.model_comparison import model_comparison_with_napari
        model_comparison_with_napari(output_folder, show_points=True)


def main(args):
    get_mi_qualitative_plots(
        dataset_name=args.dataset,
        experiment_folder=args.experiment_folder,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-e", "--experiment_folder", type=str, default="./results")
    args = parser.parse_args()
    main(args)
