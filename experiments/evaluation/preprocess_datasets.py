import os
from tqdm import tqdm

import numpy as np
from skimage.measure import label as connected_components

from torch_em.data import datasets, MinInstanceSampler

from tukra.io import write_image

from micro_sam.training import identity
from micro_sam.training.util import normalize_to_8bit


def _transform_identity(raw, labels):  # This is done to avoid any transformations.
    return raw, labels


def _store_images(name, data_path, loader, view, is_rgb=False):
    raw_dir = os.path.join(data_path, "slices", "test", "raw")
    labels_dir = os.path.join(data_path, "slices", "test", "labels")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    counter = 0
    for x, y in tqdm(loader, desc=f"Preprocessing '{name}'"):
        x, y = x.squeeze().numpy(), y.squeeze().numpy()

        if is_rgb:  # Convert batch inputs to channels last.
            x = x.transpose(1, 2, 0)

        if view:
            import napari
            v = napari.Viewer()
            v.add_image(x, name="Image")
            v.add_labels(y.astype("uint8"), name="Labels")
            napari.run()

        fname = f"image_{counter:05}.tif"
        raw_path = os.path.join(raw_dir, fname)
        labels_path = os.path.join(labels_dir, fname)

        write_image(raw_path, x)
        write_image(labels_path, y)

        counter += 1


def _process_papila_data(data_path, view):
    loader = datasets.get_papila_loader(
        path=os.path.join(data_path, "papila"),
        batch_size=1,
        patch_shape=(1, 512, 512),
        split="test",
        task="cup",
        raw_transform=identity,
        transform=_transform_identity,
        sampler=MinInstanceSampler(),
        resize_inputs=True,
        download=True,
        shuffle=True,
        num_workers=16,
    )
    _store_images("papila", os.path.join(data_path, "papila"), loader, view, is_rgb=True)


def _process_motum_data(data_path, view):
    loader = datasets.get_motum_loader(
        path=os.path.join(data_path, "motum"),
        batch_size=1,
        patch_shape=(1, 512, 512),
        split="test",
        modality="flair",
        raw_transform=normalize_to_8bit,
        transform=_transform_identity,
        sampler=MinInstanceSampler(min_size=50),
        resize_inputs=True,
        n_samples=50,
        download=True,
        shuffle=True,
        num_workers=16,
    )
    _store_images("motum", os.path.join(data_path, "motum"), loader, view)


def _process_psfhs_data(data_path, view):
    loader = datasets.get_psfhs_loader(
        path=os.path.join(data_path, "psfhs"),
        batch_size=1,
        patch_shape=(1, 512, 512),
        split="test",
        raw_transform=identity,
        transform=_transform_identity,
        sampler=MinInstanceSampler(),
        resize_inputs=True,
        download=True,
        shuffle=True,
        num_workers=16,
    )
    _store_images("psfhs", os.path.join(data_path, "psfhs"), loader, view, is_rgb=True)


def _process_jsrt_data(data_path, view):
    def _label_trafo(labels):  # maps labels to expected semantic structure.
        neu_label = np.zeros_like(labels)
        lungs = (labels == 255)  # Labels for lungs
        lungs = connected_components(lungs)  # Ensure both lung volumes unique
        neu_label[lungs > 0] = lungs[lungs > 0]  # Map both lungs to new label.
        neu_label[labels == 85] = np.max(neu_label) + 1   # Belongs to heart labels.
        return neu_label

    loader = datasets.get_jsrt_loader(
        path=os.path.join(data_path, "jsrt"),
        batch_size=1,
        patch_shape=(512, 512),
        split="test",
        choice="Segmentation02",
        raw_transform=identity,
        transform=_transform_identity,
        label_transform=_label_trafo,
        sampler=MinInstanceSampler(),
        resize_inputs=True,
        download=True,
        shuffle=True,
        num_workers=16,
    )
    _store_images("jsrt", os.path.join(data_path, "jsrt"), loader, view)


def _process_amd_sd_data(data_path, view):

    def _amd_sd_label_trafo(labels):
        labels = connected_components(labels).astype(labels.dtype)
        return labels

    loader = datasets.get_amd_sd_loader(
        path=os.path.join(data_path, "amd_sd"),
        batch_size=1,
        patch_shape=(1, 512, 512),
        split="test",
        raw_transform=identity,
        transform=_transform_identity,
        label_transform=_amd_sd_label_trafo,
        sampler=MinInstanceSampler(min_num_instances=6),
        resize_inputs=True,
        n_samples=100,
        download=True,
        shuffle=True,
        num_workers=16,
    )
    loader.dataset.max_sampling_attempts = 10000
    _store_images("amd-sd", os.path.join(data_path, "amd_sd"), loader, view, is_rgb=True)


def _process_mice_tumseg_data(data_path, view):

    def _raw_trafo(raw):
        raw = normalize_to_8bit(raw)
        raw = raw.transpose(0, 2, 1)
        return raw

    def _label_trafo(labels):
        labels = connected_components(labels).astype(labels.dtype)
        labels = labels.transpose(0, 2, 1)
        return labels

    loader = datasets.get_mice_tumseg_loader(
        path=os.path.join(data_path, "mice_tumseg"),
        batch_size=1,
        patch_shape=(1, 512, 512),
        ndim=2,
        split="test",
        raw_transform=_raw_trafo,
        label_transform=_label_trafo,
        transform=_transform_identity,
        sampler=MinInstanceSampler(min_size=25),
        n_samples=100,
        resize_inputs=True,
        download=True,
        shuffle=True,
        num_workers=16,
    )
    _store_images("mice-tumorseg", os.path.join(data_path, "mice_tumseg"), loader, view)


def _process_sega(data_path, view):
    ...


def _process_ircadb(data_path, view):
    ...


def _process_dsad(data_path, view):
    ...


def main(args):
    data_path = args.input_path
    view = args.view

    # Download the medical imaging datasets
    # NOTE: uncomment the lines below to download datasets
    # from util import download_all_datasets
    # download_all_datasets(path=args.input_path, for_microscopy=False)

    # _process_papila_data(data_path, view)
    # _process_motum_data(data_path, view)
    # _process_psfhs_data(data_path, view)
    # _process_jsrt_data(data_path, view)
    # _process_amd_sd_data(data_path, view)
    # _process_mice_tumseg_data(data_path, view)

    # NEW DATASETS
    _process_sega(data_path, view)
    # _process_ircadb(data_path, view)
    # _process_dsad(data_path, view)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", type=str, default="/mnt/vast-nhr/projects/cidas/cca/experiments/peft_sam/data"
    )
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    main(args)
