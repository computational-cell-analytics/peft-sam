import os
from tqdm import tqdm

import random
import numpy as np
from skimage.measure import label as connected_components

import torch_em
from torch_em.data import datasets, MinInstanceSampler

from tukra.io import write_image

from micro_sam.training import identity
from micro_sam.training.util import normalize_to_8bit

from peft_sam.dataset.get_data_loaders import _to_8bit, _cc_label_trafo


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
    # The break-down below is done to choose custom test split.

    # Get one specific split of this data and use that for train-val-test.
    raw_paths, label_paths = datasets.medical.sega.get_sega_paths(
        path=os.path.join(data_path, "sega"), data_choice="Rider", download=True,
    )

    # First 12 images are for training. We use the rest for evaluation
    raw_paths, label_paths = raw_paths[12:], label_paths[12:]

    # Get the resize transforms.
    patch_shape = (1, 512, 512)
    kwargs, patch_shape = datasets.util.update_kwargs_for_resize_trafo(
        kwargs={"raw_transform": _to_8bit, "transform": _transform_identity, "label_transform": _cc_label_trafo},
        patch_shape=patch_shape, resize_inputs=True, resize_kwargs={"patch_shape": patch_shape, "is_rgb": False},
    )

    # Get the dataset.
    dataset = torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key="data",
        label_paths=label_paths,
        label_key="data",
        patch_shape=patch_shape,
        is_seg_dataset=True,
        ndim=2,
        n_samples=100,
        sampler=MinInstanceSampler(min_size=50),
        **kwargs
    )

    # Get the loader
    loader = torch_em.get_data_loader(dataset, batch_size=1, shuffle=True, num_workers=16)

    _store_images("sega", os.path.join(data_path, "sega"), loader, view)


def _process_ircadb(data_path, view):

    loader = datasets.get_ircadb_loader(
        path=os.path.join(data_path, "ircadb"),
        batch_size=1,
        patch_shape=(1, 512, 512),
        ndim=2,
        label_choice="liver",
        split="test",
        resize_inputs=True,
        download=True,
        raw_transform=_to_8bit,
        transform=_transform_identity,
        sampler=MinInstanceSampler(min_size=50),
        n_samples=100,
        shuffle=True,
        num_workers=16,
    )

    _store_images("ircadb", os.path.join(data_path, "ircadb"), loader, view)


def _process_dsad(data_path, view):
    # The break-down below is done to choose custom test split.

    # Get the image and label paths.
    raw_paths, label_paths = [], []
    for _organ in ["liver", "pancreas", "spleen", "colon"]:
        _rpaths, _lpaths = datasets.dsad.get_dsad_paths(
            path=os.path.join(data_path, "dsad"), organ=_organ, download=True,
        )

        # We need to do this step randomly, i.e. randomly pick 50 images
        _rpaths, _lpaths = _rpaths[250:], _lpaths[250:]  # first, exclude the train set.
        idxx = random.sample(range(len(_rpaths)), 25)  # choose the random indices first

        # Now we know the indices, Sample those images and corresponding labels.
        _rpaths = [_rpaths[i] for i in idxx]
        _lpaths = [_lpaths[i] for i in idxx]

        # The first 250 per organ are for training. We take randomly chosen 50 over the remaining.
        raw_paths.extend(_rpaths)
        label_paths.extend(_lpaths)

    # Get the resize transforms.
    kwargs, patch_shape = datasets.util.update_kwargs_for_resize_trafo(
        kwargs={
            "raw_transform": identity,
            "transform": _transform_identity,
            "label_transform": _cc_label_trafo,
            "sampler": MinInstanceSampler(min_size=25),
        },
        patch_shape=(1, 512, 512),
        resize_inputs=True,
        resize_kwargs={"patch_shape": (1, 512, 512), "is_rgb": True},
    )

    # Get the dataset.
    dataset = torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        with_channels=True,
        is_seg_dataset=False,
        **kwargs
    )

    # Get the dataloader.
    loader = torch_em.get_data_loader(dataset, batch_size=1, shuffle=True, num_workers=16)

    _store_images("dsad", os.path.join(data_path, "dsad"), loader, view, is_rgb=True)


def main(args):
    data_path = args.input_path
    view = args.view

    # Download the medical imaging datasets
    # NOTE: uncomment the lines below to download datasets
    # from util import download_all_datasets
    # download_all_datasets(path=args.input_path, for_microscopy=False)

    _process_papila_data(data_path, view)
    _process_motum_data(data_path, view)
    _process_psfhs_data(data_path, view)
    _process_jsrt_data(data_path, view)
    _process_amd_sd_data(data_path, view)
    _process_mice_tumseg_data(data_path, view)

    # NEW DATASETS
    _process_sega(data_path, view)
    _process_ircadb(data_path, view)
    _process_dsad(data_path, view)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", type=str, default="/mnt/vast-nhr/projects/cidas/cca/experiments/peft_sam/data"
    )
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    main(args)
