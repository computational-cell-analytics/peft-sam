import os

import numpy as np

import torch

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.data.datasets import light_microscopy, electron_microscopy

import micro_sam.training as sam_training
from micro_sam.training.util import ResizeLabelTrafo

from ..util import RawTrafo
from . import (
    get_hpa_segmentation_loader, get_livecell_loader, get_gonuclear_loader, get_orgasegment_loader
)


def _fetch_loaders(
    dataset_name,
    data_root,
    train_sample_range=None,
    val_sample_range=None,
    train_rois=None,
    val_rois=None,
    n_train_samples=None,
    n_val_samples=None,
    batch_size=2
):
    if dataset_name == "covid_if":

        # 1, Covid IF does not have internal splits. For this example I chose first 10 samples for training,
        # and next 3 samples for validation, left the rest for testing.

        raw_transform = RawTrafo(desired_shape=(512, 512))
        label_transform = ResizeLabelTrafo((512, 512))

        sampler = MinInstanceSampler()
        if train_sample_range is None:
            train_sample_range = (0, 10)
        if val_sample_range is None:
            val_sample_range = (10, 13)
        train_loader = light_microscopy.get_covid_if_loader(
            path=os.path.join(data_root, "covid_if"),
            patch_shape=(512, 512),
            batch_size=batch_size,
            sample_range=train_sample_range,
            target="cells",
            num_workers=16,
            shuffle=True,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
            n_samples=n_train_samples
        )
        val_loader = light_microscopy.get_covid_if_loader(
            path=os.path.join(data_root, "covid_if"),
            patch_shape=(512, 512),
            batch_size=batch_size,
            sample_range=val_sample_range,
            target="cells",
            num_workers=16,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
            n_samples=n_val_samples
        )

    elif dataset_name == "livecell":
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True,
            min_size=25
        )

        sampler = MinInstanceSampler(min_num_instances=25)

        raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]
        train_loader = get_livecell_loader(
            path=os.path.join(data_root, "livecell"),
            patch_shape=(520, 704),
            split="train",
            batch_size=batch_size,
            num_workers=16,
            cell_types=None,
            download=True,
            shuffle=True,
            label_transform=label_transform,
            raw_transform=raw_transform,
            label_dtype=torch.float32,
            sampler=sampler,
            sample_range=train_sample_range,
            n_samples=n_train_samples
        )
        val_loader = get_livecell_loader(
            path=os.path.join(data_root, "livecell"),
            patch_shape=(520, 704),
            split="val",
            batch_size=batch_size,
            num_workers=16,
            cell_types=None,
            download=True,
            shuffle=True,
            label_transform=label_transform,
            raw_transform=raw_transform,
            label_dtype=torch.float32,
            sampler=sampler,
            sample_range=val_sample_range,
            n_samples=n_val_samples
        )

    elif dataset_name == "orgasegment":
        # 2. OrgaSegment has internal splits provided. We follow the respective splits for our experiments.

        raw_transform = RawTrafo(desired_shape=(512, 512), triplicate_dims=True, do_padding=False)
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True,
            min_size=5
        )

        sampler = MinInstanceSampler(min_num_instances=5)

        train_loader = get_orgasegment_loader(
            path=os.path.join(data_root, "orgasegment"),
            patch_shape=(512, 512),
            split="train",
            batch_size=batch_size,
            num_workers=16,
            shuffle=True,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
            sampler=sampler,
            sample_range=train_sample_range,
            n_samples=n_train_samples
        )
        val_loader = get_orgasegment_loader(
            path=os.path.join(data_root, "orgasegment"),
            patch_shape=(512, 512),
            split="val",
            batch_size=batch_size,
            num_workers=16,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
            sampler=sampler,
            sample_range=val_sample_range,
            n_samples=n_val_samples
        )

    elif dataset_name == "mitolab_glycolytic_muscle":
        # 4. This dataset would need aspera-cli to be installed, I'll provide you with this data
        # ...
        if train_rois is None:
            train_rois = np.s_[0:175, :, :]
        if val_rois is None:
            val_rois = np.s_[175:225, :, :]

        raw_transform = RawTrafo((512, 512), do_padding=True)
        label_transform = ResizeLabelTrafo((512, 512), min_size=5)

        sampler = MinInstanceSampler(min_num_instances=5)

        train_loader = electron_microscopy.cem.get_benchmark_loader(
            path=os.path.join(data_root, "mitolab"),
            dataset_id=3,
            batch_size=batch_size,
            patch_shape=(1, 512, 512),
            download=False,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(min_num_instances=5),
            rois=train_rois,
            raw_transform=raw_transform,
            label_transform=label_transform,
            ndim=2,
            n_samples=n_train_samples
        )
        val_loader = electron_microscopy.cem.get_benchmark_loader(
            path=os.path.join(data_root, "mitolab"),
            dataset_id=3,
            batch_size=batch_size,
            patch_shape=(1, 512, 512),
            download=False,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(min_num_instances=5),
            rois=val_rois,
            raw_transform=raw_transform,
            label_transform=label_transform,
            ndim=2,
            n_samples=n_val_samples
        )

    elif dataset_name == "platy_cilia":
        # 5. Platynereis (Cilia)
        # the logic used here is: I use the first 85 slices per volume from the training split for training
        # and the next ~10-15 slices per volume from the training split for validation
        # and we use the third volume from the trainng set for testing
        if train_rois is None:
            train_rois = {1: np.s_[0:85, :, :], 2: np.s_[0:85, :, :]}
        if val_rois is None:
            val_rois = {1: np.s_[85:, :, :], 2: np.s_[85:, :, :]}

        raw_transform = RawTrafo((1, 512, 512))
        label_transform = ResizeLabelTrafo((512, 512), min_size=3)

        sampler = MinInstanceSampler(min_num_instances=3)

        train_loader = electron_microscopy.get_platynereis_cilia_loader(
            path=os.path.join(data_root, "platynereis"),
            patch_shape=(1, 512, 512),
            ndim=2,
            batch_size=batch_size,
            rois=train_rois,
            download=True,
            num_workers=16,
            shuffle=True,
            sampler=sampler,
            raw_transform=raw_transform,
            label_transform=label_transform,
            sample_ids=list(train_rois.keys()),
            n_samples=n_val_samples
        )
        val_loader = electron_microscopy.get_platynereis_cilia_loader(
            path=os.path.join(data_root, "platynereis"),
            patch_shape=(1, 512, 512),
            ndim=2,
            batch_size=batch_size,
            rois=val_rois,
            download=True,
            num_workers=16,
            sampler=sampler,
            raw_transform=raw_transform,
            label_transform=label_transform,
            sample_ids=list(val_rois.keys()),
            n_samples=n_val_samples
        )

    elif dataset_name == "gonuclear":
        # Dataset contains 5 volumes. Use volumes 1-3 for training, volume 4 for validation and volume 5 for testing.

        if train_rois is None:
            train_rois = {1135: np.s_[:, :, :], 1136: np.s_[:, :, :], 1137: np.s_[:, :, :]}
        if val_rois is None:
            val_rois = {1139: np.s_[:, :, :]}

        train_loader = get_gonuclear_loader(
            path=os.path.join(data_root, "gonuclear"),
            patch_shape=(1, 512, 512),
            batch_size=batch_size,
            segmentation_task="nuclei",
            download=True,
            sample_ids=list(train_rois.keys()),
            raw_transform=RawTrafo((512, 512)),
            label_transform=ResizeLabelTrafo((512, 512), min_size=5),
            num_workers=16,
            sampler=MinInstanceSampler(min_num_instances=5),
            ndim=2,
            rois=train_rois,
            n_samples=n_train_samples
        )
        val_loader = get_gonuclear_loader(
            path=os.path.join(data_root, "gonuclear"),
            patch_shape=(1, 512, 512),
            batch_size=batch_size,
            segmentation_task="nuclei",
            download=True,
            sample_ids=list(val_rois.keys()),
            raw_transform=RawTrafo((512, 512)),
            label_transform=ResizeLabelTrafo((512, 512), min_size=5),
            num_workers=16,
            sampler=MinInstanceSampler(min_num_instances=5),
            ndim=2,
            rois=val_rois,
            n_samples=n_val_samples
        )

    elif dataset_name == "hpa":

        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=5
        )

        sampler = MinInstanceSampler(min_num_instances=5)

        train_loader = get_hpa_segmentation_loader(
            path=os.path.join(data_root, "hpa"),
            split="train",
            patch_shape=(512, 512),
            batch_size=batch_size,
            channels=["protein"],
            download=True,
            n_workers_preproc=16,
            raw_transform=RawTrafo((512, 512), do_padding=False),
            label_transform=label_transform,
            sampler=sampler,
            ndim=2,
            sample_range=train_sample_range,
            n_samples=n_train_samples
        )
        val_loader = get_hpa_segmentation_loader(
            path=os.path.join(data_root, "hpa"),
            split="val",
            patch_shape=(512, 512),
            batch_size=batch_size,
            channels=["protein"],
            download=True,
            n_workers_preproc=16,
            raw_transform=RawTrafo((512, 512), do_padding=False),
            label_transform=label_transform,
            sampler=sampler,
            ndim=2,
            sample_range=val_sample_range,
            n_samples=n_val_samples
        )

    else:
        raise ValueError(f"{dataset_name} is not a valid dataset name.")

    return train_loader, val_loader


def _verify_loaders():
    for dataset_name in [
        "covid_if", "livecell", "orgasegment", "mitolab_glycolytic_muscle", "platy_cilia", "gonuclear", "hpa"
    ]:
        train_loader, val_loader = _fetch_loaders(dataset_name=dataset_name, data_root="/scratch/usr/nimcarot/data")

        # NOTE: if using on the cluster, napari visualization won't work with "check_loader".
        # turn "plt=True" and provide path to save the matplotlib outputs of the loader.
        check_loader(train_loader, 8, plt=True, save_path=f"./{dataset_name}_train_loader.png")
        check_loader(val_loader, 8, plt=True, save_path=f"./{dataset_name}_val_loader.png")


if __name__ == "__main__":
    _verify_loaders()
