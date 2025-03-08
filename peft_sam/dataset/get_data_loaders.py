import os

import numpy as np
from skimage.measure import label as connected_components

import torch

import torch_em
from torch_em.data import datasets
from torch_em.data.datasets import util
from torch_em.data import MinInstanceSampler
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training
from micro_sam.training.util import ResizeLabelTrafo

from ..util import RawTrafo

from . import (
    get_hpa_segmentation_loader, get_livecell_loader, get_gonuclear_loader, get_orgasegment_loader,
    get_psfhs_loader, get_papila_loader, get_motum_loader, get_jsrt_loader, get_amd_sd_loader, get_mice_tumseg_loader,
    get_sega_paths, get_ircadb_loader, get_dsad_paths
)


def _fetch_microscopy_loaders(
    dataset_name,
    data_root,
    train_sample_range=None,
    val_sample_range=None,
    train_rois=None,
    val_rois=None,
    n_train_samples=None,
    n_val_samples=None,
    batch_size=2,
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

        train_loader = datasets.get_covid_if_loader(
            path=os.path.join(data_root, "covid_if"),
            patch_shape=(512, 512),
            batch_size=batch_size,
            sample_range=train_sample_range,
            target="cells",
            num_workers=16,
            shuffle=True,
            download=True,
            sampler=sampler,
            raw_transform=raw_transform,
            label_transform=label_transform,
            n_samples=n_train_samples
        )
        val_loader = datasets.get_covid_if_loader(
            path=os.path.join(data_root, "covid_if"),
            patch_shape=(512, 512),
            batch_size=batch_size,
            sample_range=val_sample_range,
            target="cells",
            num_workers=16,
            download=True,
            sampler=sampler,
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
            min_size=5,
        )

        sampler = MinInstanceSampler(min_num_instances=25)

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
        if train_rois is None:
            train_rois = np.s_[0:175, :, :]
        if val_rois is None:
            val_rois = np.s_[175:225, :, :]

        raw_transform = RawTrafo((512, 512), do_padding=True)
        label_transform = ResizeLabelTrafo((512, 512), min_size=5)
        sampler = MinInstanceSampler(min_num_instances=5)

        train_loader = datasets.cem.get_benchmark_loader(
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
        val_loader = datasets.cem.get_benchmark_loader(
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

        train_loader = datasets.get_platynereis_cilia_loader(
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
        val_loader = datasets.get_platynereis_cilia_loader(
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
            n_samples=n_train_samples,
            num_workers=16,
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
            n_samples=n_val_samples,
            num_workers=16,
        )

    else:
        raise ValueError(f"'{dataset_name}' is not a valid microscopy dataset name.")

    return train_loader, val_loader


#
# MEDICAL IMAGING DATALOADERS AND IMPORTANT TRANSFORMS
#


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


# Ensures all labels are unique.
def _cc_label_trafo(labels):
    labels = connected_components(labels).astype(labels.dtype)
    return labels


# Normalize inputs
def _to_8bit(raw):
    raw = sam_training.util.normalize_to_8bit(raw)
    return raw


def _fetch_medical_loaders(
        dataset_name,
        data_root,
        train_sample_range=None,
        val_sample_range=None,
        train_rois=None,
        val_rois=None,
        n_train_samples=None,
        n_val_samples=None,
        batch_size=2,
):

    if dataset_name == "papila":

        def _get_papila_loaders(split):
            # Optic disc in fundus.
            return get_papila_loader(
                path=os.path.join(data_root, "papila"),
                batch_size=2 if split == "train" else 1,
                patch_shape=(1, 512, 512),
                split=split,
                task="cup",
                raw_transform=sam_training.identity,
                transform=_transform_identity,
                sampler=MinInstanceSampler(),
                resize_inputs=True,
                download=True,
                shuffle=True,
                num_workers=16,
                n_samples=200,
                sample_range=train_sample_range if split == "train" else val_sample_range
            )
        get_loaders = _get_papila_loaders

    elif dataset_name == "motum":

        def _get_motum_loaders(split):
            # Tumor segmentation in MRI.
            return get_motum_loader(
                path=os.path.join(data_root, "motum"),
                batch_size=2 if split == "train" else 1,
                patch_shape=(1, 512, 512),
                ndim=2,
                split=split,
                modality="flair",
                raw_transform=sam_training.util.normalize_to_8bit,
                transform=_transform_identity,
                sampler=MinInstanceSampler(min_size=50),
                n_samples=200,
                resize_inputs=True,
                download=True,
                shuffle=True,
                num_workers=16,
                sample_range=train_sample_range if split == "train" else val_sample_range
            )
        get_loaders = _get_motum_loaders

    elif dataset_name == "psfhs":

        def _get_psfhs_loaders(split):
            # Pubic symphysis and fetal head in US.
            return get_psfhs_loader(
                path=os.path.join(data_root, "psfhs"),
                batch_size=2 if split == "train" else 1,
                patch_shape=(1, 512, 512),
                split=split,
                raw_transform=sam_training.identity,
                transform=_transform_identity,
                sampler=MinInstanceSampler(),
                resize_inputs=True,
                download=True,
                shuffle=True,
                num_workers=16,
                n_samples=200,
                sample_range=train_sample_range if split == "train" else val_sample_range
            )
        get_loaders = _get_psfhs_loaders

    elif dataset_name == "jsrt":

        def _get_jsrt_loaders(split):
            # Lung and heart segmentation in X-Ray
            dataset = get_jsrt_loader(
                path=os.path.join(data_root, "jsrt"),
                batch_size=batch_size if split == "train" else 1,
                patch_shape=(512, 512),
                split="train",
                choice="Segmentation02",
                raw_transform=sam_training.identity,
                transform=_transform_identity,
                label_transform=_jsrt_label_trafo,
                sampler=MinInstanceSampler(),
                resize_inputs=True,
                download=True,
                shuffle=True,
                num_workers=16,
                n_samples=200,
                sample_range=train_sample_range if split == "train" else val_sample_range
            )

            # Get splits on the fly.
            val_fraction = 0.2
            generator = torch.Generator().manual_seed(42)
            train_ds, val_ds = torch.utils.data.random_split(
                dataset=dataset, lengths=[1 - val_fraction, val_fraction], generator=generator
            )

            if split == "train":
                return torch_em.get_data_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=16)
            else:
                return torch_em.get_data_loader(val_ds, batch_size=batch_size, shuffle=True, num_workers=16)

        get_loaders = _get_jsrt_loaders

    elif dataset_name == "amd_sd":

        def _get_amd_sd_loaders(split):
            # Lesion segmentation in OCT.
            loader = get_amd_sd_loader(
                path=os.path.join(data_root, "amd_sd"),
                batch_size=2 if split == "train" else 1,
                patch_shape=(1, 512, 512),
                split=split,
                raw_transform=sam_training.identity,
                transform=_transform_identity,
                label_transform=_amd_sd_label_trafo,
                sampler=MinInstanceSampler(min_size=10),
                resize_inputs=True,
                download=True,
                shuffle=True,
                num_workers=16,
                n_samples=200,
                sample_range=train_sample_range if split == "train" else val_sample_range
            )
            loader.dataset.max_sampling_attempts = 10000
            return loader

        get_loaders = _get_amd_sd_loaders

    elif dataset_name == "mice_tumseg":

        def _get_mice_tumseg_loaders(split):
            # Tumor segmentation in microCT.
            return get_mice_tumseg_loader(
                path=os.path.join(data_root, "mice_tumseg"),
                batch_size=2 if split == "train" else 1,
                patch_shape=(1, 512, 512),
                ndim=2,
                split=split,
                raw_transform=_mice_tumseg_raw_trafo,
                label_transform=_mice_tumseg_label_trafo,
                transform=_transform_identity,
                sampler=MinInstanceSampler(min_size=25),
                n_samples=250,
                resize_inputs=True,
                download=True,
                shuffle=True,
                num_workers=16,
                sample_range=train_sample_range if split == "train" else val_sample_range
            )
        get_loaders = _get_mice_tumseg_loaders

    elif dataset_name == "sega":
        # Aorta segmentation in CT.

        # Get one specific split of this data and use that for train-val-test.
        raw_paths, label_paths = get_sega_paths(
            path=os.path.join(data_root, "sega"), data_choice="Rider", download=True,
            sample_range=train_sample_range
        )
        # Create splits on-the-fly (use the first 12 volumes for train and val).
        raw_paths, label_paths = raw_paths[:12], label_paths[:12]

        # Get the resize transforms.
        patch_shape = (1, 512, 512)
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs={
                "raw_transform": _to_8bit,
                "transform": _transform_identity,
                "label_transform": _cc_label_trafo,
            },
            patch_shape=patch_shape, resize_inputs=True, resize_kwargs={"patch_shape": patch_shape, "is_rgb": False},
        )

        def _get_sega_loaders(split):
            dataset = torch_em.default_segmentation_dataset(
                raw_paths=raw_paths,
                raw_key="data",
                label_paths=label_paths,
                label_key="data",
                patch_shape=patch_shape,
                is_seg_dataset=True,
                ndim=2,
                n_samples=200,
                sampler=MinInstanceSampler(min_size=25),
                **kwargs
            )
            # Split the dataset into train and val.
            val_fraction = 0.2
            generator = torch.Generator().manual_seed(42)
            train_ds, val_ds = torch.utils.data.random_split(
                dataset=dataset, lengths=[1 - val_fraction, val_fraction], generator=generator
            )

            if split == "train":
                return torch_em.get_data_loader(train_ds, batch_size=2, shuffle=True, num_workers=16)
            else:
                return torch_em.get_data_loader(val_ds, batch_size=1, shuffle=True, num_workers=16)

        get_loaders = _get_sega_loaders

    elif dataset_name == "ircadb":
        # Liver segmentation in CT.

        def _get_ircadb_loaders(split):
            return get_ircadb_loader(
                path=os.path.join(data_root, "ircadb"),
                batch_size=2 if split == "train" else 1,
                patch_shape=(1, 512, 512),
                ndim=2,
                label_choice="liver",
                split=split,
                resize_inputs=True,
                download=True,
                raw_transform=_to_8bit,
                transform=_transform_identity,
                sampler=MinInstanceSampler(min_size=50),
                n_samples=250,
                shuffle=True,
                num_workers=16,
                sample_range=train_sample_range if split == "train" else val_sample_range
            )

        get_loaders = _get_ircadb_loaders

    elif dataset_name == "dsad":
        # Organ segmentation in Laparoscopy.

        # Get the image and label paths.
        raw_paths, label_paths = [], []
        for i, _organ in enumerate(["liver", "pancreas", "spleen", "colon"]):
            if train_sample_range[1] - train_sample_range[0] <= i:
                continue
            _rpaths, _lpaths = get_dsad_paths(
                path=os.path.join(data_root, "dsad"), organ=_organ, download=True,
                sample_range=train_sample_range
            )
            # Get only the first 250 per organ
            raw_paths.extend(_rpaths[:250])
            label_paths.extend(_lpaths[:250])

        # Get the resize transforms.
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs={
                "raw_transform": sam_training.identity,
                "transform": _transform_identity,
                "label_transform": _cc_label_trafo,
                "sampler": MinInstanceSampler(min_size=25),
            },
            patch_shape=(1, 512, 512),
            resize_inputs=True,
            resize_kwargs={"patch_shape": (1, 512, 512), "is_rgb": True},
        )

        def _get_dsad_loaders(split):
            dataset = torch_em.default_segmentation_dataset(
                raw_paths=raw_paths,
                raw_key=None,
                label_paths=label_paths,
                label_key=None,
                patch_shape=patch_shape,
                with_channels=True,
                is_seg_dataset=False,
                n_samples=200,
                **kwargs
            )
            # Split the data into training and val.
            val_fraction = 0.2
            generator = torch.Generator().manual_seed(42)
            train_ds, val_ds = torch.utils.data.random_split(
                dataset=dataset, lengths=[1 - val_fraction, val_fraction], generator=generator
            )

            if split == "train":
                return torch_em.get_data_loader(train_ds, batch_size=2, shuffle=True, num_workers=16)
            else:
                return torch_em.get_data_loader(val_ds, batch_size=1, shuffle=True, num_workers=16)

        get_loaders = _get_dsad_loaders

    else:
        raise ValueError(f"'{dataset_name}' is not a valid medical imaging dataset name.")

    train_loader, val_loader = get_loaders("train"), get_loaders("val")
    return train_loader, val_loader
