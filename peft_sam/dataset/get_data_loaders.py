import os

import numpy as np
from skimage.measure import label as connected_components

import torch

from torch_em.data import datasets
from torch_em.data import MinInstanceSampler
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training
from micro_sam.training.util import ResizeLabelTrafo

from ..util import RawTrafo
from .hpa import get_hpa_segmentation_loader


def _fetch_microscopy_loaders(dataset_name, data_root):

    if dataset_name == "covid_if":

        # 1, Covid IF does not have internal splits. For this example I chose first 10 samples for training,
        # and next 3 samples for validation, left the rest for testing.

        raw_transform = RawTrafo(desired_shape=(512, 512))
        label_transform = ResizeLabelTrafo((512, 512))

        train_loader = datasets.get_covid_if_loader(
            path=os.path.join(data_root, "covid_if"),
            patch_shape=(512, 512),
            batch_size=2,
            sample_range=(0, 10),
            target="cells",
            num_workers=16,
            shuffle=True,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform
        )
        val_loader = datasets.get_covid_if_loader(
            path=os.path.join(data_root, "covid_if"),
            patch_shape=(512, 512),
            batch_size=1,
            sample_range=(10, 13),
            target="cells",
            num_workers=16,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
        )

    elif dataset_name == "livecell":
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True,
            min_size=25
        )
        raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]
        train_loader = datasets.get_livecell_loader(
            path=os.path.join(data_root, "livecell"),
            patch_shape=(520, 704),
            split="train",
            batch_size=2,
            num_workers=16,
            cell_types=None,
            download=True,
            shuffle=True,
            label_transform=label_transform,
            raw_transform=raw_transform,
            label_dtype=torch.float32,
            sampler=MinInstanceSampler(min_num_instances=25)
        )
        val_loader = datasets.get_livecell_loader(
            path=os.path.join(data_root, "livecell"),
            patch_shape=(520, 704),
            split="val",
            batch_size=1,
            num_workers=16,
            cell_types=None,
            download=True,
            shuffle=True,
            label_transform=label_transform,
            raw_transform=raw_transform,
            label_dtype=torch.float32,
            sampler=MinInstanceSampler(min_num_instances=25)
        )

    elif dataset_name == "orgasegment":
        # 2. OrgaSegment has internal splits provided. We follow the respective splits for our experiments.

        raw_transform = RawTrafo(desired_shape=(512, 512), triplicate_dims=True, do_padding=False)
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True,
            min_size=5
        )

        train_loader = datasets.get_orgasegment_loader(
            path=os.path.join(data_root, "orgasegment"),
            patch_shape=(512, 512),
            split="train",
            batch_size=2,
            num_workers=16,
            shuffle=True,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
            sampler=MinInstanceSampler(min_num_instances=5)
        )
        val_loader = datasets.get_orgasegment_loader(
            path=os.path.join(data_root, "orgasegment"),
            patch_shape=(512, 512),
            split="val",
            batch_size=1,
            num_workers=16,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
            sampler=MinInstanceSampler(min_num_instances=5)
        )

    elif dataset_name == "mitolab_glycolytic_muscle":
        # 4. This dataset would need aspera-cli to be installed, I'll provide you with this data
        # ...
        train_rois = np.s_[0:175, :, :]
        val_rois = np.s_[175:225, :, :]

        raw_transform = RawTrafo((512, 512), do_padding=True)
        label_transform = ResizeLabelTrafo((512, 512), min_size=5)
        train_loader = datasets.cem.get_benchmark_loader(
            path=os.path.join(data_root, "mitolab"),
            dataset_id=3,
            batch_size=2,
            patch_shape=(1, 512, 512),
            download=False,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(min_num_instances=5),
            rois=train_rois,
            raw_transform=raw_transform,
            label_transform=label_transform,
            ndim=2
        )
        val_loader = datasets.cem.get_benchmark_loader(
            path=os.path.join(data_root, "mitolab"),
            dataset_id=3,
            batch_size=2,
            patch_shape=(1, 512, 512),
            download=False,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(min_num_instances=5),
            rois=val_rois,
            raw_transform=raw_transform,
            label_transform=label_transform,
            ndim=2
        )

    elif dataset_name == "platy_cilia":
        # 5. Platynereis (Cilia)
        # the logic used here is: I use the first 85 slices per volume from the training split for training
        # and the next ~10-15 slices per volume from the training split for validation
        # and we use the third volume from the trainng set for testing
        train_rois = {
            1: np.s_[0:85, :, :], 2: np.s_[0:85, :, :]
        }
        val_rois = {
            1: np.s_[85:, :, :], 2: np.s_[85:, :, :]
        }

        raw_transform = RawTrafo((1, 512, 512))
        label_transform = ResizeLabelTrafo((512, 512), min_size=3)

        train_loader = datasets.get_platynereis_cilia_loader(
            path=os.path.join(data_root, "platynereis"),
            patch_shape=(1, 512, 512),
            ndim=2,
            batch_size=2,
            rois=train_rois,
            download=True,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(min_num_instances=3),
            raw_transform=raw_transform,
            label_transform=label_transform
        )
        val_loader = datasets.get_platynereis_cilia_loader(
            path=os.path.join(data_root, "platynereis"),
            patch_shape=(1, 512, 512),
            ndim=2,
            batch_size=1,
            rois=val_rois,
            download=True,
            num_workers=16,
            sampler=MinInstanceSampler(min_num_instances=3),
            raw_transform=raw_transform,
            label_transform=label_transform
        )

    elif dataset_name == "gonuclear":
        # Dataset contains 5 volumes. Use volumes 1-3 for training, volume 4 for validation and volume 5 for testing.

        train_loader = datasets.get_gonuclear_loader(
            path=os.path.join(data_root, "gonuclear"),
            patch_shape=(1, 512, 512),
            batch_size=2,
            segmentation_task="nuclei",
            download=True,
            sample_ids=[1135, 1136, 1137],
            raw_transform=RawTrafo((512, 512)),
            label_transform=ResizeLabelTrafo((512, 512), min_size=5),
            num_workers=16,
            sampler=MinInstanceSampler(min_num_instances=5),
            ndim=2
        )
        val_loader = datasets.get_gonuclear_loader(
            path=os.path.join(data_root, "gonuclear"),
            patch_shape=(1, 512, 512),
            batch_size=2,
            segmentation_task="nuclei",
            download=True,
            sample_ids=[1139],
            raw_transform=RawTrafo((512, 512)),
            label_transform=ResizeLabelTrafo((512, 512), min_size=5),
            num_workers=16,
            sampler=MinInstanceSampler(min_num_instances=5),
            ndim=2
        )

    elif dataset_name == "hpa":

        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=5
        )
        train_loader = get_hpa_segmentation_loader(
            path=os.path.join(data_root, "hpa"),
            split="train",
            patch_shape=(512, 512),
            batch_size=2,
            channels=["protein"],
            download=True,
            n_workers_preproc=16,
            raw_transform=RawTrafo((512, 512), do_padding=False),
            label_transform=label_transform,
            sampler=MinInstanceSampler(min_num_instances=5),
            ndim=2
        )
        val_loader = get_hpa_segmentation_loader(
            path=os.path.join(data_root, "hpa"),
            split="val",
            patch_shape=(512, 512),
            batch_size=2,
            channels=["protein"],
            download=True,
            n_workers_preproc=16,
            raw_transform=RawTrafo((512, 512), do_padding=False),
            label_transform=label_transform,
            sampler=MinInstanceSampler(min_num_instances=5),
            ndim=2
        )

    else:
        raise ValueError(f"{dataset_name} is not a valid microscopy dataset name.")

    return train_loader, val_loader


def _fetch_medical_loaders(dataset_name, data_root):
    def _transform_identity(raw, labels):  # This is done to avoid any transformations.
        return raw, labels

    if dataset_name == "papila":

        def _get_papila_loaders(split):
            # Optic disc in fundus.
            return datasets.get_papila_loader(
                path=os.path.join(data_root, "papila"),
                batch_size=2 if split == "train" else 1,
                patch_shape=(1, 512, 512),
                split=split,
                task="cup",
                raw_transform=sam_training.identity,
                transform=_transform_identity,
                sampler=MinInstanceSampler(),
                resize_inputs=True,
            )
        get_loaders = _get_papila_loaders

    elif dataset_name == "motum":

        def _get_motum_loaders(split):
            # Tumor segmentation in MRI.
            return datasets.get_motum_loader(
                path=os.path.join(data_root, "motum"),
                batch_size=2 if split == "train" else 1,
                patch_shape=(1, 512, 512),
                split=split,
                modality="flair",
                raw_transform=sam_training.identity,
                transform=_transform_identity,
                sampler=MinInstanceSampler(min_size=50),
                resize_inputs=True,
            )
        get_loaders = _get_motum_loaders

    elif dataset_name == "psfhs":

        def _get_psfhs_loaders(split):
            # Pubic symphysis and fetal head in US.
            return datasets.get_psfhs_loader(
                path=os.path.join(data_root, "psfhs"),
                batch_size=2 if split == "train" else 1,
                patch_shape=(1, 512, 512),
                split=split,
                raw_transform=sam_training.identity,
                transform=_transform_identity,
                sampler=MinInstanceSampler(),
                resize_inputs=True,
            )
        get_loaders = _get_psfhs_loaders

    elif dataset_name == "jsrt":
        def _label_trafo(labels):  # maps labels to expected instance structure (to train for instance segmentation).
            neu_label = np.zeros_like(labels)

            # Labels for lungs
            lungs = (labels == 255)
            # Ensure both lung volumes unique
            lungs = connected_components(lungs)
            # Map both lungs to new label.
            neu_label[lungs > 0] = lungs[lungs > 0]

            # Belongs to heart labels.
            neu_label[labels == 85] = np.max(neu_label) + 1
            return neu_label

        def _get_jsrt_loaders(split):
            # Lung and heart segmentation in X-Ray
            return datasets.get_jsrt_loader(
                path=os.path.join(data_root, "jsrt"),
                batch_size=2 if split == "train" else 1,
                patch_shape=(512, 512),
                split="train",
                choice="Segmentation02",
                raw_transform=sam_training.identity,
                transform=_transform_identity,
                label_transform=_label_trafo,
                sampler=MinInstanceSampler(),
                resize_inputs=True,
            )
        get_loaders = _get_jsrt_loaders

    elif dataset_name == "amd_sd":

        def _label_trafo(labels):
            labels = connected_components(labels).astype(labels.dtype)
            return labels

        def _get_amd_sd_loaders(split):
            # Lesion segmentation in OCT.
            loader = datasets.get_amd_sd_loader(
                path=os.path.join(data_root, "amd_sd"),
                batch_size=2 if split == "train" else 1,
                patch_shape=(1, 512, 512),
                split=split,
                raw_transform=sam_training.identity,
                transform=_transform_identity,
                label_transform=_label_trafo,
                sampler=MinInstanceSampler(min_size=10),
                resize_inputs=True,
            )
            loader.dataset.max_sampling_attempts = 10000
            return loader

        get_loaders = _get_amd_sd_loaders

    elif dataset_name == "mice_tumseg":
        # Adjusting the data alignment with switching axes.
        def _raw_trafo(raw):
            raw = raw.transpose(0, 2, 1)
            return raw

        def _label_trafo(labels):
            labels = labels.transpose(0, 2, 1)
            return labels

        def _get_mice_tumseg_loaders(split):
            # Tumor segmentation in microCT.
            return datasets.get_mice_tumseg_loader(
                path=os.path.join(data_root, "mice_tumseg"),
                batch_size=1,
                patch_shape=(1, 512, 512),
                split="test",
                raw_transform=_raw_trafo,
                label_transform=_label_trafo,
                transform=_transform_identity,
                sampler=MinInstanceSampler(min_size=25),
                resize_inputs=True,
            )
        get_loaders = _get_mice_tumseg_loaders

    else:
        raise ValueError(f"{dataset_name} is not a valid medical imaging dataset name.")

    train_loader, val_loader = get_loaders("train"), get_loaders("val")
    return train_loader, val_loader
