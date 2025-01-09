import os

import numpy as np
import torch

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import light_microscopy, electron_microscopy
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training
from micro_sam.training.util import ResizeLabelTrafo

from peft_sam.util import RawTrafo
from peft_sam import datasets


def _fetch_loaders(dataset_name, data_root):
    if dataset_name == "covid_if":

        # 1, Covid IF does not have internal splits. For this example I chose first 10 samples for training,
        # and next 3 samples for validation, left the rest for testing.

        raw_transform = RawTrafo(desired_shape=(512, 512))
        label_transform = ResizeLabelTrafo((512, 512))
        train_sample = (0, 1)    # replace with adequate image
        val_sample = (10, 11)
        sampler = MinInstanceSampler()
        # estimate the total number of patches

        train_loader = light_microscopy.get_covid_if_loader(
            path=os.path.join(data_root, "covid_if"), patch_shape=(512, 512),
            batch_size=1, target="cells", download=True, sampler=sampler, sample_range=train_sample,
        )
        val_loader = light_microscopy.get_covid_if_loader(
            path=os.path.join(data_root, "covid_if"), patch_shape=(512, 512),
            batch_size=1, target="cells", download=True, sampler=sampler, sample_range=val_sample,
        )

        train_loader = light_microscopy.get_covid_if_loader(
            path=os.path.join(data_root, "covid_if"),
            patch_shape=(512, 512),
            batch_size=1,
            sample_range=train_sample,
            target="cells",
            num_workers=16,
            shuffle=True,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
            n_samples=50 if len(train_loader) < 50 else None
        )
        val_loader = light_microscopy.get_covid_if_loader(
            path=os.path.join(data_root, "covid_if"),
            patch_shape=(512, 512),
            batch_size=1,
            sample_range=val_sample,
            target="cells",
            num_workers=16,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
            n_samples=50 if len(val_loader) < 50 else None
        )

    elif dataset_name == "livecell":
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True,
            min_size=25
        )

        train_sample = (2, 3)
        val_sample = (25, 26)

        sampler = MinInstanceSampler(min_num_instances=25)

        train_loader = datasets.get_livecell_loader(
            path=os.path.join(data_root, "livecell"), patch_shape=(512, 704),
            batch_size=1, split='train', download=True, sampler=sampler, sample_range=train_sample
        )
        val_loader = datasets.get_livecell_loader(
            path=os.path.join(data_root, "livecell"), patch_shape=(512, 704),
            batch_size=1, split='val', download=True, sampler=sampler, sample_range=val_sample
        )

        raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]
        train_loader = datasets.get_livecell_loader(
            path=os.path.join(data_root, "livecell"),
            patch_shape=(520, 704),
            split="train",
            batch_size=1,
            num_workers=16,
            cell_types=None,
            download=True,
            shuffle=True,
            label_transform=label_transform,
            raw_transform=raw_transform,
            label_dtype=torch.float32,
            sampler=sampler,
            sample_range=train_sample,
            n_samples=50 if len(val_loader) < 50 else None
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
            sampler=sampler,
            sample_range=val_sample,
            n_samples=50 if len(val_loader) < 50 else None
        )

    elif dataset_name == "orgasegment":
        # 2. OrgaSegment has internal splits provided. We follow the respective splits for our experiments.

        raw_transform = RawTrafo(desired_shape=(512, 512), triplicate_dims=True, do_padding=False)
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True,
            min_size=5
        )
        train_sample = (0, 1)
        val_sample = (0, 1)

        sampler = MinInstanceSampler(min_num_instances=5)

        train_loader = datasets.get_orgasegment_loader(
            path=os.path.join(data_root, "orgasegment"), patch_shape=(512, 512), split="train", batch_size=1,
            download=True, sampler=sampler, sample_range=train_sample
        )
        val_loader = datasets.get_orgasegment_loader(
            path=os.path.join(data_root, "orgasegment"), patch_shape=(512, 512), split="val", batch_size=1,
            download=True, sampler=sampler, sample_range=val_sample
        )

        train_loader = datasets.get_orgasegment_loader(
            path=os.path.join(data_root, "orgasegment"),
            patch_shape=(512, 512),
            split="train",
            batch_size=1,
            num_workers=16,
            shuffle=True,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
            sampler=sampler,
            sample_range=train_sample,
            n_samples=50 if len(train_loader) < 50 else None 
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
            sampler=sampler,
            sample_range=val_sample,
            n_samples=50 if len(val_loader) < 50 else None
        )

    elif dataset_name == "mitolab_glycolytic_muscle":
        # 4. This dataset would need aspera-cli to be installed, I'll provide you with this data
        # ...
        train_rois = np.s_[20:21, :, :]
        val_rois = np.s_[180:181, :, :]

        raw_transform = RawTrafo((512, 512), do_padding=True)
        label_transform = ResizeLabelTrafo((512, 512), min_size=5)

        sampler = MinInstanceSampler(min_num_instances=5)

        train_loader = electron_microscopy.cem.get_benchmark_loader(
            path=os.path.join(data_root, "mitolab"), dataset_id=3, batch_size=1, patch_shape=(1, 512, 512),
            download=False, sampler=sampler, rois=train_rois,
        )
        val_loader = electron_microscopy.cem.get_benchmark_loader(
            path=os.path.join(data_root, "mitolab"), dataset_id=3, batch_size=1, patch_shape=(1, 512, 512),
            download=False, sampler=sampler, rois=val_rois
        )
        train_loader = electron_microscopy.cem.get_benchmark_loader(
            path=os.path.join(data_root, "mitolab"),
            dataset_id=3,
            batch_size=1,
            patch_shape=(1, 512, 512),
            download=False,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(min_num_instances=5),
            rois=train_rois,
            raw_transform=raw_transform,
            label_transform=label_transform,
            ndim=2,
            n_samples=50 if len(val_loader) < 50 else None
        )
        val_loader = electron_microscopy.cem.get_benchmark_loader(
            path=os.path.join(data_root, "mitolab"),
            dataset_id=3,
            batch_size=1,
            patch_shape=(1, 512, 512),
            download=False,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(min_num_instances=5),
            rois=val_rois,
            raw_transform=raw_transform,
            label_transform=label_transform,
            ndim=2,
            n_samples=50 if len(val_loader) < 50 else None
        )

    elif dataset_name == "platy_cilia":
        # 5. Platynereis (Cilia)
        # the logic used here is: I use the first 85 slices per volume from the training split for training
        # and the next ~10-15 slices per volume from the training split for validation
        # and we use the third volume from the trainng set for testing
        train_rois = {
            2: np.s_[49:50, :, :]
        }
        val_rois = {
            2: np.s_[65:66, :, :]
        }

        raw_transform = RawTrafo((1, 512, 512))
        label_transform = ResizeLabelTrafo((512, 512), min_size=3)

        sampler = MinInstanceSampler(min_num_instances=3)
        train_loader = electron_microscopy.get_platynereis_cilia_loader(
            path=os.path.join(data_root, "platynereis"), patch_shape=(1, 512, 512), ndim=2, batch_size=1,
            download=True, sampler=sampler, rois=train_rois, sample_ids=[2]
        )
        val_loader = electron_microscopy.get_platynereis_cilia_loader(
            path=os.path.join(data_root, "platynereis"), patch_shape=(1, 512, 512), ndim=2, batch_size=1,
            download=True, sampler=sampler, rois=val_rois, sample_ids=[2]
        )

        train_loader = electron_microscopy.get_platynereis_cilia_loader(
            path=os.path.join(data_root, "platynereis"),
            patch_shape=(1, 512, 512),
            ndim=2,
            batch_size=1,
            rois=train_rois,
            download=True,
            num_workers=16,
            shuffle=True,
            sampler=sampler,
            raw_transform=raw_transform,
            label_transform=label_transform,
            sample_ids=[2],
            n_samples=50 if len(train_loader) < 50 else None
        )
        val_loader = electron_microscopy.get_platynereis_cilia_loader(
            path=os.path.join(data_root, "platynereis"),
            patch_shape=(1, 512, 512),
            ndim=2,
            batch_size=1,
            rois=val_rois,
            download=True,
            num_workers=16,
            sampler=sampler,
            raw_transform=raw_transform,
            label_transform=label_transform,
            sample_ids=[2],
            n_samples=50 if len(val_loader) < 50 else None
        )

    elif dataset_name == "gonuclear":
        # Dataset contains 5 volumes. Use volumes 1-3 for training, volume 4 for validation and volume 5 for testing.

        train_rois = {1136: np.s_[60:61, :, :]}
        val_rois = {1139: np.s_[40:41, :, :]}

        train_loader = datasets.get_gonuclear_loader(
            path=os.path.join(data_root, "gonuclear"), patch_shape=(1, 512, 512), batch_size=1,
            segmentation_task="nuclei", download=True, sample_ids=[1136], ndim=2, rois=train_rois
        )
        val_loader = datasets.get_gonuclear_loader(
            path=os.path.join(data_root, "gonuclear"), patch_shape=(1, 512, 512), batch_size=1,
            segmentation_task="nuclei", download=True, sample_ids=[1139], ndim=2, rois=val_rois
        )
        train_loader = datasets.get_gonuclear_loader(
            path=os.path.join(data_root, "gonuclear"),
            patch_shape=(1, 512, 512),
            batch_size=1,
            segmentation_task="nuclei",
            download=True,
            sample_ids=[1136],
            raw_transform=RawTrafo((512, 512)),
            label_transform=ResizeLabelTrafo((512, 512), min_size=5),
            num_workers=16,
            sampler=MinInstanceSampler(min_num_instances=5),
            ndim=2,
            rois=train_rois,
            n_samples=50 if len(train_loader) < 50 else None
        )
        val_loader = datasets.get_gonuclear_loader(
            path=os.path.join(data_root, "gonuclear"),
            patch_shape=(1, 512, 512),
            batch_size=1,
            segmentation_task="nuclei",
            download=True,
            sample_ids=[1139],
            raw_transform=RawTrafo((512, 512)),
            label_transform=ResizeLabelTrafo((512, 512), min_size=5),
            num_workers=16,
            sampler=MinInstanceSampler(min_num_instances=5),
            ndim=2,
            rois=val_rois,
            n_samples=50 if len(val_loader) < 50 else None
        )

    elif dataset_name == "hpa":

        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=5
        )

        sampler = MinInstanceSampler(min_num_instances=5)
        train_sample = (1, 2)
        val_sample = (1, 2)

        train_loader = datasets.get_hpa_segmentation_loader(
            path=os.path.join(data_root, "hpa"), patch_shape=(512, 512), batch_size=1, split="train",
            channels=["protein"], download=True, ndim=2, sample_range=train_sample, sampler=sampler
        )
        val_loader = datasets.get_hpa_segmentation_loader(
            path=os.path.join(data_root, "hpa"), patch_shape=(512, 512), batch_size=1, split="val",
            channels=["protein"], download=True, ndim=2, sample_range=val_sample, sampler=sampler
        )

        train_loader = datasets.get_hpa_segmentation_loader(
            path=os.path.join(data_root, "hpa"),
            split="train",
            patch_shape=(512, 512),
            batch_size=1,
            channels=["protein"],
            download=True,
            n_workers_preproc=16,
            raw_transform=RawTrafo((512, 512), do_padding=False),
            label_transform=label_transform,
            sampler=sampler,
            ndim=2,
            sample_range=train_sample,
            n_samples=50 if len(train_loader) < 50 else None
        )
        val_loader = datasets.get_hpa_segmentation_loader(
            path=os.path.join(data_root, "hpa"),
            split="val",
            patch_shape=(512, 512),
            batch_size=1,
            channels=["protein"],
            download=True,
            n_workers_preproc=16,
            raw_transform=RawTrafo((512, 512), do_padding=False),
            label_transform=label_transform,
            sampler=sampler,
            ndim=2,
            sample_range=val_sample,
            n_samples=50 if len(val_loader) < 50 else None,
        )

    else:
        raise ValueError(f"{dataset_name} is not a valid dataset name.")

    return train_loader, val_loader


def _verify_loaders():

    for dataset_name in ["covid_if", "livecell", "orgasegment", "mitolab_glycolytic_muscle", "platy_cilia",
                         "gonuclear", "hpa"]:
        if dataset_name != "covid_if":
            continue
        train_loader, val_loader = _fetch_loaders(dataset_name=dataset_name, data_root="/scratch/usr/nimcarot/data")

        # breakpoint()
        # NOTE: if using on the cluster, napari visualization won't work with "check_loader".
        # turn "plt=True" and provide path to save the matplotlib outputs of the loader.
        check_loader(train_loader, 1, plt=True, save_path=f"./{dataset_name}_train_loader.png")
        check_loader(val_loader, 1, plt=True, save_path=f"./{dataset_name}_val_loader.png")


if __name__ == "__main__":
    _verify_loaders()
