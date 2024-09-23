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


def _fetch_loaders(dataset_name, data_root):

    if dataset_name == "covid_if":

        # 1, Covid IF does not have internal splits. For this example I chose first 10 samples for training,
        # and next 3 samples for validation, left the rest for testing.

        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True,
            min_size=0
        )
        raw_transform = RawTrafo(do_padding=False, do_rescaling=True)

        train_loader = light_microscopy.get_covid_if_loader(
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
        val_loader = light_microscopy.get_covid_if_loader(
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
        train_loader = light_microscopy.get_livecell_loader(
            path=os.path.join(data_root, "livecell"),
            patch_shape=(512, 512),
            split="train",
            batch_size=2,
            num_workers=16,
            cell_types=None,
            download=True,
            shuffle=True,
            label_transform=label_transform,
            raw_transform=raw_transform,
            label_dtype=torch.float32,
            sampler=MinInstanceSampler()
        )
        val_loader = light_microscopy.get_livecell_loader(
            path=os.path.join(data_root, "livecell"),
            patch_shape=(512, 512),
            split="val",
            batch_size=2,
            num_workers=16,
            cell_types=None,
            download=True,
            shuffle=True,
            label_transform=label_transform,
            raw_transform=raw_transform,
            label_dtype=torch.float32,
            sampler=MinInstanceSampler()
        )

    elif dataset_name == "orgasegment":
        # 2. OrgaSegment has internal splits provided. We follow the respective splits for our experiments.
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True,
            min_size=0
        )
        raw_transform = RawTrafo(do_padding=False, triplicate_dims=True)

        train_loader = light_microscopy.get_orgasegment_loader(
            path=os.path.join(data_root, "orgasegment"),
            patch_shape=(512, 512),
            split="train",
            batch_size=2,
            num_workers=16,
            shuffle=True,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
            sampler=MinInstanceSampler()
        )
        val_loader = light_microscopy.get_orgasegment_loader(
            path=os.path.join(data_root, "orgasegment"),
            patch_shape=(512, 512),
            split="val",
            batch_size=1,
            num_workers=16,
            download=True,
            raw_transform=raw_transform,
            label_transform=label_transform,
            sampler=MinInstanceSampler()
        )

    elif dataset_name == "mitolab_glycolytic_muscle":
        # 4. This dataset would need aspera-cli to be installed, I'll provide you with this data
        # ...
        train_rois = np.s_[0:175, :, :]
        val_rois = np.s_[175:225, :, :]

        raw_transform = RawTrafo((1, 512, 512), do_padding=False)
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=0
        )
        train_loader = electron_microscopy.cem.get_benchmark_loader(
            path=os.path.join(data_root, "mitolab"),
            dataset_id=3,
            batch_size=2,
            patch_shape=(1, 512, 512),
            download=False,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(),
            rois=train_rois,
            raw_transform=raw_transform,
            label_transform=label_transform,
            ndim=2
        )
        val_loader = electron_microscopy.cem.get_benchmark_loader(
            path=os.path.join(data_root, "mitolab"),
            dataset_id=3,
            batch_size=2,
            patch_shape=(1, 512, 512),
            download=False,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(),
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
        label_transform = ResizeLabelTrafo((512, 512))

        train_loader = electron_microscopy.get_platynereis_cilia_loader(
            path=os.path.join(data_root, "platynereis"),
            patch_shape=(1, 512, 512),
            ndim=2,
            batch_size=2,
            rois=train_rois,
            download=True,
            num_workers=16,
            shuffle=True,
            sampler=MinInstanceSampler(),
            raw_transform=raw_transform,
            label_transform=label_transform
        )
        val_loader = electron_microscopy.get_platynereis_cilia_loader(
            path=os.path.join(data_root, "platynereis"),
            patch_shape=(1, 512, 512),
            ndim=2,
            batch_size=1,
            rois=val_rois,
            download=True,
            num_workers=16,
            sampler=MinInstanceSampler(),
            raw_transform=raw_transform,
            label_transform=label_transform
        )

    elif dataset_name == "gonuclear":
        # Dataset contains 5 volumes. Use volumes 1-3 for training, volume 4 for validation and volume 5 for testing.

        train_loader = light_microscopy.get_gonuclear_loader(
            path=os.path.join(data_root, "gonuclear"),
            patch_shape=(1, 512, 512),
            batch_size=2,
            segmentation_task="nuclei",
            download=True,
            sample_ids=[1135, 1136, 1137],
            raw_transform=RawTrafo((1, 512, 512), do_rescaling=True),
            label_transform=ResizeLabelTrafo((512, 512)),
            num_workers=16,
            sampler=MinInstanceSampler(),
            ndim=2
        )
        val_loader = light_microscopy.get_gonuclear_loader(
            path=os.path.join(data_root, "gonuclear"),
            patch_shape=(1, 512, 512),
            batch_size=2,
            segmentation_task="nuclei",
            download=True,
            sample_ids=[1139],
            raw_transform=RawTrafo((1, 512, 512), do_rescaling=True),
            label_transform=ResizeLabelTrafo((512, 512)),
            num_workers=16,
            sampler=MinInstanceSampler(),
            ndim=2
        )

    else:
        raise ValueError(f"{dataset_name} is not a valid dataset name.")

    return train_loader, val_loader


def _verify_loaders():

    for dataset_name in ["covid_if", "livecell", "orgasegment", "mitolab_glycolytic_muscle", "platy_cilia",
                         "gonuclear"]:
        train_loader, val_loader = _fetch_loaders(dataset_name=dataset_name, data_root="/scratch/usr/nimcarot/data")

        #breakpoint()
        # NOTE: if using on the cluster, napari visualization won't work with "check_loader".
        # turn "plt=True" and provide path to save the matplotlib outputs of the loader.
        check_loader(train_loader, 8, plt=True, save_path=f"./{dataset_name}_train_loader.png")
        check_loader(val_loader, 8, plt=True, save_path=f"./{dataset_name}_val_loader.png")


if __name__ == "__main__":
    _verify_loaders()
