import os
from glob import glob
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch_em
from torch_em.data.datasets import util
from torch_em.data.datasets.light_microscopy.gonuclear import get_gonuclear_data


def get_gonuclear_paths(
    path: Union[os.PathLike, str],
    sample_ids: Optional[Union[int, Tuple[int, ...]]] = None,
    rois: Dict[int, Any] = {},
    download: bool = False
) -> List[str]:
    """Get paths to the GoNuclear data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        sample_ids: The sample ids to load. The valid sample ids are:
            1135, 1136, 1137, 1139, 1170. If none is given all samples will be loaded.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths for the stored data.
    """
    data_root = get_gonuclear_data(path, download)

    if sample_ids is None:
        paths = sorted(glob(os.path.join(data_root, "*.h5")))
        sample_ids = [1135, 1136, 1137, 1139, 1170]
    else:
        paths = []
        for sample_id in sample_ids:
            sample_path = os.path.join(data_root, f"{sample_id}.h5")
            if not os.path.exists(sample_path):
                raise ValueError(f"Invalid sample id {sample_id}.")
            paths.append(sample_path)
    data_rois = [rois.get(sample, np.s_[:, :, :]) for sample in sample_ids]
    return paths, data_rois


def get_gonuclear_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    segmentation_task: str = "nuclei",
    sample_ids: Optional[Union[int, Tuple[int, ...]]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    rois: Dict[int, Any] = {},
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the GoNuclear dataset for segmenting nuclei in 3d fluorescence microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        segmentation_task: The segmentation task. Either 'nuclei' or 'cells'.
        sample_ids: The sample ids to load. The valid sample ids are:
            1135, 1136, 1137, 1139, 1170. If none is given all samples will be loaded.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        rois: The region of interest to use for the data blocks
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
       The segmentation dataset.
    """
    paths, rois = get_gonuclear_paths(path, sample_ids, rois, download)

    if segmentation_task == "nuclei":
        raw_key = "raw/nuclei"
        label_key = "labels/nuclei"
    elif segmentation_task == "cells":
        raw_key = "raw/cells"
        label_key = "labels/cells"
    else:
        raise ValueError(f"Invalid segmentation task {segmentation_task}, expect one of 'cells' or 'nuclei'.")

    kwargs = util.update_kwargs(kwargs, "rois", rois)
    kwargs, _ = util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets,
    )

    return torch_em.default_segmentation_dataset(
        raw_paths=paths,
        raw_key=raw_key,
        label_paths=paths,
        label_key=label_key,
        patch_shape=patch_shape,
        **kwargs
    )


def get_gonuclear_loader(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    batch_size: int,
    segmentation_task: str = "nuclei",
    sample_ids: Optional[Union[int, Tuple[int, ...]]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    rois: Dict[int, Any] = {},
    **kwargs
) -> DataLoader:
    """Get the GoNuclear dataloader for segmenting nuclei in 3d fluorescence microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        segmentation_task: The segmentation task. Either 'nuclei' or 'cells'.
        sample_ids: The sample ids to load. The valid sample ids are:
            1135, 1136, 1137, 1139, 1170. If none is given all samples will be loaded.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        rois: The region of interest to use for the data blocks
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_gonuclear_dataset(
        path=path,
        patch_shape=patch_shape,
        segmentation_task=segmentation_task,
        sample_ids=sample_ids,
        offsets=offsets,
        boundaries=boundaries,
        binary=binary,
        download=download,
        rois=rois,
        **ds_kwargs,
    )
    return torch_em.get_data_loader(dataset, batch_size=batch_size, **loader_kwargs)
