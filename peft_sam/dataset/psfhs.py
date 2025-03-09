"""The PSFHS dataset contains annotations for segmentation of pubic symphysis and fetal head
in ultrasound images.

This dataset is located at https://zenodo.org/records/10969427.
The dataset is from the publication https://doi.org/10.1038/s41597-024-03266-4.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em

from torch_em.data.datasets import util
from torch_em.data.datasets.medical import psfhs


def get_psfhs_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'val', 'test'],
    download: bool = False,
    sample_range: Tuple[int, int] = None,
) -> Tuple[List[int], List[int]]:
    """Get paths to the PSFHS dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        download: Whether to download the data if it is not present.
        sample_range: Range of samples to load from the dataset.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = psfhs.get_psfhs_data(path, download)

    raw_paths = natsorted(glob(os.path.join(data_dir, "image_mha", "*.mha")))
    label_paths = natsorted(glob(os.path.join(data_dir, "label_mha", "*.mha")))

    if split == "train":
        raw_paths, label_paths = raw_paths[:900], label_paths[:900]
    elif split == "val":
        raw_paths, label_paths = raw_paths[900:1050], label_paths[900:1050]
    elif split == "test":
        raw_paths, label_paths = raw_paths[1050:], label_paths[1050:]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    assert len(raw_paths) == len(label_paths) and len(raw_paths) > 0

    if sample_range is not None:
        start, stop = sample_range
        if start is None:
            start = 0
        if stop is None:
            stop = len(raw_paths)
        raw_paths = raw_paths[start:stop]
        label_paths = label_paths[start:stop]
        assert all(os.path.exists(fp) for fp in raw_paths + label_paths), f"Invalid sample range {sample_range}"

    return raw_paths, label_paths


def get_psfhs_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    resize_inputs: bool = False,
    download: bool = False,
    sample_range: Tuple[int, int] = None,
    **kwargs
) -> Dataset:
    """Get the PSFHS dataset for segmentation of pubic symphysis and fetal head.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the patch shape.
        download: Whether to download the data if it is not present.
        sample_range: Range of samples to load from the dataset.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    raw_paths, label_paths = get_psfhs_paths(path, split, download, sample_range)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        ndim=2,
        is_seg_dataset=False,
        with_channels=True,
        patch_shape=patch_shape,
        **kwargs
    )


def get_psfhs_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    resize_inputs: bool = False,
    download: bool = False,
    sample_range: Tuple[int, int] = None,
    **kwargs
) -> DataLoader:
    """Get the PSFHS dataset for segmentation of pubic symphysis and fetal head.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        download: Whether to download the data if it is not present.
        resize_inputs: Whether to resize the inputs to the patch shape.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_psfhs_dataset(path, patch_shape, split, resize_inputs, download, sample_range, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
