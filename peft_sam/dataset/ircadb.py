"""The IRCADb dataset contains annotations for liver segmentation (and several other organs and structures)
in 3D CT scans.

The dataset is located at https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/.
This dataset is from the publication, referenced in the dataset link above.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Literal, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em
from torch_em.data.datasets import util
from torch_em.data.datasets.medical import ircadb


def get_ircadb_paths(
    path: Union[os.PathLike, str],
    split: Optional[Literal["train", "val", "test"]] = None,
    download: bool = False,
    sample_range: Tuple[int, int] = None,
) -> List[str]:
    """Get paths to the IRCADb data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        download: Whether to download the data if it is not present.
        sample_range: Range of samples to load from the dataset.

    Returns:
        List of filepaths for the volumetric data.
    """

    data_dir = ircadb.get_ircadb_data(path, download)
    volume_paths = natsorted(glob(os.path.join(data_dir, "*.h5")))

    # Create splits on-the-fly, if desired.
    if split is not None:
        if split == "train":
            volume_paths = volume_paths[:12]
        elif split == "val":
            volume_paths = volume_paths[12:15]
        elif split == "test":
            volume_paths = volume_paths[15:]
        else:
            raise ValueError(f"'{split}' is not a valid split.")
    if sample_range is not None:
        start, stop = sample_range
        if start is None:
            start = 0
        if stop is None:
            stop = len(volume_paths)
        volume_paths = volume_paths[start:stop]

    return volume_paths


def get_ircadb_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, ...],
    label_choice: str,
    split: Optional[Literal["train", "val", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    sample_range: Tuple[int, int] = None,
    **kwargs
) -> Dataset:
    """Get the IRCADb dataset for liver (and other organ) segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of labelled organs.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        sample_range: Range of samples to load from the dataset.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    volume_paths = get_ircadb_paths(path, split, download, sample_range)

    # Get the labels in the expected hierarchy name.
    assert isinstance(label_choice, str)
    label_choice = f"labels/{label_choice}"

    # Get the parameters for resizing inputs
    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=volume_paths,
        raw_key="raw",
        label_paths=volume_paths,
        label_key=label_choice,
        patch_shape=patch_shape,
        **kwargs
    )


def get_ircadb_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, ...],
    label_choice: str,
    split: Optional[Literal["train", "val", "test"]] = None,
    resize_inputs: bool = False,
    download: bool = False,
    sample_range: Tuple[int, int] = None,
    **kwargs
) -> DataLoader:
    """Get the IRCADb dataloader for liver (and other organ) segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        label_choice: The choice of labelled organs.
        split: The choice of data split.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        sample_range: Range of samples to load from the dataset.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_ircadb_dataset(
        path, patch_shape, label_choice, split, resize_inputs, download, sample_range, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
