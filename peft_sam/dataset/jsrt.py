"""The JSRT dataset contains annotations for lung segmentation
in chest X-Rays.

The database is located at http://db.jsrt.or.jp/eng.php
This dataset is from the publication https://doi.org/10.2214/ajr.174.1.1740071.
Please cite it if you use this dataset for a publication.
"""

import os
from glob import glob
from pathlib import Path
from typing import Optional, Union, Tuple, Literal, List

from torch.utils.data import Dataset, DataLoader

import torch_em
from torch_em.data.datasets import util
from torch_em.data.datasets.medical import jsrt


def get_jsrt_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'test'],
    choice: Optional[Literal['Segmentation01', 'Segmentation02']] = None,
    download: bool = False,
    sample_range: Tuple[int, int] = None
) -> Tuple[List[str], List[str]]:
    """Get paths to the JSRT data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The data split to use. Either 'train', or 'test'.
        choice: The choice of data subset. Either 'Segmentation01' or 'Segmentation02'.
        download: Whether to download the data if it is not present.
        sample_range: Range of samples to load from the dataset.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    available_splits = ["train", "test"]
    assert split in available_splits, f"{split} isn't a valid split choice. Please choose from {available_splits}."

    if choice is None:
        choice = list(jsrt.URL.keys())
    else:
        if isinstance(choice, str):
            choice = [choice]

    image_paths, gt_paths = [], []
    for per_choice in choice:
        jsrt.get_jsrt_data(path=path, download=download, choice=per_choice)

        if per_choice == "Segmentation01":
            root_dir = os.path.join(path, Path(jsrt.ZIP_PATH[per_choice]).stem, split)
            all_image_paths = sorted(glob(os.path.join(root_dir, "org", "*.png")))
            all_gt_paths = sorted(glob(os.path.join(root_dir, "label", "*.png")))

        elif per_choice == "Segmentation02":
            root_dir = os.path.join(path, Path(jsrt.ZIP_PATH[per_choice]).stem, "segmentation")
            all_image_paths = sorted(glob(os.path.join(root_dir, f"org_{split}", "*.bmp")))
            all_gt_paths = sorted(glob(os.path.join(root_dir, f"label_{split}", "*.png")))

        else:
            raise ValueError(f"{per_choice} is not a valid segmentation dataset choice.")

        image_paths.extend(all_image_paths)
        gt_paths.extend(all_gt_paths)

    if sample_range is not None:
        start, stop = sample_range
        if start is None:
            start = 0
        if stop is None:
            stop = len(image_paths)
        image_paths = image_paths[start:stop]
        gt_paths = gt_paths[start:stop]
    assert len(image_paths) == len(gt_paths)

    return image_paths, gt_paths


def get_jsrt_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
    choice: Optional[Literal['Segmentation01', 'Segmentation02']] = None,
    resize_inputs: bool = False,
    download: bool = False,
    sample_range: Tuple[int, int] = None,
    **kwargs
) -> Dataset:
    """Get the JSRT dataset for lung segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', or 'test'.
        choice: The choice of data subset. Either 'Segmentation01' or 'Segmentation02'.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        sample_range: Range of samples to load from the dataset.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_jsrt_paths(path, split, choice, download, sample_range)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": False}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths, raw_key=None, label_paths=gt_paths, label_key=None, patch_shape=patch_shape, **kwargs
    )


def get_jsrt_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'test'],
    choice: Optional[Literal['Segmentation01', 'Segmentation02']] = None,
    resize_inputs: bool = False,
    download: bool = False,
    sample_range: Tuple[int, int] = None,
    **kwargs
) -> DataLoader:
    """Get the JSRT dataloader for lung segmentation.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The data split to use. Either 'train', or 'test'.
        choice: The choice of data subset. Either 'Segmentation01' or 'Segmentation02'.
        resize_inputs: Whether to resize the inputs.
        download: Whether to download the data if it is not present.
        sample_range: Range of samples to load from the dataset.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_jsrt_dataset(path, patch_shape, split, choice, resize_inputs, download, sample_range, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
