import os
from glob import glob
from typing import Tuple, Union, Literal, List, Optional

from torch.utils.data import Dataset, DataLoader

import torch_em
from torch_em.data.datasets import util
from torch_em.data.datasets.light_microscopy.orgasegment import get_orgasegment_data


def get_orgasegment_paths(
    path: Union[os.PathLike, str],
    split: Literal["train", "val", "eval"],
    sample_range: Optional[Tuple[int, int]] = None,
    download: bool = False
) -> Tuple[List[str], List[str]]:
    """Get paths for the OrgaSegment data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split to download. Either 'train', 'val or 'eval'.
        sample_range: The range of samples to use for training.
        download: Whether to download the data if it is not present.

    Returns:
        List of filepaths to the image data.
        List of filepaths to the label data.
    """
    data_dir = get_orgasegment_data(path=path, split=split, download=download)
    image_paths = sorted(glob(os.path.join(data_dir, "*_img.jpg")))
    label_paths = sorted(glob(os.path.join(data_dir, "*_masks_organoid.png")))

    if sample_range is not None:
        start, stop = sample_range
        if start is None:
            start = 0
        if stop is None:
            stop = len(image_paths)
        image_paths = image_paths[start:stop]
        label_paths = label_paths[start:stop]
    return image_paths, label_paths


def get_orgasegment_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "eval"],
    sample_range: Optional[Tuple[int, int]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> Dataset:
    """Get the OrgaSegment dataset for organoid segmentation

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        patch_shape: The patch shape to use for training.
        split: The split to download. Either 'train', 'val or 'eval'.
        sample_range: The range of samples to use for training.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert split in ["train", "val", "eval"]

    image_paths, label_paths = get_orgasegment_paths(path, split, sample_range, download)

    kwargs, _ = util.add_instance_label_transform(kwargs, add_binary_target=True, binary=binary, boundaries=boundaries)

    return torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=label_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )


def get_orgasegment_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal["train", "val", "eval"],
    sample_range: Optional[Tuple[int, int]] = None,
    boundaries: bool = False,
    binary: bool = False,
    download: bool = False,
    **kwargs
) -> DataLoader:
    """Get the OrgaSegment dataloader for organoid segmentation

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The split to download. Either 'train', 'val or 'eval'.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_orgasegment_dataset(path, patch_shape, split, sample_range, boundaries, binary, download, **ds_kwargs)
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
