import os
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

import torch_em
from torch_em.data import ImageCollectionDataset
from torch_em.data.datasets import util
from torch_em.data.datasets.light_microscopy.livecell import get_livecell_data, _download_livecell_annotations


def get_livecell_paths(
    path: Union[os.PathLike, str],
    split: str,
    sample_range: Optional[Tuple[int, int]] = None,
    download: bool = False,
    cell_types: Optional[Sequence[str]] = None,
    label_path: Optional[Union[os.PathLike, str]] = None,
) -> Tuple[List[str], List[str]]:
    """Get paths to the LIVECell data.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        download: Whether to download the data if it is not present.
        cell_types: The cell types for which to get the data paths.
        label_path: Optional path for loading the label data.
        sample_range: Id range of samples to load from the training dataset.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    get_livecell_data(path, download)
    image_paths, seg_paths = _download_livecell_annotations(path, split, download, cell_types, label_path)

    image_paths = sorted(image_paths)
    seg_paths = sorted(seg_paths)

    assert split in ["train", "val"]
    # replace images with adequate image
    if sample_range is not None:
        start, stop = sample_range
        if start is None:
            start = 0
        if stop is None:
            stop = len(image_paths)

        image_paths = image_paths[start:stop]
        seg_paths = seg_paths[start:stop]

    return image_paths, seg_paths


def get_livecell_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    sample_range: Optional[Tuple[int, int]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    cell_types: Optional[Sequence[str]] = None,
    label_path: Optional[Union[os.PathLike, str]] = None,
    label_dtype=torch.int64,
    **kwargs
) -> Dataset:
    """Get the LIVECell dataset for segmenting cells in phase-contrast microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        patch_shape: The patch shape to use for training.
        sample_range: Id range of samples to load from the training dataset.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        cell_types: The cell types for which to get the data paths.
        label_path: Optional path for loading the label data.
        label_dtype: The datatype of the label data.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    assert split in ("train", "val", "test")
    if cell_types is not None:
        assert isinstance(cell_types, (list, tuple)), \
            f"cell_types must be passed as a list or tuple instead of {cell_types}"

    image_paths, seg_paths = get_livecell_paths(path, split, sample_range, download, cell_types, label_path)

    kwargs = util.ensure_transforms(ndim=2, **kwargs)
    kwargs, label_dtype = util.add_instance_label_transform(
        kwargs, add_binary_target=True, label_dtype=label_dtype, offsets=offsets, boundaries=boundaries, binary=binary
    )

    return ImageCollectionDataset(
        raw_image_paths=image_paths,
        label_image_paths=seg_paths,
        patch_shape=patch_shape,
        label_dtype=label_dtype,
        **kwargs
    )


def get_livecell_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    sample_range: Optional[Tuple[int, int]] = None,
    download: bool = False,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    cell_types: Optional[Sequence[str]] = None,
    label_path: Optional[Union[os.PathLike, str]] = None,
    label_dtype=torch.int64,
    **kwargs
) -> DataLoader:
    """Get the LIVECell dataloader for segmenting cells in phase-contrast microscopy.

    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The data split to use. Either 'train', 'val' or 'test'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        sample_range: Id range of samples to load from the training dataset.
        download: Whether to download the data if it is not present.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        cell_types: The cell types for which to get the data paths.
        label_path: Optional path for loading the label data.
        label_dtype: The datatype of the label data.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_livecell_dataset(
        path, split, patch_shape, sample_range=sample_range, download=download,
        offsets=offsets, boundaries=boundaries, binary=binary,
        cell_types=cell_types, label_path=label_path, label_dtype=label_dtype, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
