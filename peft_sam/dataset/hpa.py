import os
from glob import glob
from typing import List, Optional, Sequence, Tuple, Union

from torch.utils.data import DataLoader, Dataset

from torch_em.data.datasets import util as _util
from torch_em import default_segmentation_dataset, get_data_loader
from torch_em.data.datasets.light_microscopy.hpa import get_hpa_segmentation_data


def get_hpa_segmentation_dataset(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    sample_range: Optional[Tuple[int, int]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    channels: Sequence[str] = ["microtubules", "protein", "nuclei", "er"],
    download: bool = False,
    n_workers_preproc: int = 8,
    **kwargs
) -> Dataset:
    """Get the HPA dataset for segmenting cells in confocal microscopy.
    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split for the dataset. Available splits are 'train', 'val' or 'test'.
        patch_shape: The patch shape to use for training.
        sample_range: Id range of samples to load from the training dataset.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        channels: The image channels to extract. Available channels are
            'microtubules', 'protein', 'nuclei' or 'er'.
        download: Whether to download the data if it is not present.
        n_workers_preproc: The number of workers to use for preprocessing.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.
    Returns:
       The segmentation dataset.
    """

    get_hpa_segmentation_data(path, download, n_workers_preproc)

    kwargs, _ = _util.add_instance_label_transform(
        kwargs, add_binary_target=True, binary=binary, boundaries=boundaries, offsets=offsets
    )

    kwargs, patch_shape = _util.update_kwargs_for_resize_trafo(
       kwargs=kwargs,
       patch_shape=patch_shape,
       resize_inputs=True,
       resize_kwargs={"patch_shape": patch_shape, "is_rgb": False},
    )

    kwargs = _util.update_kwargs(kwargs, "ndim", 2)
    kwargs = _util.update_kwargs(kwargs, "with_channels", True)

    if split == 'train':
        paths = glob(os.path.join(path, "train", "*.h5"))[:210]
    elif split == 'val':
        paths = glob(os.path.join(path, "train", "*.h5"))[210:]
    elif split == 'test':
        paths = glob(os.path.join(path, "val", "*.h5"))

    raw_key = [f"raw/{chan}" for chan in channels]
    label_key = "labels"

    if sample_range is not None:
        start, stop = sample_range
        if start is None:
            start = 0
        if stop is None:
            stop = len(paths)
        paths = paths[start:stop]

    return default_segmentation_dataset(paths, raw_key, paths, label_key, patch_shape, **kwargs)


def get_hpa_segmentation_loader(
    path: Union[os.PathLike, str],
    split: str,
    patch_shape: Tuple[int, int],
    batch_size: int,
    sample_range: Optional[Tuple[int, int]] = None,
    offsets: Optional[List[List[int]]] = None,
    boundaries: bool = False,
    binary: bool = False,
    channels: Sequence[str] = ["microtubules", "protein", "nuclei", "er"],
    download: bool = False,
    n_workers_preproc: int = 8,
    **kwargs
) -> DataLoader:
    """Get the HPA dataloader for segmenting cells in confocal microscopy.
    Args:
        path: Filepath to a folder where the downloaded data will be saved.
        split: The split for the dataset. Available splits are 'train', 'val' or 'test'.
        patch_shape: The patch shape to use for training.
        batch_size: The batch size for training.
        sample_range: Id range of samples to load from the training dataset.
        offsets: Offset values for affinity computation used as target.
        boundaries: Whether to compute boundaries as the target.
        binary: Whether to use a binary segmentation target.
        channels: The image channels to extract. Available channels are
            'microtubules', 'protein', 'nuclei' or 'er'.
        download: Whether to download the data if it is not present.
        n_workers_preproc: The number of workers to use for preprocessing.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.
    Returns:
       The DataLoader.
    """
    ds_kwargs, loader_kwargs = _util.split_kwargs(default_segmentation_dataset, **kwargs)
    dataset = get_hpa_segmentation_dataset(
        path, split, patch_shape, sample_range=sample_range,
        offsets=offsets, boundaries=boundaries, binary=binary,
        channels=channels, download=download, n_workers_preproc=n_workers_preproc,
        **ds_kwargs
    )
    loader = get_data_loader(dataset, batch_size, **loader_kwargs)
    return loader
