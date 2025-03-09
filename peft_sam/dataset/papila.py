"""The Papila dataset contains annotations for optic disc and optic cup
segmentation in Fundus images.

This dataset is located at https://figshare.com/articles/dataset/PAPILA/14798004/2.
The dataset is from the publication https://doi.org/10.1038/s41597-022-01388-1.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from typing import Union, Tuple, Literal, List

import torch_em

from torch_em.data.datasets import util
from torch_em.data.datasets.medical import papila


def get_papila_paths(
    path: Union[os.PathLike, str],
    split: Literal['train', 'val', 'test'],
    task: Literal["cup", "disc"] = "disc",
    expert_choice: Literal["exp1", "exp2"] = "exp1",
    download: bool = False,
    sample_range: Tuple[int, int] = None
) -> Tuple[List[str], List[str]]:
    """Get paths to the Papila dataset.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        split: The choice of data split.
        task: The choice of labels for specific task.
        expert_choice: The choice of expert annotator.
        download: Whether to download the data if it is not present.
        sample_range: Range of samples to load from the dataset.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = papila.get_papila_data(path=path, download=download)

    assert expert_choice in ["exp1", "exp2"], f"'{expert_choice}' is not a valid expert choice."
    assert task in ["cup", "disc"], f"'{task}' is not a valid task."

    image_paths = sorted(glob(os.path.join(data_dir, "FundusImages", "*.jpg")))
    gt_paths = papila._preprocess_labels(data_dir, image_paths, task, expert_choice)

    if split == "train":
        image_paths, gt_paths = image_paths[:350], gt_paths[:350]
    elif split == "val":
        image_paths, gt_paths = image_paths[350:400], gt_paths[350:400]
    elif split == "test":
        image_paths, gt_paths = image_paths[400:], gt_paths[400:]
    else:
        raise ValueError(f"'{split}' is not a valid split.")

    assert len(image_paths) == len(gt_paths) and len(image_paths) > 0
    if sample_range is not None:
        start, stop = sample_range
        if start is None:
            start = 0
        if stop is None:
            stop = len(image_paths)
        image_paths = image_paths[start:stop]
        gt_paths = gt_paths[start:stop]

    return image_paths, gt_paths


def get_papila_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    task: Literal["cup", "disc"] = "disc",
    expert_choice: Literal["exp1", "exp2"] = "exp1",
    resize_inputs: bool = False,
    download: bool = False,
    sample_range: Tuple[int, int] = None,
    **kwargs
):
    """Get the Papila dataset for segmentation of optic cup and optic disc in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        patch_shape: The patch shape to use for training.
        split: Te choice of data split.
        task: The choice of labels for specific task.
        expert_choice: The choice of expert annotator.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        sample_range: Range of samples to load from the dataset.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    image_paths, gt_paths = get_papila_paths(path, split, task, expert_choice, download, sample_range)

    if resize_inputs:
        resize_kwargs = {"patch_shape": patch_shape, "is_rgb": True}
        kwargs, patch_shape = util.update_kwargs_for_resize_trafo(
            kwargs=kwargs, patch_shape=patch_shape, resize_inputs=resize_inputs, resize_kwargs=resize_kwargs
        )

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_paths,
        raw_key=None,
        label_paths=gt_paths,
        label_key=None,
        patch_shape=patch_shape,
        is_seg_dataset=False,
        **kwargs
    )

    return dataset


def get_papila_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int],
    split: Literal['train', 'val', 'test'],
    task: Literal["cup", "disc"] = "disc",
    expert_choice: Literal["exp1", "exp2"] = "exp1",
    resize_inputs: bool = False,
    download: bool = False,
    sample_range: Tuple[int, int] = None,
    **kwargs
):
    """Get the Papila dataloader for segmentation of optic cup and optic disc in fundus images.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        split: The choice of data split.
        task: The choice of labels for specific task.
        expert_choice: The choice of expert annotator.
        resize_inputs: Whether to resize the inputs to the expected patch shape.
        download: Whether to download the data if it is not present.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_papila_dataset(
        path, patch_shape, split, task, expert_choice, resize_inputs, download, sample_range, **ds_kwargs
    )
    return torch_em.get_data_loader(dataset, batch_size, **loader_kwargs)
