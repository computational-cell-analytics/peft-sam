"""The DSAD dataset contains annotations for abdominal organs in laparoscopy images.

This dataset is located at https://springernature.figshare.com/articles/dataset/The_Dresden_Surgical_Anatomy_Dataset_for_abdominal_organ_segmentation_in_surgical_data_science/21702600
The dataset is from the publication https://doi.org/10.1038/s41597-022-01719-2.
Please cite it if you use this dataset for your research.
"""  # noqa

import os
from glob import glob
from natsort import natsorted
from typing import Union, Tuple, List, Optional

from torch_em.data.datasets.medical import dsad


def get_dsad_paths(
    path: Union[os.PathLike, str],
    organ: Optional[str] = None,
    download: bool = False,
    sample_range: Tuple[int, int] = None
) -> Tuple[List[str], List[str]]:
    """Get paths to the DSAD data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        organ: The choice of organ annotations.
        download: Whether to download the data if it is not present.
        sample_range: Range of samples to load from the dataset.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    data_dir = dsad.get_dsad_data(path, download)

    if organ is None:
        organ = "*"
    else:
        assert organ in dsad.ORGANS, f"'{organ}' is not a valid organ choice."
        assert isinstance(organ, str), "We currently support choosing one organ at a time."

    image_paths = natsorted(glob(os.path.join(data_dir, organ, "*", "image*.png")))
    # Remove multi-label inputs.
    image_paths = [p for p in image_paths if "multilabel" not in p]

    # Get label paths.
    mask_paths = [p.replace("image", "mask") for p in image_paths]
    assert all([os.path.exists(p) for p in mask_paths])

    if sample_range is not None:
        start, stop = sample_range
        if start is None:
            start = 0
        if stop is None:
            stop = len(image_paths)
        image_paths = image_paths[start:stop]
        mask_paths = mask_paths[start:stop]

    assert image_paths and len(image_paths) == len(mask_paths)

    return image_paths, mask_paths
