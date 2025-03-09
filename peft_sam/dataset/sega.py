"""The SegA dataset contains annotations for aorta segmentation in CT scans.

The dataset is from the publication https://doi.org/10.1007/978-3-031-53241-2.
Please cite it if you use this dataset for your research.
"""

import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple, Optional, Literal, List

from torch_em.data.datasets.medical import sega


def get_sega_paths(
    path: Union[os.PathLike, str],
    data_choice: Optional[Literal["KiTS", "Rider", "Dongyang"]] = None,
    download: bool = False,
    sample_range: Tuple[int, int] = None
) -> Tuple[List[str], List[str]]:
    """Get paths to the SegA data.

    Args:
        path: Filepath to a folder where the data is downloaded for further processing.
        data_choice: The choice of dataset.
        download: Whether to download the data if it is not present.
        sample_range: Range of samples to load from the dataset.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    if data_choice is None:
        data_choices = sega.URL.keys()
    else:
        if isinstance(data_choice, str):
            data_choices = [data_choice]

    data_dirs = [
        sega.get_sega_data(path=path, data_choice=data_choice, download=download) for data_choice in data_choices
    ]

    image_paths, gt_paths = [], []
    for data_dir in data_dirs:
        all_volumes_paths = glob(os.path.join(data_dir, "*", "*.nrrd"))
        for volume_path in all_volumes_paths:
            if volume_path.endswith(".seg.nrrd"):
                gt_paths.append(volume_path)
            else:
                image_paths.append(volume_path)

    # now let's wrap the volumes to nifti format
    fimage_dir = os.path.join(path, "data", "images")
    fgt_dir = os.path.join(path, "data", "labels")

    os.makedirs(fimage_dir, exist_ok=True)
    os.makedirs(fgt_dir, exist_ok=True)

    fimage_paths, fgt_paths = [], []
    for image_path, gt_path in zip(natsorted(image_paths), natsorted(gt_paths)):
        fimage_path = os.path.join(fimage_dir, f"{Path(image_path).stem}.nii.gz")
        fgt_path = os.path.join(fgt_dir, f"{Path(image_path).stem}.nii.gz")

        fimage_paths.append(fimage_path)
        fgt_paths.append(fgt_path)

        if os.path.exists(fimage_path) and os.path.exists(fgt_path):
            continue

        import nrrd
        import numpy as np
        import nibabel as nib

        image = nrrd.read(image_path)[0]
        gt = nrrd.read(gt_path)[0]

        image_nifti = nib.Nifti2Image(image, np.eye(4))
        gt_nifti = nib.Nifti2Image(gt, np.eye(4))

        nib.save(image_nifti, fimage_path)
        nib.save(gt_nifti, fgt_path)

    if sample_range is not None:
        start, stop = sample_range
        if start is None:
            start = 0
        if stop is None:
            stop = len(fimage_paths)
        fimage_paths = fimage_paths[start:stop]
        fgt_paths = fgt_paths[start:stop]

    return natsorted(fimage_paths), natsorted(fgt_paths)
