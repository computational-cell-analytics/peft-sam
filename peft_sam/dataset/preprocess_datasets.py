import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path

import h5py
import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

from torch_em.data import datasets
from torch_em.transform.generic import ResizeLongestSideInputs
from torch_em.transform.raw import normalize, normalize_percentile

import nifty.tools as nt

from elf.wrapper import RoiWrapper

from . import get_hpa_segmentation_dataset


ROOT = "/scratch/usr/nimcarot/data/"


def preprocess_data(dataset):

    if dataset == "covid_if":
        for_covid_if(os.path.join(ROOT, "covid_if", "slices"))
    elif dataset == "platynereis":
        for_platynereis(os.path.join(ROOT, "platynereis", "slices"), choice="cilia")
    elif dataset == "mitolab":
        for_mitolab(os.path.join(ROOT, "mitolab", "slices"))
    elif dataset == "orgasegment":
        for_orgasegment(os.path.join(ROOT, "orgasegment", "slices"))
    elif dataset == "gonuclear":
        for_gonuclear(os.path.join(ROOT, "gonuclear", "slices"))
    elif dataset == "hpa":
        for_hpa(os.path.join(ROOT, "hpa", "slices"))
    else:
        import warnings
        warnings.warn(
            f"Seems like the processing for '{dataset}' is either done or excluded. Please be aware of this."
        )


def convert_rgb(raw):
    raw = normalize_percentile(raw, axis=(1, 2))
    raw = np.mean(raw, axis=0)
    raw = normalize(raw)
    raw = raw * 255
    return raw


def has_foreground(label):
    if len(np.unique(label)) > 1:
        return True
    else:
        return False


def slices_overlap(slice1, slice2):
    """Check if two slices overlap in any dimension."""
    return not (
        slice1[0].stop <= slice2[0].start or  # No overlap in the first dimension
        slice1[0].start >= slice2[0].stop or
        slice1[1].stop <= slice2[1].start or  # No overlap in the second dimension
        slice1[1].start >= slice2[1].stop
    )


def get_best_crops(raw, labels, desired_shape):
    shape = raw.shape[:2]
    ndim = 2
    blocking = nt.blocking([0] * ndim, shape, desired_shape)
    n_blocks = blocking.numberOfBlocks

    # Total number of patches that fit into the image
    n_patches = (shape[0] // desired_shape[0]) * (shape[1] // desired_shape[1])
    assert n_patches > 0, "Chosen patch shape is too big for the image"

    patches = []
    valid_patches = []

    # get all possible blocks
    for block_id in range(n_blocks):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        # only allow blocks that have the desired shape
        if raw[bb].shape[:2] == desired_shape:
            patches.append(bb)

    # Extract the patches with the most instances and remove all patches that overlap with the best patch.
    # Repeat we have the correct number of valid patches.
    while len(valid_patches) < n_patches:
        max_idx = 0
        max_objects = 0
        for i in range(len(patches)):
            n_objects = len(np.unique(labels[patches[i]]))
            if n_objects > max_objects:
                max_objects = n_objects
                max_idx = i
        valid_patches.append(patches.pop(max_idx))
        for patch in patches:
            if slices_overlap(valid_patches[-1], patch):
                patches.remove(patch)

    raw_patches = []
    label_patches = []
    for slice_ in valid_patches:
        raw_patches.append(raw[slice_])
        label_patches.append(labels[slice_])

    return raw_patches, label_patches


def resize_image(raw, label, crop_shape):
    # for hpa
    resize_transform = ResizeLongestSideInputs(target_shape=crop_shape)
    resize_label_transform = ResizeLongestSideInputs(target_shape=crop_shape, is_label=True)

    raw = resize_transform(raw)
    label = resize_label_transform(label)

    return raw, label


def save_to_tif(i, raw, label, crop_shape, raw_dir, labels_dir, do_connected_components, slice_prefix_name):

    if crop_shape is not None:
        raw, labels = get_best_crops(raw, label, crop_shape)
    else:
        raw, labels = [raw], [label]
    for _raw, _label in zip(raw, labels):
        if has_foreground(_label):
            if do_connected_components:
                instances = connected_components(_label)
            else:
                instances = _label
            _raw = normalize(_raw)
            _raw = _raw * 255

            raw_path = os.path.join(raw_dir, f"{slice_prefix_name}_{i+1:05}.tif")
            labels_path = os.path.join(labels_dir, f"{slice_prefix_name}_{i+1:05}.tif")
            imageio.imwrite(raw_path, _raw, compression="zlib")
            imageio.imwrite(labels_path, instances, compression="zlib")


def from_h5_to_tif(
    h5_vol_path, raw_key, raw_dir, labels_key, labels_dir, slice_prefix_name, do_connected_components=True,
    interface=h5py, crop_shape=None, roi_slices=None, resize_longest_side=False
):
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    if h5_vol_path.split(".")[-1] == "zarr":
        kwargs = {"use_zarr_format": True}
    else:
        kwargs = {}

    with interface.File(h5_vol_path, "r", **kwargs) as f:
        raw = f[raw_key][:]
        labels = f[labels_key][:]

        if roi_slices is not None:  # for platynereis cilia
            raw = RoiWrapper(raw, roi_slices)[:]
            labels = RoiWrapper(labels, roi_slices)[:]

        if raw.ndim == 2 and labels.ndim == 2:  # for axondeepseg tem modality
            raw, labels = raw[None], labels[None]

        if raw.ndim == 3 and labels.ndim == 3:  # when we have a volume or mono-channel image
            if resize_longest_side:  # hpa
                raw, labels = resize_image(raw, labels, crop_shape)
                crop_shape = None
            for i, (_raw, _label) in tqdm(enumerate(zip(raw, labels)), total=raw.shape[0], desc=h5_vol_path):
                save_to_tif(i, _raw, _label, crop_shape, raw_dir, labels_dir, do_connected_components,
                            slice_prefix_name)


def for_covid_if(save_path):
    all_image_paths = sorted(glob(os.path.join(ROOT, "covid_if", "*.h5")))

    # val images
    for image_path in tqdm(all_image_paths[10:13]):
        image_id = Path(image_path).stem

        image_save_dir = os.path.join(save_path, "val", "raw")
        label_save_dir = os.path.join(save_path, "val", "labels")

        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)

        with h5py.File(image_path, "r") as f:
            raw = f["raw/serum_IgG/s0"][:]
            labels = f["labels/cells/s0"][:]

            raw, labels = get_best_crops(raw, labels, (512, 512))
            for i, (raw_, labels_) in enumerate(zip(raw, labels)):

                raw_ = normalize(raw_)
                raw_ = raw_ * 255

                imageio.imwrite(os.path.join(image_save_dir, f"{image_id}_{i}.tif"), raw_)
                imageio.imwrite(os.path.join(label_save_dir, f"{image_id}_{i}.tif"), labels_)

    # test images
    for image_path in tqdm(all_image_paths[13:]):
        image_id = Path(image_path).stem

        image_save_dir = os.path.join(save_path, "test", "raw")
        label_save_dir = os.path.join(save_path, "test", "labels")

        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)

        with h5py.File(image_path, "r") as f:
            raw = f["raw/serum_IgG/s0"][:]
            labels = f["labels/cells/s0"][:]

            raw, labels = get_best_crops(raw, labels, (512, 512))
            for i, (raw_, labels_) in enumerate(zip(raw, labels)):

                raw_ = normalize(raw_)
                raw_ = raw_ * 255

                imageio.imwrite(os.path.join(image_save_dir, f"{image_id}_{i}.tif"), raw_)
                imageio.imwrite(os.path.join(label_save_dir, f"{image_id}_{i}.tif"), labels_)


def for_platynereis(save_dir, choice="cilia"):
    """
    for cilia:
        for training   : we take regions of training vol 1-2
        for validation: we take regions of training vol 1-2
        for test: we take train vol 3

    """
    roi_slice = np.s_[85:, :, :]
    if choice == "cilia":
        vol_paths = sorted(glob(os.path.join(ROOT, "platynereis", "cilia", "train_*")))
        for vol_path in vol_paths:
            vol_id = os.path.split(vol_path)[-1].split(".")[0][-2:]

            split = "test" if vol_id == "03" else "val"
            crop_shape = None if vol_id == "01" else (512, 512)
            from_h5_to_tif(
                h5_vol_path=vol_path,
                raw_key="volumes/raw",
                raw_dir=os.path.join(save_dir, choice, "raw"),
                labels_key="volumes/labels/segmentation",
                labels_dir=os.path.join(save_dir, choice, "labels"),
                slice_prefix_name=f"platy_{choice}_{split}_{vol_id}",
                roi_slices=roi_slice if split == "val" else None,
                crop_shape=crop_shape,
            )


def for_mitolab(save_path):
    """
    for mitolab glycolytic muscle

    train_rois = np.s_[0:175, :, :]
    val_rois = np.s_[175:225, :, :]
    test_rois = np.s_[225:, :, :]

    """
    vol_path = os.path.join(ROOT, "mitolab", "10982", "data", "mito_benchmarks", "glycolytic_muscle")
    dataset_id = os.path.split(vol_path)[-1]
    os.makedirs(os.path.join(save_path, dataset_id, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_path, dataset_id, "labels"), exist_ok=True)

    em_path = glob(os.path.join(vol_path, "*_em.tif"))[0]
    mito_path = glob(os.path.join(vol_path, "*_mito.tif"))[0]

    vem = imageio.imread(em_path)
    vmito = imageio.imread(mito_path)

    for i, (slice_em, slice_mito) in tqdm(
        enumerate(zip(vem, vmito)), total=len(vem), desc=f"Processing {dataset_id}"
    ):

        if has_foreground(slice_mito):
            instances = connected_components(slice_mito)

            raw_path = os.path.join(save_path, dataset_id, "raw", f"{dataset_id}_{i+1:05}.tif")
            labels_path = os.path.join(save_path, dataset_id, "labels", f"{dataset_id}_{i+1:05}.tif")

            imageio.imwrite(raw_path, slice_em, compression="zlib")
            imageio.imwrite(labels_path, instances, compression="zlib")

    def move_samples(split, all_raw_files, all_label_files):
        for raw_path, label_path in (zip(all_raw_files, all_label_files)):
            # let's move the raw slice
            slice_id = os.path.split(raw_path)[-1]
            dst = os.path.join(save_path, dataset_id, split, "raw", slice_id)
            shutil.move(raw_path, dst)

            # let's move the label slice
            slice_id = os.path.split(label_path)[-1]
            dst = os.path.join(save_path, dataset_id, split, "labels", slice_id)
            shutil.move(label_path, dst)

    # make a custom splitting logic
    # 1. move to val dir
    os.makedirs(os.path.join(save_path, dataset_id, "val", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_path, dataset_id, "val", "labels"), exist_ok=True)

    move_samples(
        split="val",
        all_raw_files=sorted(glob(os.path.join(save_path, dataset_id, "raw", "*")))[175:225],
        all_label_files=sorted(glob(os.path.join(save_path, dataset_id, "labels", "*")))[175:225]
    )

    # 2. move to test dir
    os.makedirs(os.path.join(save_path, dataset_id, "test", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_path, dataset_id, "test", "labels"), exist_ok=True)

    move_samples(
        split="test",
        all_raw_files=sorted(glob(os.path.join(save_path, dataset_id, "raw", "*")))[175:],
        all_label_files=sorted(glob(os.path.join(save_path, dataset_id, "labels", "*")))[175:]
    )

    # let's remove the left-overs
    shutil.rmtree(os.path.join(save_path, dataset_id, "raw"))
    shutil.rmtree(os.path.join(save_path, dataset_id, "labels"))


def for_orgasegment(save_path):

    val_img_paths = sorted(glob(os.path.join(ROOT, "orgasegment", "val", "*_img.jpg")))
    val_label_paths = sorted(glob(os.path.join(ROOT, "orgasegment", "val", "*_masks_organoid.png")))
    test_img_paths = sorted(glob(os.path.join(ROOT, "orgasegment", "eval", "*_img.jpg")))
    test_label_paths = sorted(glob(os.path.join(ROOT, "orgasegment", "eval", "*_masks_organoid.png")))

    os.makedirs(os.path.join(save_path, "test", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "val", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "test", "labels"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "val", "labels"), exist_ok=True)

    volumes = [val_img_paths, val_label_paths, test_img_paths, test_label_paths]
    for vol in range(2):
        for i, (image_path, label_path) in enumerate(zip(volumes[vol*2], volumes[vol*2+1])):
            _split = "test" if "eval" in str(image_path) else "val"
            image = imageio.imread(image_path)
            label = imageio.imread(label_path)

            raw, labels = get_best_crops(image, label, (512, 512))

            for j, (_raw, _labels) in enumerate(zip(raw, labels)):

                _raw = normalize(_raw)
                _raw = _raw * 255

                if has_foreground(_labels):
                    _labels = connected_components(_labels)

                    imageio.imwrite(
                        os.path.join(save_path, _split, "raw", f"orgasegment_{_split}_{i+1:05}_{j}.tif"), _raw
                    )
                    imageio.imwrite(
                        os.path.join(save_path, _split, "labels", f"orgasegment_{_split}_{i+1:05}_{j}.tif"), _labels
                    )


def for_gonuclear(save_path):

    go_nuclear_val_vol = os.path.join(ROOT, "gonuclear", "gonuclear_datasets", "1139.h5")
    go_nuclear_test_vol = os.path.join(ROOT, "gonuclear", "gonuclear_datasets", "1170.h5")
    from_h5_to_tif(
        h5_vol_path=go_nuclear_val_vol,
        raw_key="raw/nuclei",
        raw_dir=os.path.join(save_path, "val", "raw"),
        labels_key="labels/nuclei",
        labels_dir=os.path.join(save_path, "val", "labels"),
        slice_prefix_name="gonuclear_val_1139",
        crop_shape=(1024, 1024)
    )
    from_h5_to_tif(
        h5_vol_path=go_nuclear_test_vol,
        raw_key="raw/nuclei",
        raw_dir=os.path.join(save_path, "test", "raw"),
        labels_key="labels/nuclei",
        labels_dir=os.path.join(save_path, "test", "labels"),
        slice_prefix_name="gonuclear_test_1170",
    )


def for_hpa(save_dir):
    """
    take the last 58 volumes from the train split for validation and use the validation split for testing
    """
    hpa_val_vols = sorted(glob(os.path.join(ROOT, "hpa", "train", "*.h5")))[210:]
    hpa_test_vols = sorted(glob(os.path.join(ROOT, "hpa", "val", "*.h5")))

    def save_slices_per_split(all_vol_paths, split):
        for vol_path in all_vol_paths:
            vol_id = Path(vol_path).stem

            from_h5_to_tif(
                h5_vol_path=vol_path,
                raw_key="raw/protein",
                raw_dir=os.path.join(save_dir, split, "raw"),
                labels_key="labels",
                labels_dir=os.path.join(save_dir, split, "labels"),
                slice_prefix_name=f"hpa_{split}_{vol_id}",
                crop_shape=(512, 512),
                resize_longest_side=True
            )

    save_slices_per_split(hpa_val_vols, "val")
    save_slices_per_split(hpa_test_vols, "test")


def download_all_datasets(path):
    datasets.get_platynereis_cilia_dataset(os.path.join(path, "platynereis"), patch_shape=(1, 512, 512), download=True)
    datasets.get_covid_if_dataset(os.path.join(path, "covid_if"), patch_shape=(1, 512, 512), download=True)
    datasets.get_orgasegment_dataset(os.path.join(path, "orgasegment"), split="val",
                                     patch_shape=(512, 512), download=True)
    datasets.get_orgasegment_dataset(os.path.join(path, "orgasegment"), split="eval",
                                     patch_shape=(512, 512), download=True)
    datasets.get_gonuclear_dataset(os.path.join(path, "gonuclear"), patch_shape=(1, 512, 512),
                                   segmentation_task="nuclei", download=True)
    get_hpa_segmentation_dataset(os.path.join(path, "hpa"), split="val", patch_shape=(512, 512),
                                 channels=["protein"], download=True)
    get_hpa_segmentation_dataset(os.path.join(path, "hpa"), split="test", patch_shape=(512, 512),
                                 channels=["protein"], download=True)
