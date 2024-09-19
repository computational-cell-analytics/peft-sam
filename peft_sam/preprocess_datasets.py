import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path

import h5py
import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

from elf.wrapper import RoiWrapper

from torch_em.data import datasets
from torch_em.transform.raw import normalize, normalize_percentile


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


def make_center_crop(image, desired_shape):
    if image.shape < desired_shape:
        return image

    center_coords = (int(image.shape[0] / 2), int(image.shape[1] / 2))
    tolerances = (int(desired_shape[0] / 2), int(desired_shape[1] / 2))

    # let's take the center crop from the image
    cropped_image = image[
        center_coords[0] - tolerances[0]: center_coords[0] + tolerances[0],
        center_coords[1] - tolerances[1]: center_coords[1] + tolerances[1]
    ]
    return cropped_image


def save_to_tif(i, _raw, _label, crop_shape, raw_dir, labels_dir, do_connected_components, slice_prefix_name):
    if crop_shape is not None:
        _raw = make_center_crop(_raw, crop_shape)
        _label = make_center_crop(_label, crop_shape)

    # we only save labels with foreground
    if has_foreground(_label):
        if do_connected_components:
            instances = connected_components(_label)
        else:
            instances = _label

        raw_path = os.path.join(raw_dir, f"{slice_prefix_name}_{i+1:05}.tif")
        labels_path = os.path.join(labels_dir, f"{slice_prefix_name}_{i+1:05}.tif")
        imageio.imwrite(raw_path, _raw, compression="zlib")
        imageio.imwrite(labels_path, instances, compression="zlib")


def from_h5_to_tif(
    h5_vol_path, raw_key, raw_dir, labels_key, labels_dir, slice_prefix_name, do_connected_components=True,
    interface=h5py, crop_shape=None, roi_slices=None, to_one_channel=False
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

        if raw.ndim == 3 and raw.shape[0] == 3 and to_one_channel:  # for tissuenet
            print("Got an RGB image, converting it to one-channel.")
            raw = convert_rgb(raw)

        if roi_slices is not None:  # for cremi
            raw = RoiWrapper(raw, roi_slices)[:]
            labels = RoiWrapper(labels, roi_slices)[:]

        if raw.ndim == 2 and labels.ndim == 2:  # for axondeepseg tem modality
            raw, labels = raw[None], labels[None]

        if raw.ndim == 3 and labels.ndim == 3:  # when we have a volume or mono-channel image
            for i, (_raw, _label) in tqdm(enumerate(zip(raw, labels)), total=raw.shape[0], desc=h5_vol_path):
                save_to_tif(
                    i, _raw, _label, crop_shape, raw_dir, labels_dir, do_connected_components, slice_prefix_name
                )

        elif raw.ndim == 3 and labels.ndim == 2:  # when we have a multi-channel input (rgb)
            if raw.shape[0] == 4:  # hpa has 4 channel inputs (0: microtubules, 1: protein, 2: nuclei, 3: er)
                raw = raw[1:]

            # making channels last (to make use of 3-channel inputs)
            raw = raw.transpose(1, 2, 0)

            save_to_tif(0, raw, labels, crop_shape, raw_dir, labels_dir, do_connected_components, slice_prefix_name)


def for_covid_if(save_path):
    all_image_paths = sorted(glob(os.path.join(ROOT, "covid_if", "*.h5")))

    # val images
    for image_path in tqdm(all_image_paths[10:13]):
        image_id = Path(image_path).stem

        image_save_dir = os.path.join(save_path, "val", "raw")
        label_save_dir = os.path.join(save_path, "slices", "val", "labels")

        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)

        with h5py.File(image_path, "r") as f:
            raw = f["raw/serum_IgG/s0"][:]
            labels = f["labels/cells/s0"][:]

            imageio.imwrite(os.path.join(image_save_dir, f"{image_id}.tif"), raw)
            imageio.imwrite(os.path.join(label_save_dir, f"{image_id}.tif"), labels)

    # test images
    for image_path in tqdm(all_image_paths[13:]):
        image_id = Path(image_path).stem

        image_save_dir = os.path.join(Path(image_path).parent, "slices", "test", "raw")
        label_save_dir = os.path.join(Path(image_path).parent, "slices", "test", "labels")

        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)

        with h5py.File(image_path, "r") as f:
            raw = f["raw/serum_IgG/s0"][:]
            labels = f["labels/cells/s0"][:]

            imageio.imwrite(os.path.join(image_save_dir, f"{image_id}.tif"), raw)
            imageio.imwrite(os.path.join(label_save_dir, f"{image_id}.tif"), labels)


def for_platynereis(save_dir, choice="cilia"):
    """
    for cilia:
        for training   : we take regions of training vol 1-3
        for validation: we take regions of training vol 1-3
        for test: we take validation vol 1

    """
    roi_slice = np.s_[85:, :, :]
    if choice == "cilia":
        vol_paths = sorted(glob(os.path.join(ROOT, "platynereis", "cilia", "train_*")))
        for vol_path in vol_paths:
            vol_id = os.path.split(vol_path)[-1].split(".")[0][-2:]

            split = "test" if vol_id == "03" else "val"
            from_h5_to_tif(
                h5_vol_path=vol_path,
                raw_key="volumes/raw",
                raw_dir=os.path.join(save_dir, choice, "raw"),
                labels_key="volumes/labels/segmentation",
                labels_dir=os.path.join(save_dir, choice, "labels"),
                slice_prefix_name=f"platy_{choice}_{split}_{vol_id}",
                roi_slices=roi_slice if split == "val" else None,
            )


def for_mitolab(save_path):
    """
    for mitolab glycolytic muscle

    train_rois = np.s_[0:175, :, :]
    val_rois = np.s_[175:225, :, :]
    test_rois = np.s_[225:, :, :]

    """
    all_dataset_ids = []
    _roi_vol_paths = sorted(glob(os.path.join(ROOT, "mitolab", "10982", "data", "mito_benchmarks", "*")))

    for vol_path in _roi_vol_paths:
        dataset_id = os.path.split(vol_path)[-1]
        all_dataset_ids.append(dataset_id)

        os.makedirs(os.path.join(save_path, dataset_id, "raw"), exist_ok=True)
        os.makedirs(os.path.join(save_path, dataset_id, "labels"), exist_ok=True)

        em_path = glob(os.path.join(vol_path, "*_em.tif"))[0]
        mito_path = glob(os.path.join(vol_path, "*_mito.tif"))[0]

        vem = imageio.imread(em_path)
        vmito = imageio.imread(mito_path)

        for i, (slice_em, slice_mito) in tqdm(
            enumerate(zip(vem, vmito)), total=len(vem), desc=f"Processing {dataset_id}"
        ):

            if Path(em_path).stem.startswith("salivary_gland"):
                slice_em = make_center_crop(slice_em, (1024, 1024))
                slice_mito = make_center_crop(slice_mito, (1024, 1024))

            if has_foreground(slice_mito):
                instances = connected_components(slice_mito)

                raw_path = os.path.join(save_path, dataset_id, "raw", f"{dataset_id}_{i+1:05}.tif")
                labels_path = os.path.join(save_path, dataset_id, "labels", f"{dataset_id}_{i+1:05}.tif")

                imageio.imwrite(raw_path, slice_em, compression="zlib")
                imageio.imwrite(labels_path, instances, compression="zlib")

    # now, let's work on the tem dataset
    image_paths = sorted(glob(os.path.join(ROOT, "mitolab", "10982", "data", "tem_benchmark", "images", "*")))
    mask_paths = sorted(glob(os.path.join(ROOT, "mitolab", "10982", "data", "tem_benchmark", "masks", "*")))

    os.makedirs(os.path.join(save_path, "tem", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "tem", "labels"), exist_ok=True)

    # let's move the tem data to slices
    for image_path, mask_path in tqdm(zip(image_paths, mask_paths), desc="Preprocessimg tem", total=len(image_paths)):
        sample_id = os.path.split(image_path)[-1]
        image_dst = os.path.join(save_path, "tem", "raw", sample_id)
        mask_dst = os.path.join(save_path, "tem", "labels", sample_id)

        tem_image = make_center_crop(imageio.imread(image_path), (768, 768))
        tem_mask = make_center_crop(imageio.imread(mask_path), (768, 768))

        if has_foreground(tem_mask):
            imageio.imwrite(image_dst, tem_image)
            imageio.imwrite(mask_dst, tem_mask)

    all_dataset_ids.append("tem")

    for dataset_id in all_dataset_ids:
        make_custom_splits(save_dir=os.path.join(save_path, dataset_id))


def make_custom_splits(save_dir):
    def move_samples(split, all_raw_files, all_label_files):
        for raw_path, label_path in (zip(all_raw_files, all_label_files)):
            # let's move the raw slice
            slice_id = os.path.split(raw_path)[-1]
            dst = os.path.join(save_dir, split, "raw", slice_id)
            shutil.move(raw_path, dst)

            # let's move the label slice
            slice_id = os.path.split(label_path)[-1]
            dst = os.path.join(save_dir, split, "labels", slice_id)
            shutil.move(label_path, dst)

    # make a custom splitting logic
    # 1. move to val dir
    os.makedirs(os.path.join(save_dir, "val", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "val", "labels"), exist_ok=True)

    move_samples(
        split="val",
        all_raw_files=sorted(glob(os.path.join(save_dir, "raw", "*")))[175:225],
        all_label_files=sorted(glob(os.path.join(save_dir, "labels", "*")))[175:225]
    )

    # 2. move to test dir
    os.makedirs(os.path.join(save_dir, "test", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "test", "labels"), exist_ok=True)

    move_samples(
        split="test",
        all_raw_files=sorted(glob(os.path.join(save_dir, "raw", "*")))[225:],
        all_label_files=sorted(glob(os.path.join(save_dir, "labels", "*")))[225:]
    )

    # let's remove the left-overs
    shutil.rmtree(os.path.join(save_dir, "raw"))
    shutil.rmtree(os.path.join(save_dir, "labels"))


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
            img_shape = image.shape
            label = imageio.imread(label_path)

            if len(img_shape) == 2:
                image = np.stack([image, image, image], axis=2)

            imageio.imwrite(os.path.join(save_path, _split, "raw", f"orgasegment_{_split}_{i+1:05}.tif"), image)
            imageio.imwrite(os.path.join(save_path, _split, "labels", f"orgasegment_{_split}_{i+1:05}.tif"), label)


def for_gonuclear(save_path):

    go_nuclear_val_vol = os.path.join(ROOT, "gonuclear", "gonuclear_datasets", "1139.h5")
    go_nuclear_test_vol = os.path.join(ROOT, "gonuclear", "gonuclear_datasets", "1170.h5")
    from_h5_to_tif(
        h5_vol_path=go_nuclear_val_vol,
        raw_key="raw/nuclei",
        raw_dir=os.path.join(save_path, "val", "raw"),
        labels_key="labels/nuclei",
        labels_dir=os.path.join(save_path, "val", "labels"),
        slice_prefix_name="gonuclear_val_1129",
        roi_slices=None
    )
    from_h5_to_tif(
        h5_vol_path=go_nuclear_test_vol,
        raw_key="raw/nuclei",
        raw_dir=os.path.join(save_path, "test", "raw"),
        labels_key="labels/nuclei",
        labels_dir=os.path.join(save_path, "test", "labels"),
        slice_prefix_name="gonuclear_test_1170",
        roi_slices=None
    )


def download_all_datasets(path):
    datasets.get_platynereis_cilia_dataset(os.path.join(path, "platynereis"), patch_shape=(1, 512, 512), download=True)
    datasets.get_covid_if_dataset(os.path.join(path, "covid_if"), patch_shape=(1, 512, 512), download=True)
    datasets.get_orgasegment_dataset(os.path.join(path, "orgasegment"), split="val",
                                     patch_shape=(512, 512), download=True)
    datasets.get_orgasegment_dataset(os.path.join(path, "orgasegment"), split="eval",
                                     patch_shape=(512, 512), download=True)
    datasets.get_gonuclear_dataset(os.path.join(path, "gonuclear"), patch_shape=(1, 512, 512),
                                   segmentation_task="nuclei", download=True)


def main():

    download_all_datasets(ROOT)

    preprocess_data("covid_if")
    preprocess_data("platynereis")
    preprocess_data("mitolab")
    preprocess_data("orgasegment")
    preprocess_data("gonuclear")


if __name__ == "__main__":
    main()
