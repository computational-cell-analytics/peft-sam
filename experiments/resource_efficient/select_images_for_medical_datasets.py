import os
from glob import glob

import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

from torch_em.util.image import load_data
from torch_em.data.datasets import medical
from torch_em.transform.raw import normalize
from torch_em.transform.generic import ResizeLongestSideInputs


# The images below are mostly cherry picked by me.
MATCH_IMAGE_IDS = {
    "papila": ["RET004OS.jpg", "RET008OD.jpg"],
    "motum": ["sub-0001_flair.nii.gz", "sub-0004_flair.nii.gz"],
    "psfhs": ["03744.mha", "03745.mha"],
    "jsrt": ["case001.bmp", "case004.bmp"],
    "amd_sd": ["2_3.tif", "3_6.tif"],
    "mice_tumseg": ["CT_M01_0h.nii.gz", "CT_M02_22.nii.gz"],
    # NEW DATASETS
    "sega": ["R3.nii.gz", "R4.nii.gz"],
    "ircadb": ["3Dircadb1.1.h5", "3Dircadb1.2.h5"],
    "dsad": ["01/image06.png", "01/image47.png"]
}


def _get_medical_images(dataset_name, view):
    # Get the data directory
    data_dir = os.path.join("/media/anwai/ANWAI/data")
    if not os.path.exists(data_dir):
        data_dir = "./data"

    _get_paths = {
        # 2d fundus images for optic disc segmentation
        "papila": lambda: medical.papila.get_papila_paths(
            path=os.path.join(data_dir, "papila"), split="train", task="cup", download=True,
        ),
        "motum": lambda: medical.motum.get_motum_paths(
            path=os.path.join(data_dir, "motum"), split="train", modality="flair", download=True,
        ),
        "psfhs": lambda: medical.psfhs.get_psfhs_paths(
            path=os.path.join(data_dir, "psfhs"), split="train", download=True,
        ),
        "jsrt": lambda: medical.jsrt.get_jsrt_paths(
            path=os.path.join(data_dir, "jsrt"), split="train", choice="Segmentation02", download=True,
        ),
        "amd_sd": lambda: medical.amd_sd.get_amd_sd_paths(
            path=os.path.join(data_dir, "amd_sd"), split="train", download=True,
        ),
        "mice_tumseg": lambda: medical.mice_tumseg.get_mice_tumseg_paths(
            path=os.path.join(data_dir, "mice_tumseg"), split="train", download=True,
        ),
        # NEW DATASETS
        "sega": lambda: medical.sega.get_sega_paths(
            path=os.path.join(data_dir, "sega"), data_choice="Rider", download=True,
        ),
        "ircadb": lambda: medical.ircadb.get_ircadb_paths(
            path=os.path.join(data_dir, "ircadb"), split="train", download=True,
        ),
        "dsad": lambda: medical.dsad.get_dsad_paths(
            path=os.path.join(data_dir, "dsad"), organ="liver", download=True,
        )
    }

    paths = _get_paths[dataset_name]()

    if isinstance(paths, tuple):
        raw_paths, label_paths = paths
        raw_key, label_key = None, None
    else:  # only the case for IRCADb, hence I hard-code the key values.
        raw_paths = label_paths = paths
        raw_key, label_key = "raw", "labels/liver"

    # Load the images one by one.
    for rpath, lpath in zip(raw_paths, label_paths):
        image = load_data(rpath, "data" if rpath.endswith(".nii.gz") else raw_key)[:]
        labels = load_data(lpath, "data" if lpath.endswith(".nii.gz") else label_key)[:]

        # Resize the images and labels on the fly.
        _change_axes = False
        if image.ndim == 3 and (image.shape[0] == 3 or image.shape[-1] == 3):
            is_rgb = True
            if image.shape[-1] == 3:
                image = image.transpose(2, 0, 1)
                _change_axes = True
        else:
            is_rgb = False

        raw_trafo = ResizeLongestSideInputs(target_shape=(512, 512), is_rgb=is_rgb)
        image = raw_trafo(image)

        # Switch back the axes.
        if _change_axes:
            image = image.transpose(1, 2, 0)

        label_trafo = ResizeLongestSideInputs(target_shape=(512, 512), is_label=True)
        labels = label_trafo(labels)

        if view:
            import napari
            v = napari.Viewer()
            v.add_image(image, name="Image")
            v.add_labels(labels, name="Labels")
            napari.run()

        # If the image ids match the chosen one, we save them.
        if dataset_name in MATCH_IMAGE_IDS:

            target_dir = f"./chosen_data/{dataset_name}"
            os.makedirs(target_dir, exist_ok=True)

            for i, v in enumerate(MATCH_IMAGE_IDS[dataset_name], start=1):
                if v in rpath:
                    imageio.imwrite(
                        os.path.join(target_dir, f"{dataset_name}_{i}_image.tif"), image, compression="zlib",
                    )
                    imageio.imwrite(
                        os.path.join(target_dir, f"{dataset_name}_{i}_labels.tif"), labels, compression="zlib",
                    )

            if len(glob(os.path.join(target_dir, "*.tif"))) == 4:
                break


def _preprocess_data_final_version(dataset_name, view):
    target_dir = f"./chosen_data/{dataset_name}"
    image_paths = sorted(glob(os.path.join(target_dir, "*_image.tif")))
    gt_paths = sorted(glob(os.path.join(target_dir, "*_labels.tif")))

    if not image_paths:
        print("Well, no images, then no processing.")
        return

    for i, (ipath, gpath) in enumerate(zip(image_paths, gt_paths)):
        image = imageio.imread(ipath)
        gt = imageio.imread(gpath)

        # Remove misc labels from JSRT
        if dataset_name == "jsrt" and len(np.unique(gt)) > 4:
            gt[gt == 1] = 0

        # Run connected components on labels
        gt = connected_components(gt).astype("uint8")

        # For some datasets, need to extract either slices or postprocess them in a certain way.
        # NOTE: These slices are chosen manually by me.
        if image.ndim == 3:
            z_slice = None
            if dataset_name == "motum":
                z_slice = 13 if i == 0 else 9
            elif dataset_name == "mice_tumseg":
                z_slice = 149 if i == 0 else 168
            elif dataset_name == "sega":
                z_slice = 821 if i == 0 else 837
            elif dataset_name == "ircadb":
                z_slice = 76 if i == 0 else 132

            if z_slice:
                image, gt = image[z_slice], gt[z_slice]

                if dataset_name == "sega":  # Run cc on slice as aorta in 3d is a joint structure.
                    gt = connected_components(gt).astype("uint8")

                imageio.imwrite(ipath, image, compression="zlib")
                imageio.imwrite(gpath, gt, compression="zlib")

        # Other datasets with tiny changes.
        if dataset_name == "psfhs" and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
            imageio.imwrite(ipath, image, compression="zlib")

        if dataset_name in ["jsrt", "amd_sd"]:
            imageio.imwrite(gpath, gt, compression="zlib")

        if view:
            import napari
            v = napari.Viewer()
            v.add_image(image)
            v.add_labels(gt)
            napari.run()


def _normalize_inputs(dataset_name, view):
    target_dir = f"./chosen_data/{dataset_name}"
    image_paths = sorted(glob(os.path.join(target_dir, "*_image.tif")))

    if not image_paths:
        print("No images found to normalize. This is not 'normal'.")
        return

    for ipath in image_paths:
        image = imageio.imread(ipath)

        if dataset_name in ["sega", "ircadb", "mice_tumseg", "motum"]:
            if dataset_name in ["sega", "motum"]:
                image = np.clip(image, 0, None)

            image = normalize(image) * 255
            imageio.imwrite(ipath.replace("_image.tif", "_image_normalized.tif"), image, compression="zlib")
            os.remove(ipath)

        if view:
            import napari
            v = napari.Viewer()
            v.add_image(image)
            napari.run()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    parser.add_argument("--choose_images", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()

    # Step 1: Choose images.
    if args.choose_images:
        _get_medical_images(args.dataset_name, args.view)

    # Step 2: Preprocess inputs to match training SAM.
    if args.preprocess:
        _preprocess_data_final_version(args.dataset_name, args.view)

    # Step 3: Normalize inputs.
    if args.normalize:
        _normalize_inputs(args.dataset_name, args.view)
