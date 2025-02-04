import os
import argparse

import torch

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_orgasegment_loader
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model, export_custom_qlora_model

from peft_sam.util import get_default_peft_kwargs, RawTrafo


DATA_ROOT = "./data/orgasegment"


def get_data_loaders(input_path):
    additional_kwargs = {
        "raw_transform": RawTrafo(desired_shape=(512, 512), triplicate_dims=True, do_padding=False),
        "label_transform": PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True,
        ),
        "sampler": MinInstanceSampler(),
        "shuffle": True,
    }

    train_loader = get_orgasegment_loader(
        path=input_path, patch_shape=(512, 512), split="train", batch_size=1, download=True, **additional_kwargs
    )
    val_loader = get_orgasegment_loader(
        path=input_path, patch_shape=(512, 512), split="val", batch_size=1, download=True, **additional_kwargs
    )
    return train_loader, val_loader


def finetune_sam(args):
    """Script for finetuning SAM (using PEFT methods) on microscopy images.
    """
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    n_objects_per_batch = 5  # this is the number of objects per batch that will be sampled

    # whether to freeze the entire image encoder.
    if args.peft_method == "freeze_encoder":
        freeze_parts = "image_encoder"
    else:
        freeze_parts = None

    # specify checkpoint path depending on the type of finetuning
    if args.peft_method is None:
        checkpoint_name = f"{args.model_type}/full_finetuning/orgasegment_sam"
    else:
        checkpoint_name = f"{args.model_type}/{args.peft_method}/orgasegment_sam"

    # all the stuff we need for training
    train_loader, val_loader = get_data_loaders(DATA_ROOT)
    peft_kwargs = get_default_peft_kwargs(args.peft_method)
    print("PEFT arguments: ", peft_kwargs)

    # Run training.
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping=10,
        lr=1e-5,
        n_epochs=100,
        n_objects_per_batch=n_objects_per_batch,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts,  # override this to freeze different parts of the model
        device=device,
        save_root=args.save_root,
        peft_kwargs=peft_kwargs,
        with_segmentation_decoder=True,
    )

    if args.export_path is not None:
        checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root, "checkpoints", checkpoint_name, "best.pt"
        )
        export_custom_sam_model(
            checkpoint_path=checkpoint_path, model_type=model_type, save_path=args.export_path,
        )

    # Exports the finetuned QLoRA model weights in desired format.
    if args.peft_method == "qlora":
        checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root, "checkpoints", checkpoint_name, "best.pt"
        )
        export_custom_qlora_model(
            checkpoint_path=checkpoint_path, model_type=model_type, save_path=args.checkpoint_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for microscopy data.")
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_b, vit_l or vit_h."
    )
    parser.add_argument(
        "--save_root", "-s", default=None,
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run."
    )
    parser.add_argument(
        "--export_path", "-e",
        help="Where to export the finetuned model to. The exported model can be used in the annotation tools."
    )
    parser.add_argument(
        "--peft_method", type=str, default=None, help="The method to use for PEFT."
    )
    args = parser.parse_args()
    finetune_sam(args)


if __name__ == "__main__":
    main()
