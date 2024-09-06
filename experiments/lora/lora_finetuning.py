import os
import argparse

import torch

from get_data_loaders import _fetch_loaders
import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = params / 1e6
    return f"The number of trainable parameters for the provided model is {round(params, 2)}M"


def finetune(args):
    """Code for finetuning SAM (using LoRA)
    """
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    n_objects_per_batch = args.n_objects  # this is the number of objects per batch that will be sampled
    freeze_parts = args.freeze  # override this to freeze different parts of the model
    lora_rank = args.lora_rank  # the rank for low rank adaptation
    checkpoint_ending = f"lora_{lora_rank}" if lora_rank is not None else "full_ft"
    dataset = args.dataset

    checkpoint_name = f"{args.model_type}/{dataset}_{checkpoint_ending}"

    # all the stuff we need for training
    train_loader, val_loader = _fetch_loaders(dataset, args.input_path)
    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10, "verbose": True}
    optimizer_class = torch.optim.AdamW

    # Run training.
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping=None,
        n_objects_per_batch=n_objects_per_batch,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts,
        device=device,
        lr=1e-5,
        n_iterations=50000,
        save_root=args.save_root,
        scheduler_kwargs=scheduler_kwargs,
        optimizer_class=optimizer_class,
        lora_rank=lora_rank,
    )

    if args.export_path is not None:
        checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root, "checkpoints", checkpoint_name, "best.pt"
        )
        export_custom_sam_model(
            checkpoint_path=checkpoint_path, model_type=model_type, save_path=args.export_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the LIVECell dataset.")
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_h, vit_b or vit_l."
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
        "--freeze", type=str, nargs="+", default=None,
        help="Which parts of the model to freeze for finetuning."
    )
    parser.add_argument(
        "--n_objects", type=int, default=25, help="The number of instances (objects) per batch used for finetuning."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=None, help="The rank for low rank adaptation."
    )
    parser.add_argument(
        "--dataset", "-d", type=str, required=True,
        help="The dataset to use for training. Chose from 'covid_if', 'orgasegment, 'mouse-embryo', 'mitolab_glycolytic_muscle', 'platy_cylia', 'gonuclear."
    )
    parser.add_argument(
        "--input_path", "-i", type=str, default="/scratch/usr/nimcarot/data",
        help="Specifies the path to the data directory (should be set to /usr/name/data if dataset is at /usr/name/data/dataset_name)"
    )
    args = parser.parse_args()
    finetune(args)


if __name__ == "__main__":
    main()
