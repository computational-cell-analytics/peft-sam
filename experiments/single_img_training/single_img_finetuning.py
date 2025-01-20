import os
import argparse
import numpy as np

import torch

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

from peft_sam.util import get_peft_kwargs
from peft_sam.dataset.get_data_loaders import _fetch_loaders

# Define the sample range and rois for the selected images
SAMPLE_DATA = {
    'covid_if': {'train_sample_range': (0, 1), 'val_sample_range': (10, 11), 'train_rois': None, 'val_rois': None},
    'livecell': {'train_sample_range': (2, 3), 'val_sample_range': (25, 26), 'train_rois': None, 'val_rois': None},
    'orgasegment': {'train_sample_range': (0, 1), 'val_sample_range': (0, 1), 'train_rois': None, 'val_rois': None},
    'mitolab_glycolytic_muscle': {'train_sample_range': None, 'val_sample_range': None,
                                  'train_rois': np.s_[20:21, :, :], 'val_rois': np.s_[180:181, :, :]},
    'platy_cilia': {'train_sample_range': None, 'val_sample_range': None, 'train_rois': {2: np.s_[49:50, :, :]},
                    'val_rois': {2: np.s_[65:66, :, :]}},
    'gonuclear': {'train_sample_range': None, 'val_sample_range': None, 'train_rois': {1136: np.s_[60:61, :, :]},
                  'val_rois': {1139: np.s_[40:41, :, :]}},
    'hpa': {'train_sample_range': (1, 2), 'val_sample_range': (1, 2), 'train_rois': None, 'val_rois': None},
}


def finetune(args):
    """Code for finetuning SAM (using PEFT methods) on different data
    """
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    n_objects_per_batch = 5
    freeze_parts = args.freeze  # override this to freeze different parts of the model
    dataset = args.dataset

    # specify checkpoint path depending on the type of finetuning
    if args.checkpoint_name is not None:
        checkpoint_name = args.checkpoint_name
    elif args.peft_method is not None:
        checkpoint_name = f"{args.model_type}/{args.peft_method}/{dataset}_sam"
    elif freeze_parts is not None:
        checkpoint_name = f"{args.model_type}/frozen_encoder/{dataset}_sam"
    else:
        checkpoint_name = f"{args.model_type}/full_ft/{dataset}_sam"

    # all the stuff we need for training
    train_sample_range = SAMPLE_DATA[dataset]['train_sample_range']
    val_sample_range = SAMPLE_DATA[dataset]['val_sample_range']
    train_rois = SAMPLE_DATA[dataset]['train_rois']
    val_rois = SAMPLE_DATA[dataset]['val_rois']

    train_loader, val_loader = _fetch_loaders(
        dataset, args.input_path, train_sample_range=train_sample_range, val_sample_range=val_sample_range,
        train_rois=train_rois, val_rois=val_rois
    )

    n_samples_train = 50 if len(train_loader) < 50 else None
    n_samples_val = 50 if len(val_loader) < 50 else None

    train_loader, val_loader = _fetch_loaders(
        dataset, args.input_path, train_sample_range=train_sample_range, val_sample_range=val_sample_range,
        train_rois=train_rois, val_rois=val_rois, n_train_samples=n_samples_train, n_val_samples=n_samples_val,
        batch_size=1
    )
    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10, "verbose": True}
    optimizer_class = torch.optim.AdamW

    peft_kwargs = get_peft_kwargs(
        args.peft_rank,
        args.peft_method,
        alpha=args.alpha,
        dropout=args.dropout,
        projection_size=args.projection_size,
        quantize=args.quantize,
    )
    print("PEFT arguments: ", peft_kwargs)

    # Run training.
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping=10,
        n_objects_per_batch=n_objects_per_batch,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts,
        device=device,
        lr=args.learning_rate,
        n_iterations=None,
        n_epochs=100,
        save_root=args.save_root,
        scheduler_kwargs=scheduler_kwargs,
        optimizer_class=optimizer_class,
        peft_kwargs=peft_kwargs,
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
        "--dataset", "-d", type=str, required=True,
        help="The dataset to use for training."
    )
    parser.add_argument(
        "--input_path", "-i", type=str, default="/scratch/usr/nimcarot/data",
        help="Specifies the path to the data directory (set to ./data if dataset is at ./data/<dataset_name>)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="The learning rate for finetuning."
    )
    parser.add_argument(
        "--peft_rank", type=int, default=None, help="The rank for peft training."
    )
    parser.add_argument(
        "--peft_method", type=str, default=None, help="The method to use for PEFT."
    )
    parser.add_argument(
        "--dropout", type=float, default=None, help="The dropout rate to use for FacT and AdaptFormer."
    )
    parser.add_argument(
        "--alpha", default=None, help="Scaling Factor for PEFT methods"
    )
    parser.add_argument(
        "--projection_size", type=int, default=None, help="Projection size for Adaptformer"
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Quantize the model."
    )
    parser.add_argument(
        "--checkpoint_name", type=str, default=None, help="Custom checkpoint name"
    )
    args = parser.parse_args()
    finetune(args)


if __name__ == "__main__":
    main()
