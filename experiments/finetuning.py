import os
import argparse
from pathlib import Path

import torch

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model, export_custom_qlora_model

from peft_sam.util import get_peft_kwargs
from peft_sam.dataset.get_data_loaders import _fetch_microscopy_loaders, _fetch_medical_loaders


def finetune_sam(args):
    """Code for finetuning SAM (using PEFT methods) on multiple biomedical imaging datasets.
    """
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path  # specify custom checkpoint to start training from
    n_objects_per_batch = args.n_objects  # this is the number of objects per batch that will be sampled
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
    get_loaders = _fetch_medical_loaders if args.medical_imaging else _fetch_microscopy_loaders
    train_loader, val_loader = get_loaders(dataset, args.input_path)

    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10, "verbose": True}
    optimizer_class = torch.optim.AdamW
    peft_kwargs = get_peft_kwargs(
        peft_rank=args.peft_rank,
        peft_module=args.peft_method,
        alpha=args.alpha,
        dropout=args.dropout,
        projection_size=args.projection_size,
        quantize=args.quantize,
        attention_layers_to_update=args.attention_layers_to_update,
        update_matrices=args.update_matrices,
    )
    print("PEFT arguments: ", peft_kwargs)

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
        lr=args.learning_rate,
        n_iterations=50000,
        save_root=args.save_root,
        scheduler_kwargs=scheduler_kwargs,
        optimizer_class=optimizer_class,
        peft_kwargs=peft_kwargs,
        with_segmentation_decoder=(not args.medical_imaging),
    )

    if args.export_path is not None:
        checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root, "checkpoints", checkpoint_name, "best.pt"
        )
        export_custom_sam_model(
            checkpoint_path=checkpoint_path, model_type=model_type, save_path=args.export_path,
        )

    if args.quantize:
        checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root, "checkpoints", checkpoint_name, "best.pt"
        )
        save_path = os.path.join(
            "" if args.save_root is None else args.save_root, "checkpoints", checkpoint_name, "for_inference", "best.pt"
        )
        os.makedirs(Path(save_path).parent, exist_ok=True)

        export_custom_qlora_model(
            checkpoint_path=None,  # i.e. use the weights of "model_type" chosen model.
            finetuned_path=checkpoint_path,  # filepath to the custom finetuned model.
            model_type=model_type,
            save_path=save_path,  # filepath where the desired qlora model will be exported.
        )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the biomedical imaging datasets.")
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_h, vit_b or vit_l."
    )
    parser.add_argument(
        "--checkpoint_path", "-c", default=None,
        help="The checkpoint path to start training from."
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
        "--freeze", type=str, nargs="+", default=None, help="Which parts of the model to freeze for finetuning."
    )
    parser.add_argument(
        "--n_objects", type=int, default=25, help="The number of instances (objects) per batch used for finetuning."
    )
    parser.add_argument(
        "--dataset", "-d", type=str, required=True, help="The dataset to use for training."
    )
    parser.add_argument(
        "--input_path", "-i", type=str, default="/mnt/vast-nhr/projects/cidas/cca/experiments/peft_sam/data",
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
        "--checkpoint_name", type=str, default=None, help="Custom checkpoint name"
    )
    parser.add_argument(
        "--medical_imaging", action="store_true", help="Whether to finetune SAM on medical imaging datasets."
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Whether to quantize the model."
    )
    parser.add_argument(
        '--attention_layers_to_update', default=[], nargs='+', type=int,
        help='A list of attention blocks to update during PEFT',
    )
    parser.add_argument('--update_matrices', nargs='+', help='A list of matrices to update during LoRA')
    args = parser.parse_args()
    finetune_sam(args)


if __name__ == "__main__":
    main()
