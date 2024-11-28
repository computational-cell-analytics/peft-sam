import bitsandbytes as  bnb

import torch

from micro_sam.util import get_sam_model


def check_qlora():
    _, model = get_sam_model(model_type="vit_b", return_sam=True)

    for name, module in model.image_encoder.named_modules():
        if isinstance(module, torch.nn.Linear):
            *parent_path, layer_name = name.split(".")
            parent_module = model.image_encoder

            for sub_module in parent_path:
                parent_module = getattr(parent_module, sub_module)

            setattr(parent_module, layer_name, bnb.nn.Linear4bit(module.in_features, module.out_features))

    breakpoint()


def main():
    check_qlora()


if __name__ == "__main__":
    main()
