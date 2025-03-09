import numpy as np
from math import ceil, floor
from typing import Tuple, Callable, Optional, List

from torch_em.transform.raw import normalize

from micro_sam.models import peft_sam


EXPERIMENT_ROOT = "/scratch/usr/nimcarot/sam/experiments/"


class RawTrafo:
    """Transforms the input data.

    Args:
        desired_shape: The desired patch shape to perform padding transformations.
        do_padding: Whether to pad the inputs to match the 'desired_shape'.
        do_rescaling: Whether to normalize the inputs.
        padding: The choice of padding mode.
        triplicate_dims: Whether to have the inputs in RGB style, i.e. 3 channels in the last axis.
    """
    def __init__(
        self,
        desired_shape: Tuple[int, int],
        do_padding: bool = True,
        do_rescaling: bool = True,
        padding: str = "constant",
        triplicate_dims: bool = False,
    ):
        self.desired_shape = desired_shape
        self.padding = padding
        self.do_rescaling = do_rescaling
        self.triplicate_dims = triplicate_dims
        self.do_padding = do_padding

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        if self.do_rescaling:
            raw = normalize(raw)
            raw = raw * 255

        if self.do_padding:
            tmp_ddim = (self.desired_shape[-2] - raw.shape[-2], self.desired_shape[-1] - raw.shape[-1])
            ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
            raw = np.pad(
                raw,
                pad_width=((ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))),
                mode=self.padding
            )
            assert raw.shape[-2:] == self.desired_shape[-2:], raw.shape

        if self.triplicate_dims:
            if raw.ndim == 3 and raw.shape[0] == 1:
                raw = np.concatenate((raw, raw, raw), axis=0)
            if raw.ndim == 2:
                raw = np.stack((raw, raw, raw), axis=0)

        return raw


def get_default_peft_kwargs(method: str):
    """Functionaltiy to get the default (best) PEFT arguments.

    Args:
        method: The name of PEFT method.

    Returns:
        A dictionary with predefined peft arguments.
    """
    supported_peft_methods = [
        "lora", "qlora", "fact", "attention_tuning", "adaptformer",
        "bias_tuning", "layernorm_tuning", "ssf", "late_lora",
    ]

    if method is None:
        peft_kwargs = {}
    else:
        if method == "lora":
            peft_kwargs = get_peft_kwargs(peft_rank=32, peft_module=method)

        elif method == "qlora":
            peft_kwargs = get_peft_kwargs(peft_rank=32, peft_module="lora", quantize=True)

        elif method == "fact":
            peft_kwargs = get_peft_kwargs(peft_rank=16, peft_module=method, dropout=0.1)

        elif method == "attention_tuning":
            peft_kwargs = get_peft_kwargs(peft_module="AttentionSurgery")

        elif method == "bias_tuning":
            peft_kwargs = get_peft_kwargs(peft_module="BiasSurgery")

        elif method == "layernorm_tuning":
            peft_kwargs = get_peft_kwargs(peft_module="LayerNormSurgery")

        elif method == "ssf":
            peft_kwargs = get_peft_kwargs(peft_module=method)

        elif method == "adaptformer":
            peft_kwargs = get_peft_kwargs(
                peft_module=method, alpha="learnable_scalar", dropout=None, projection_size=64,
            )

        elif method == "late_lora":
            peft_kwargs = get_peft_kwargs(
                peft_rank=32,
                peft_module="lora",
                attention_layers_to_update=list(range(6, 12)),
                update_matrices=["q", "k", "v", "mlp"],
            )

        else:
            raise ValueError(f"Please choose a valid peft method from: '{supported_peft_methods}'.")

    return peft_kwargs


def get_peft_kwargs(
    peft_module: Callable,
    peft_rank: int = None,
    dropout: Optional[float] = None,
    alpha: Optional[float] = None,
    projection_size: Optional[int] = None,
    quantize: bool = False,
    attention_layers_to_update: List[str] = [],
    update_matrices: List[str] = ["q", "v"],
):
    """Functionality to get the necessary arguments in a dictionary of expected arguments under 'peft_kwargs'.

    Args:
        peft_rank: The choice of rank for the peft method.
        peft_module: The desired peft module to run the peft method.
        dropout: Whether to use dropout for the supported peft method, eg. FacT and AdaptFormer.
        alpha: Whether to use the scaling parameter in Adaptformer.
        projection_sie: The choice of projection size in AdaptFormer.
        quantize: Whether to quantize the base foundation model for QLoRA.

    Returns:
        A dictionary with all arguments and corresponding values.
    """
    if update_matrices is None:
        update_matrices = ["q", "v"]
    if peft_module is None:
        peft_kwargs = None
    else:
        if peft_module == 'lora':
            peft_kwargs = {
                "rank": peft_rank,
                "peft_module": peft_sam.LoRASurgery,
                "quantize": quantize,
                "attention_layers_to_update": attention_layers_to_update,
                "update_matrices": update_matrices,
            }

        elif peft_module == 'fact':
            peft_kwargs = {"rank": peft_rank, "peft_module": peft_sam.FacTSurgery, "dropout": dropout}

        elif peft_module == 'adaptformer':
            if alpha != 'learnable_scalar':
                alpha = float(alpha)

            peft_kwargs = {
                "peft_module": peft_sam.AdaptFormer,
                "dropout": dropout,
                "alpha": alpha,
                "projection_size": projection_size,
            }

        elif peft_module == 'AttentionSurgery':
            peft_kwargs = {"peft_module": peft_sam.AttentionSurgery}

        elif peft_module == 'ClassicalSurgery':
            peft_kwargs = {
                "peft_module": peft_sam.ClassicalSurgery,
                "attention_layers_to_update": attention_layers_to_update
            }

        elif peft_module == 'BiasSurgery':
            from micro_sam.models.peft_sam import BiasSurgery
            peft_kwargs = {"peft_module": BiasSurgery}

        elif peft_module == 'LayerNormSurgery':
            peft_kwargs = {"peft_module": peft_sam.LayerNormSurgery}

        elif peft_module == 'ssf':
            peft_kwargs = {"peft_module": peft_sam.SSFSurgery}

        else:
            raise ValueError(peft_module)

    return peft_kwargs
