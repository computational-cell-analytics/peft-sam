import numpy as np
from math import ceil, floor

from torch_em.transform.raw import normalize


EXPERIMENT_ROOT = "/scratch/usr/nimcarot/sam/experiments/"


class RawTrafo:
    """
    Transforms raw data
    desired_shape: tuple, shape of the output
    self.padding: if true pads the image to desired_shape
    self.do_rescaling: if true rescales the image to [0, 255]
    self.triplicate_dims: if true triplicates the image to 3 channels, in case some of the datasets
                             images are RGB and some aren't
    """
    def __init__(self, desired_shape=None, do_padding=True, do_rescaling=True, padding="constant",
                 triplicate_dims=False):
        self.desired_shape = desired_shape
        self.padding = padding
        self.do_rescaling = do_rescaling
        self.triplicate_dims = triplicate_dims
        self.do_padding = do_padding

    def __call__(self, raw):
        if self.do_rescaling:
            raw = normalize(raw)
            raw = raw * 255

        if self.do_padding:
            assert self.desired_shape is not None
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


def get_peft_kwargs(peft_rank, peft_module, dropout=None, alpha=None, projection_size=None):
    if peft_module is None:
        peft_kwargs = None
    else:
        assert peft_rank is not None, "Missing rank for peft finetuning."
        from micro_sam.models.peft_sam import LoRASurgery, FacTSurgery
        if peft_module == 'lora':
            module = LoRASurgery
            peft_kwargs = {"rank": peft_rank, "peft_module": module}
        elif peft_module == 'fact':
            module = FacTSurgery
            peft_kwargs = {"rank": peft_rank, "peft_module": module, "dropout": dropout}
        elif peft_module == 'adaptformer':
            from micro_sam.models.peft_sam import AdaptFormer
            module = AdaptFormer
            if alpha != 'learnable_scalar':
                alpha = float(alpha)
            peft_kwargs = {"rank": peft_rank, "peft_module": module, "dropout": dropout, "alpha": alpha,
                           "projection_size": projection_size}
        elif peft_module == 'AttentionSurgery':
            from micro_sam.models.peft_sam import AttentionSurgery
            peft_kwargs = {"rank": 2, "peft_module": AttentionSurgery}
        elif peft_module == 'BiasSurgery':
            from micro_sam.models.peft_sam import BiasSurgery
            peft_kwargs = {"rank": 2, "peft_module": BiasSurgery}
        elif peft_module == 'LayerNormSurgery':
            from micro_sam.models.peft_sam import LayerNormSurgery
            peft_kwargs = {"rank": 2, "peft_module": LayerNormSurgery}
        elif peft_module == 'ssf':
            from micro_sam.models.peft_sam import SSFSurgery
            peft_kwargs = {"rank": 2, "peft_module": SSFSurgery}
    return peft_kwargs
