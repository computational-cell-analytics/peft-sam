import numpy as np
from math import ceil, floor

from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.transform.raw import normalize_percentile, normalize



class RawTrafo:
    """
    Transforms raw data
    desired_shape: tuple, shape of the output
    self.padding: if true pads the image to desired_shape
    self.do_rescaling: if true rescales the image to [0, 255]
    self.triplicate_dims: if true triplicates the image to 3 channels, in case some of the datasets
                             images are RGB and some aren't
    """
    def __init__(self, desired_shape=None, do_padding=True, do_rescaling=False, padding="constant", triplicate_dims=False):
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
                raw = np.stack((raw, raw, raw), axis = 0)

        return raw


class LabelTrafo:
    """
    Transform for Labels
    """
    def __init__(self, desired_shape=None, padding="constant", min_size=0, do_padding=True):
        self.padding = padding
        self.desired_shape = desired_shape
        self.min_size = min_size
        self.do_padding = do_padding

    def __call__(self, labels):
        if labels.ndim == 3:
            assert labels.shape[0] == 1
            labels = labels[0]

        distance_trafo = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=self.min_size
        )

        labels = distance_trafo(labels)

        if self.do_padding:
            # choosing H and W from labels (4, H, W), from above dist trafo outputs
            assert self.desired_shape is not None
            tmp_ddim = (self.desired_shape[0] - labels.shape[1], self.desired_shape[1] - labels.shape[2])
            ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2)
            labels = np.pad(
                labels,
                pad_width=((0,0), (ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1]))),
                mode=self.padding
            )
            assert labels.shape[1:] == self.desired_shape, labels.shape

        return labels
