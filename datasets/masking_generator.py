# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import random
import math
import numpy as np
from einops import rearrange
from functools import partial

def shuffle_along_axis(a, axis):
    # uniform sampling
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Random Mask: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask # numpy.ndarray, shape: (N, )


class RandomBlockMaskingGenerator:
    # mask block-wisely
    def __init__(self, input_size, mask_ratio, block_size = 1):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size
        self.block_size = block_size # patch per block

        self.num_patches = self.height * self.width
        self.num_blocks = self.num_patches // (self.block_size**2)
        self.num_mask = int(mask_ratio * self.num_patches) // (self.block_size**2)

    def __repr__(self):
        repr_str = "Random Block-wise Mask: total blocks {}, masked blocks {}".format(
            self.num_blocks, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.zeros((self.height, self.width))
        mask = rearrange(mask, "(hn hb) (wn wb) -> (hn wn) (hb wb)", hn=int(math.sqrt(self.num_blocks)), wn=int(math.sqrt(self.num_blocks)))

        idx = np.hstack([
            np.zeros(self.num_blocks - self.num_mask),
            np.ones(self.num_mask),
        ]).astype(np.bool)
        # print(idx)
        np.random.shuffle(idx)
        mask[idx, :] = 1

        mask = rearrange(mask, "(hn wn) (hb wb) -> (hn hb wn wb)", hn=int(math.sqrt(self.num_blocks)), hb=self.block_size)

        return mask # numpy.ndarray, shape: (N, )


class CropMaskingGenerator:
    def __init__(self, input_size, crop_size):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2

        self.height, self.width = input_size
        self.croph, self.cropw = crop_size

    def __repr__(self):
        repr_str = "Crop Mask: crop size {},{}".format(
            self.croph, self.cropw
        )
        return repr_str

    def __call__(self):
        hbound = self.height - self.croph
        wbound = self.width - self.cropw
        topleft_h = np.random.randint(0, hbound+1)
        topleft_w = np.random.randint(0, wbound+1)

        idx_range = [i*self.width+j for j in range(topleft_w, topleft_w+self.cropw) for i in range(topleft_h, topleft_h+self.croph)]

        mask = np.ones((self.height, self.width))
        mask.put(idx_range, 0)

        return mask.flatten()