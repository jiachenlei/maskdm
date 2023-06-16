import os

import PIL
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.transforms.functional as F

from .utils import ShortSideResize
from .base import MaskdmDataset

class VggFace(MaskdmDataset):


    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.name = "vggface"

    def build_dataset(self):
        data_path = os.path.join(self.path)
        identity = os.listdir(data_path)

        if self.verbose:
            tqdm.write("Reading vggface...")

        for label_idx, id in enumerate(tqdm(identity,  disable = not self.verbose)):
            imgs = os.listdir(os.path.join(data_path, id))

            self.package.extend([
                {
                    "raw_label": id,
                    "label": label_idx,
                    "img_path": os.path.join(data_path, id, img),
                }
                for img in imgs
            ])

    def init_transform(self):
        if self.mode == "train":
            self.data_transform = transforms.Compose([
                transforms.ToTensor(),
                # ShortSideResize(self.short_side_size),
                transforms.RandomHorizontalFlip(self.hflip_prob),
                # transforms.CenterCrop((self.imsize, self.imsize)),
                # transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            raise NotImplementedError("No transformations are defined for [test/val] mode")
