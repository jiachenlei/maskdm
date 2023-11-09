import os

import PIL
import random
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.transforms.functional as F

from .base import MaskdmDataset

class LSUN(MaskdmDataset):
    # Church
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "lsun"

    def build_dataset(self):
        data_path = self.path

        if self.verbose:
            tqdm.write(f"Reading LSUN church...")        
        self.package = [os.path.join(data_path, img) for img in os.listdir(data_path) ]

    def init_transform(self):
        if self.mode == "train":
            self.data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(self.hflip_prob),
            ])
        else:
            raise NotImplementedError("No transformations are defined for [test/val] mode")

    def __getitem__(self, index):

        img  = self.load_img(self.package[index])
        m = self.generate_mask()
        m = torch.from_numpy(m)
        m = torch.cat([torch.tensor([0]), m], dim=0) # skip time token
        return [img, m]

    def __len__(self):
        return len(self.package)