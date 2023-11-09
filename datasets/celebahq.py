import os

import PIL
import random
from PIL import Image
from tqdm import tqdm
import numpy as np

import pywt
import torch
from torch import nn
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.transforms.functional as F

from .base import MaskdmDataset

class CelebAHQ(MaskdmDataset):

    def build_dataset(self):
        data_path = os.path.join(self.path)

        if self.verbose:
            tqdm.write("Reading CelebA-HQ...")

        self.package = [
            os.path.join(data_path, img) for img in os.listdir(data_path)
        ]

    def init_transform(self):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(self.hflip_prob),
        ])


    def __getitem__(self, index):

        img_path = self.package[index]
        img = self.load_img(img_path) # torch.Tensor

        m = self.generate_mask()
        m = torch.from_numpy(m)
        m = torch.cat([torch.tensor([0]), m], dim=0) # skip time token

        return [img, m]