import os

import random
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

from .utils import ShortSideResize

class MaskdmDataset(Dataset):

    def __init__(self, cfg, mode, mask_generator, name="base", verbose=True, **kwargs):
        
        super().__init__()

        self.name = name # name of dataset
        self.cfg = cfg   # configuration, type: Namespace
        self.mode = mode # specify split of dataset, default: "train"
        self.task = self.cfg.TASK # specify conditional or unconditional generation
        self.mask_generator = mask_generator # generator for mask
        self.verbose = verbose # whether detailed data loading information

        self.path = self.cfg.PATH # path to dataset
        self.hflip_prob = self.cfg.hflip_prob # probability of horizontal flip
        self.package = [] # variable that stores all data information, e.g., path to image, label, etc.

        # additional arguments, reserved for future use
        self.kwargs = kwargs

        # custom functions reserved for child classes to prepare/process the dataset
        self.init_dataset()
        self.build_dataset() 
        self.init_transform()

    def init_dataset(self):
        pass

    def build_dataset(self):
        raise NotImplementedError("build_dataset() not implemented!")

    def init_transform(self):
        raise NotImplementedError("No transformations are defined for [test/val] mode")

    def __getitem__(self, index):
        raise NotImplementedError()

    def generate_mask(self):
        return self.mask_generator()

    def load_img(self, path):
        img = Image.open(path)
        img = self.data_transform(img)
        return img

    def __len__(self):
        return len(self.package)