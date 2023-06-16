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
        self.imsize = self.cfg.IMG_SIZE # image size returned in __getitem__
        self.short_side_size = self.cfg.SHORT_SIDE_SIZE # if requires resize, keep h-w ratio and specify the short side size
        self.mean = self.cfg.MEAN # mean of dataset (not used, reserved for future use)
        self.std = self.cfg.STD   # std of dataset (not used, reserved for future use)
        self.hflip_prob = self.cfg.hflip_prob # probability of horizontal flip
        self.MAXIMUM_RETRY = 10 # reloading image from the dataset for MAXIMUM_RETRY times, in case corrupted data exist
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
        # default transformation, can be overwritten in child classes
        if self.mode == "train":
            if self.short_side_size <= 0:
                self.data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((self.imsize, self.imsize)),
                    transforms.RandomHorizontalFlip(self.hflip_prob),
                    # transforms.Normalize(mean=self.mean, std=self.std)
                ])
            else:
                self.data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    ShortSideResize(self.short_side_size),
                    transforms.CenterCrop((self.imsize, self.imsize)),
                    transforms.RandomHorizontalFlip(self.hflip_prob),
                    # transforms.Normalize(mean=self.mean, std=self.std)
                ])
        else:
            raise NotImplementedError("No transformations are defined for [test/val] mode")

    def log_error(self, info):

        # with open("error.log", "a+") as f:
        #     f.write(info)

        print(f"Error occured: {info}")

    def __getitem__(self, index):

        # try until success
        while self.MAXIMUM_RETRY:
            try:
                pack = self.package[index]
                img_path = pack["img_path"]
                label = pack["label"]
                img = self.load_img(img_path) # torch.Tensor
                break
            except Exception as e:
                self.log_error(str(e) + ", image path:"+ img_path + "\n")
                index = random.randint(0, len(self.package)-1)

        m = self.generate_mask()
        m = torch.from_numpy(m)
        # print(img.shape)
        if self.task == "cond":
            m = torch.cat([torch.tensor([0, 0]), m], dim=0) # skip time and condition tokens
            return img, m, int(label)
        elif self.task == "uncond":
            m = torch.cat([torch.tensor([0]), m], dim=0) # skip time token
            return img, m
        else:
            raise ValueError(f"Unsupported task for {self.name}")

    def generate_mask(self):
        return self.mask_generator()

    def load_img(self, path):

        img = Image.open(path)
        img = self.data_transform(img)
        return img

    def __len__(self):
        return len(self.package)