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

class CelebA(MaskdmDataset):

    def build_dataset(self):
        data_path = os.path.join(self.path, "img_align_resized64")
        anno_path = os.path.join(self.path, "partition.txt")
        if self.verbose:
            tqdm.write("Reading CelebA...")
        
        with open(anno_path, "r") as f:
            lines = f.readlines()

        self.package = []
        for line in lines:
            img, split = line.strip("\n").split(" ")
            img = img.split(".")[0] + ".png"
            # break
            if split == "0": # 0 denotes training set
                self.package.append(
                    {
                        "img_path": os.path.join(data_path, img),
                    }
                )
        print(f"Total training images:{len(self.package)}")

    def __getitem__(self, index):

        # try until success
        while self.MAXIMUM_RETRY:
            try:
                pack = self.package[index]
                img_path = pack["img_path"]
                img = self.load_img(img_path) # torch.Tensor
                break
            except Exception as e:
                self.log_error(str(e) + ", image path:"+ img_path + "\n")
                index = random.randint(0, len(self.package)-1)

        m = self.generate_mask()
        m = torch.from_numpy(m)

        if self.task == "uncond":
            m = torch.cat([torch.tensor([0]), m], dim=0) # skip time token
            batch = [img, m]
        else:
            raise ValueError("Unsupported task for vggface")

        return batch