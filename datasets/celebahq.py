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

    def init_dataset(self):
        self.use_dwt = getattr(self.cfg, "USE_DWT", False)
        if self.use_dwt:
            print("Using Discrete Wavelet transformation")

    def build_dataset(self):
        data_path = os.path.join(self.path)

        if self.verbose:
            tqdm.write("Reading CelebA-HQ...")

        imgs = os.listdir(data_path)

        self.package = [
            {
                "img_path": os.path.join(data_path, img),
            }
            for img in imgs
        ]

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


        if self.use_dwt:
            cA, (cH, cV, cD) = pywt.dwt2(img,"haar") # c, h, w
            img = torch.from_numpy(np.concatenate((cA, cH, cV, cD), axis=0)) # 12 dimensions

        m = self.generate_mask()
        m = torch.from_numpy(m)

        if self.task == "cond":
            raise NotImplementedError(f"Conditional training on CelebA-HQ is not supported")
        elif self.task == "uncond":
            m = torch.cat([torch.tensor([0]), m], dim=0) # skip time token
            batch = [img]
        else:
            raise ValueError("Unsupported task for vggface")

        batch.insert(1, m)

        return batch