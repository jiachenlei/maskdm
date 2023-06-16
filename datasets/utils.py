import torchvision.transforms.functional as F

class ShortSideResize():

    def __init__(self, size: int):

        self.size = size

    def __call__(self, img):
        # img: torch.Tensor

        *_, h, w = img.shape

        if h == self.size or w == self.size:
            return img
        elif h > w:
            newh = int(h/w * self.size)
            neww = self.size
        elif h <= w:
            newh = self.size
            neww = int(w/h * self.size)
        
        resized_img = F.resize(img, (newh, neww), interpolation=F.InterpolationMode.BICUBIC)

        return resized_img