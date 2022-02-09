import torchvision.transforms as transforms
from PIL import Image
import torch


class Preprocessing:
    def __init__(self, imsize, img):
        self.imsize = imsize
        self.img = img
        self.loaded = self.image_loader()

    def image_loader(self):
        loader = transforms.Compose([
            transforms.Resize(self.imsize),  # нормируем размер изображения
            transforms.CenterCrop(self.imsize),
            transforms.ToTensor()])
        image = Image.open(self.img)
        image = loader(image).unsqueeze(0)
        return image.to(torch.float)

    def _get_filename(self):
        return
