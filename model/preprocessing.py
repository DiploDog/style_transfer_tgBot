import torchvision.transforms as transforms
from PIL import Image
import torch


# def load_image(content_path, style_path):
#     style_img = Preprocessing(imsize=512, path=content_path)  # as well as here
#     content_img = Preprocessing(imsize=512, path=style_path)
#     tens_content = content_img.image_loader()
#     tens_style = style_img.image_loader()
#     return tens_content, tens_style


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

    # @staticmethod
    # def get_image(self, path):
    #     unloader = transforms.ToPILImage()
    #     image = unloader(path)
    #     return image


# content_image, style_image = load_image(content_path="images/calm_pepo.jpg",
#                                         style_path="images/cat.jpg")