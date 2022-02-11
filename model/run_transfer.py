import torch
from torchvision import transforms
from model.model import GatysNet, GatysTransfer
from model.preprocessing import Preprocessing


def run(content_image, style_image, num_steps=100):
    style_img = Preprocessing(imsize=512, img=style_image)
    content_img = Preprocessing(imsize=512, img=content_image)
    tens_content = content_img.image_loader()
    tens_style = style_img.image_loader()
    my_model = GatysNet()
    my_model.load_state_dict(torch.load('model/Gatys.model'))
    transfer = GatysTransfer(my_model.features.eval())
    transfer.build_model(tens_content, tens_style)
    pic = transfer.run(num_steps)
    pic = pic.squeeze(0)
    unloader = transforms.ToPILImage()
    pic = unloader(pic)
    pic.save('images/result.jpg')
    return pic
