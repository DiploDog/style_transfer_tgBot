from torchvision import transforms
from model.model import Model
from model.preprocessing import Preprocessing


def run(content_image, style_image):
    style_img = Preprocessing(imsize=512, img=style_image)  # as well as here
    content_img = Preprocessing(imsize=512, img=content_image)
    tens_content = content_img.image_loader()
    tens_style = style_img.image_loader()
    my_model = Model(tens_content, tens_style)
    input_img = tens_content.clone()
    pic = my_model.run_style_transfer(input_img)
    pic = pic.squeeze(0)
    unloader = transforms.ToPILImage()
    pic = unloader(pic)
    pic.save('images/result.jpg')
    return pic
