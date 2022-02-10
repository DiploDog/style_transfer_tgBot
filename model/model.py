import gc
from torchvision.models import vgg19
from copy import deepcopy
from model.Normalization import Normalization
from torch.optim import LBFGS
from model.STLoss import ContentLoss, StyleLoss
import torch.nn as nn
import warnings
warnings.simplefilter('ignore', UserWarning)


class Model:

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    cnn = vgg19(pretrained=True).features

    def __init__(self, content_image, style_image, style_weight=10000, content_weight=0.1, num_steps: int = 20):
        self.content_image = content_image
        self.style_image = style_image
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.num_steps = num_steps
        self.run = 0

    def get_style_model_and_losses(self):
        cnn = deepcopy(self.cnn)

        # normalization module
        normalization = Normalization()

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                # Переопределим relu уровень
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                target = model(self.content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)
                del target
                del content_loss
                gc.collect()

            if name in self.style_layers:
                # add style loss:
                target_feature = model(self.style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
                del style_loss
                del target_feature
                gc.collect()

        # now we trim off the layers after the last content and style losses
        # выбрасываем все уровни после последенего styel loss или content loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]
        del cnn
        gc.collect()

        return model, style_losses, content_losses

    @staticmethod
    def get_input_optimizer(input_image):
        # this line to show that input is a parameter that requires a gradient
        #добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров
        optimizer = LBFGS([input_image.requires_grad_()])
        return optimizer

    def run_style_transfer(self, input_image):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses()
        optimizer = self.get_input_optimizer(input_image)

        def closure():
            # correct the values
            # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
            input_image.data.clamp_(0, 1)

            optimizer.zero_grad()

            model(input_image)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            # взвешивание ощибки
            style_score *= self.style_weight
            content_score *= self.content_weight

            loss = style_score + content_score
            loss.backward()

            print("run {}:".format(self.run))
            print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                style_score.item(), content_score.item()))
            print()
            gc.collect()
            return style_score + content_score

        print('Optimizing..')
        while self.run <= self.num_steps:

            gc.collect()
            optimizer.step(closure)
            self.run += 1

        input_image.data.clamp_(0, 1)
        return input_image

