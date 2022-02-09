import torch
import torch.nn as nn


class Normalization(nn.Module):
    _mean = torch.tensor([0.485, 0.456, 0.406])
    _std = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(self._mean).view(-1, 1, 1)
        self.std = torch.tensor(self._std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
