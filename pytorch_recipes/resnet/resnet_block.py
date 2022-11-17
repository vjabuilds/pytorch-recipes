import torch
from torch.nn import Module, Sequential, ReLU, Conv2d, BatchNorm2d
import torch.nn.functional as F
from torchvision.models import resnet34

class ResnetBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool):
        super(ResnetBlock, self).__init__()
        self._should_downsample = downsample
        stride = 2 if downsample else 1
        self.conv1 = Sequential(
            Conv2d(in_channels, out_channels, 3, padding=1, stride = stride),
            BatchNorm2d(out_channels), 
            ReLU()
        )
        self.conv2 = Sequential(
            Conv2d(out_channels, out_channels, 3, padding=1),
            BatchNorm2d(out_channels),
        )
        if self._should_downsample:
            self.downsample_layer = Sequential(
                Conv2d(in_channels, out_channels, stride = 2, kernel_size=1),
                BatchNorm2d(out_channels)
            )

    def forward(self, x):
        copied = x
        res = self.conv1(x)
        res = self.conv2(res)
        if self._should_downsample:
            copied = self.downsample_layer(copied)
        res = res + copied
        res = F.relu(res)
        return res