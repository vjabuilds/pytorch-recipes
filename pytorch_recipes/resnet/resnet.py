import torch
from torch.nn import Module, ModuleList, Conv2d, BatchNorm2d, Sequential, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear
from .resnet_block import ResnetBlock
import torch.nn.functional as F
from typing import List

class Resnet34(Module):
    def __renset_list(self, in_size: int, length: int) -> List[ResnetBlock]:
        return [ResnetBlock(in_size, in_size *2, True)] + [ResnetBlock(in_size * 2, in_size*2, False)] * (length - 1)

    def __init__(self, num_classes: int):
        super(Resnet34, self).__init__()
        self.starting = Sequential(
            Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(64),ReLU(),
            MaxPool2d(kernel_size = 3, stride = 2)
        )
        current_size = 64
        layers_list = [ResnetBlock(current_size, current_size, False)] * 3
        while current_size < 512:
            layers_list = layers_list + self.__renset_list(current_size, 3)
            current_size *= 2
        self.layers_list = ModuleList(layers_list)
        self.pooling = AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = Linear(512, num_classes)
    
    def forward(self, x):
        x = self.starting(x)
        for l in self.layers_list:
            x = l(x)
        x = self.pooling(x).flatten(1)
        x = self.classifier(x)
        return x

        
        