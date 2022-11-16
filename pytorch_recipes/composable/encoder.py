import torch
from torch.nn import Module, Conv2d, ModuleList
from typing import List
import torch.nn.functional as F

class Encoder(Module):
    def __init__(self, layers_encoder: List[int], kernel_sizes: List[int], channels: int):
        super(Encoder, self).__init__()
        encoder_inputs = [channels] + layers_encoder[:-1]
        encoder_outputs = layers_encoder
        encoder_shape = zip(encoder_inputs, encoder_outputs, kernel_sizes)

        self.encoder = []
        for items in encoder_shape:
            self.encoder.append(Conv2d(items[0], items[1], items[2]))
        self.encoder = ModuleList(self.encoder)

    def forward(self, x):
        for ind, conv in enumerate(self.encoder):
            x = conv(x)
            x = F.relu(x)
            if ind != len(self.encoder):
                x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return x