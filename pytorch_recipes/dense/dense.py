from typing import List, Tuple

import torch
from torch.nn import Module
from torch.nn import Linear
from torch.nn import functional as F

class DenseNet(Module):
    def __init__(self, layers: List[int], data_shape: Tuple[int, int], num_classes: int):
        """
        The constructor of the DenseNet architecture. To be used in n-class classification.
        - layers : defines the size of each of the layers in the DenseNet
        - data_shape : the shape of the data that will be used furing inference
        """
        super(DenseNet, self).__init__()
        inputs = [data_shape[1]] + layers[:-1]
        outputs = layers[1:] + [num_classes]
        shape = zip(inputs, outputs)
        self._layers = []
        for s in shape:
            self._layers.append(Linear(s[0], s[1], True, dtype=torch.float32))
        self._layers = torch.nn.ModuleList(self._layers)

    def forward(self, x):
        """
        Runs inference on the supplied data point.
        """
        for i, layer in enumerate(self._layers):
            x = layer(x)
            if i != len(self._layers) - 1:
                x = F.relu(x)
        return x