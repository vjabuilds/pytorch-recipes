import torch
from .encoder import Encoder
from .head import Head
from typing import List

class ConvClassifier(torch.nn.Module):
    def __init__(self, layers_encoder: List[int], 
                kernel_sizes: List[int], 
                layers_classifier: List[int], 
                channels: int, 
                num_classes: int):
        super(ConvClassifier, self).__init__()
        self.encoder = Encoder(layers_encoder, kernel_sizes, channels)
        self.head = Head(layers_classifier, layers_encoder[-1], num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x
        
