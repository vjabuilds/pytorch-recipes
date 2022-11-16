import torch
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, ModuleList
from typing import List

class ConvClassifier(torch.nn.Module):
    def __init__(self, layers_encoder: List[int], 
                kernel_sizes: List[int], 
                layers_classifier: List[int], 
                channels: int, 
                num_classes: int):
        super(ConvClassifier, self).__init__()
        encoder_inputs = [channels] + layers_encoder[:-1]
        encoder_outputs = layers_encoder
        encoder_shape = zip(encoder_inputs, encoder_outputs, kernel_sizes)

        classifier_inputs = [layers_encoder[-1]] + layers_classifier[1:]
        classifier_outputs = layers_classifier[1:] + [num_classes]
        classifier_shape = zip(classifier_inputs, classifier_outputs)

        self.encoder = []
        for items in encoder_shape:
            self.encoder.append(Conv2d(items[0], items[1], items[2]))
        self.encoder = ModuleList(self.encoder)

        self.classifier = []
        for items in classifier_shape:
            self.classifier.append(Linear(items[0], items[1]))
        self.classifier = ModuleList(self.classifier)

    def forward(self, x):
        for ind, conv in enumerate(self.encoder):
            x = conv(x)
            x = F.relu(x)
            if ind != len(self.encoder):
                x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        for l in self.classifier:
            x = l(x)
            x = F.relu(x)
        return x
        
