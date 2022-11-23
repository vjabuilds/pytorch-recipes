import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet152, ResNet152_Weights
from torch.nn import Module, ModuleList, AdaptiveAvgPool2d, Sequential, Linear, ReLU, Dropout
from enum import Enum
from typing import List

class Encoder(Enum):
    RESNET50 = 0
    RESNET152 = 1



class ClassificationModel(Module):
    def __init__(self, num_classes: int, layers: List[int], freeze_encoder = False, enocder = Encoder.RESNET50, dropout = 0.0):
        super(ClassificationModel, self).__init__()
        with torch.no_grad():
            if enocder == Encoder.RESNET50:
                resnet = resnet50(weights = ResNet50_Weights.DEFAULT)
            else:
                resnet = resnet152(weights = ResNet152_Weights.DEFAULT)
            self.encoder = ModuleList([
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2, # the output has 512 channels here
                resnet.layer3, # the output has 1024 channels here
                resnet.layer4, # the output has 2048 channels here
                AdaptiveAvgPool2d(output_size=(1,1))]
            )
            for enc in self.encoder:
                for param in enc.parameters():
                    param.requires_grad = not freeze_encoder
        input_shape = 2048
        inputs = [input_shape] + layers[:-1]
        outputs = layers[1:] + [num_classes]
        shape = zip(inputs, outputs)
        layers = []
        for i, s in enumerate(shape):
            layers.append(Linear(s[0], s[1], True, dtype=torch.float32))
            if i != len(inputs) - 1:
                layers.append(ReLU(inplace = True))
                if dropout != 0:
                    layers.append(Dropout(dropout))
        self.classifier = Sequential(*layers)

    def forward(self, x):
        encoded = x
        for enc in self.encoder:
            encoded = enc(encoded)
        encoded = encoded.flatten(1)
        classified = self.classifier(encoded)
        return classified