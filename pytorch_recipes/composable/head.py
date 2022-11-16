from torch.nn import Module, Linear, ModuleList
from typing import List
import torch.nn.functional as F

class Head(Module):
    def __init__(self, layers_classifier: List[int], input_size: int, num_classes: int):
        super(Head, self).__init__()
        classifier_inputs = [input_size] + layers_classifier[1:]
        classifier_outputs = layers_classifier[1:] + [num_classes]
        classifier_shape = zip(classifier_inputs, classifier_outputs)

        self.classifier = []
        for items in classifier_shape:
            self.classifier.append(Linear(items[0], items[1]))
        self.classifier = ModuleList(self.classifier)

    def forward(self, x):
        for l in self.classifier:
            x = l(x)
            x = F.relu(x)
        return x