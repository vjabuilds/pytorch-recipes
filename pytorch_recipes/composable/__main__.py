from .conv_classifier import ConvClassifier
from .image_dataset import ImageDataset
from .train import Trainer
import torch

#intel image classification https://www.kaggle.com/datasets/puneet6060/intel-image-classification?select=seg_train

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

classifier = ConvClassifier([64, 128, 512, 512, 512], [3, 3, 3, 5, 5], [1024, 1024], 3, 6)
ds = ImageDataset('./pytorch_recipes/conv/data/seg_train/seg_train')
Trainer().train(ds, classifier, 20)