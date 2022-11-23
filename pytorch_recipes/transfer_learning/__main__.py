from .model import ClassificationModel, Encoder
from .image_dataset import ImageDataset
from .train import Trainer
import torch

#blood cell image classification https://www.kaggle.com/datasets/paultimothymooney/blood-cells

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

classifier = ClassificationModel(4, layers=[2048,2048,2048,2048,2048], freeze_encoder=False, enocder=Encoder.RESNET152, dropout=0.5)
ds = ImageDataset('./pytorch_recipes/transfer_learning/dataset2-master/dataset2-master/images/TRAIN')
val_ds = ImageDataset('./pytorch_recipes/transfer_learning/dataset2-master/dataset2-master/images/TEST')
Trainer().train(ds, classifier, 20, validation_dataset=val_ds)