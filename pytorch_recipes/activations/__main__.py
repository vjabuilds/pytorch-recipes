from .conv_classifier import ConvClassifier
from .image_dataset import ImageDataset
from .train import Trainer
import torch
from matplotlib import pyplot

#intel image classification https://www.kaggle.com/datasets/puneet6060/intel-image-classification?select=seg_train

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

classifier = ConvClassifier([64, 128, 512, 512, 512], [3, 3, 3, 5, 5], [1024, 1024], 3, 6)
ds = ImageDataset('./pytorch_recipes/conv/data/seg_train/seg_train')
#Trainer().train(ds, classifier, 20)
with torch.no_grad():
    classifier.load_state_dict(torch.load('weights.pth'))
    example = torch.unsqueeze(ds[0][0], 0)
    results = classifier(example)
    print(results[1].shape)
    #import pdb;pdb.set_trace()
    for i, activation in enumerate(results[1][0]):
        pyplot.matshow(activation, i)
        pyplot.show()
        print('Nice!')