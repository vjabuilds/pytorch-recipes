from pytorch_recipes.dense.dense import DenseNet
from pytorch_recipes.dense.pandas_dataset import Formats, PandasDataset
from .train import Trainer

feature_number = 784
class_number = 10
dataset = PandasDataset('./fashion-mnist_train.csv', Formats.CSV, 'label', class_number)
dn = DenseNet([512, 512, 512], [len(dataset), feature_number], class_number)

Trainer().train(dataset, dn, 20)