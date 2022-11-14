from pytorch_recipes.dense import DenseNet
import torch

def test_dense():
    n = DenseNet([30, 30, 30], (500, 10), 5)
    print(n(torch.ones(500, 10)))