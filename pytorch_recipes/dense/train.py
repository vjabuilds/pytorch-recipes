import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from .pandas_dataset import PandasDataset
from .pandas_dataset import Formats


class Trainer():
    def train(self, dataset, network, epochs):
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        crit = nn.CrossEntropyLoss()
        optimizer = optim.Adam(network.parameters(), lr=0.001)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(loader):
                input, label = data
                optimizer.zero_grad()
                res = network(input)
                loss = crit(res, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 200 == 199:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                    running_loss = 0.0
        torch.save(network.state_dict(), 'weights.pth')