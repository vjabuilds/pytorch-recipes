import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer():
    def train(self, dataset, network, epochs):
        loader = DataLoader(dataset, batch_size=96, shuffle=True, pin_memory=True, num_workers=2)
        crit = nn.CrossEntropyLoss()
        optimizer = optim.Adam(network.parameters(), lr=0.00001)
        network = network.to('cuda')
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(loader):
                input, label = data
                input = input.to('cuda')
                label = label.to('cuda')
                optimizer.zero_grad()
                results = network(input)
                res = results[-1]
                loss = crit(res, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1}')
                running_loss = 0.0
        torch.save(network.state_dict(), 'weights.pth')