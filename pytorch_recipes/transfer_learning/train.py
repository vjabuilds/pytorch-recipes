import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import time


class Trainer():
    def train(self, dataset, network, epochs, validation_dataset = None):
        loader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=2)
        crit = nn.CrossEntropyLoss()
        optimizer = optim.Adam(network.parameters(), lr=0.00001)
        network = network.to('cuda')
        for epoch in range(epochs):
            epoch_start = time.time()
            running_loss = 0.0
            for i, data in enumerate(loader):
                input, label = data
                input = input.to('cuda')
                label = label.to('cuda')
                optimizer.zero_grad()
                res = network(input)
                loss = crit(res, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 20 == 19:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20}')
                    running_loss = 0.0
            epoch_end = time.time()
            if validation_dataset is not None:
                with torch.no_grad():
                    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=2)
                    correct = torch.tensor(0, device='cuda')
                    for i, data in enumerate(validation_loader):
                        input, label = data
                        input = input.to('cuda')
                        label = label.to('cuda')
                        res: torch.Tensor = network(input)
                        correct += (torch.argmax(label, dim=1) == torch.argmax(res, dim=1)).sum()
                    print(f'The validation accuracy was {correct / len(validation_dataset)}')
                                            

            print(f'It took {epoch_end - epoch_start:5f} seconds to finish one training epoch')
        torch.save(network.state_dict(), 'weights_frozen.pth')