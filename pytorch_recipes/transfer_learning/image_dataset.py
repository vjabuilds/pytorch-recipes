import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor
import torch.nn.functional as F
import os
from PIL import Image
from typing import List

class ImageDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.classes: List[str] = os.listdir(path)
        self.resize = Resize((320, 240))
        self.toTensor = ToTensor()
        self.class_paths: List[str] = []
        for c in self.classes:
            class_path = os.path.join(self.path, c)
            for img_path in os.listdir(class_path):
                self.class_paths.append((c, os.path.join(class_path, img_path)))
                    
        

    def __len__(self):
        return len(self.class_paths)
    
    def __getitem__(self, index: int):
        label, path = self.class_paths[index]
        target = F.one_hot(torch.tensor(self.classes.index(label)), len(self.classes)).float()
        data = self.toTensor(Image.open(path))
        data = self.resize(data)
        return data, target
    
