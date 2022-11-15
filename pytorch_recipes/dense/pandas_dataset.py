import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from enum import Enum

class Formats(Enum):
    CSV = 1
    PARQUET = 2
    FEATHER = 3

class PandasDataset(Dataset):
    def __init__(self, path: os.path, format: Formats, target_col: str, labels_count: int):
        if format == Formats.CSV:
            self._df = pd.read_csv(path)
        elif format == Formats.PARQUET:
            self._df = pd.read_parquet(path)
        elif format == Formats.FEATHER:
            self._df = pd.read_feather(path)
        self._features = self._df.drop(columns = target_col)
        self._targets = self._df[target_col]
        self._labels_count = labels_count
        
    def __len__(self):
        return len(self._df)
    
    def __getitem__(self, index: int):
        targets = np.zeros(self._labels_count, dtype=np.float32)
        targets[self._targets[index]] = 1.0
        return self._features.iloc[index, :].to_numpy(dtype=np.float32), targets