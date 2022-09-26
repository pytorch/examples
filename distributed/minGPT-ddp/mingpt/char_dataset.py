import torch
from torch.utils.data import Dataset
import fsspec
from dataclasses import dataclass

"""
Adapted from https://github.com/karpathy/minGPT/blob/master/projects/chargpt/chargpt.py
"""

@dataclass
class DataConfig:
    path: str = None
    block_size: int = None
    train_split: float = None
    truncate: float = 1.0

class CharDataset(Dataset):

    def __init__(self, data_cfg: DataConfig): #data_path: str, block_size):
        data = fsspec.open(data_cfg.path).open().read().decode('utf-8')
        data = data[ : int(len(data) * data_cfg.truncate)]

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('Data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = data_cfg.block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
