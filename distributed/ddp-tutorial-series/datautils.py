import torch
from torch.utils.data import Dataset

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]

    
class MyRandomDataset(Dataset):
    def __init__(self, size, input_shape, target_shape=(1000,)):
        self.size = size
        self.shape = input_shape
        self.target_shape = target_shape

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return (torch.rand(self.shape), torch.rand(self.target_shape))
