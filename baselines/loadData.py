import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np

class DiskDataset(Dataset):
    def __init__(self, data_path, label_path):

        self.data = np.load(data_path)
        self.label = np.load(label_path)
        
    def __getitem__(self, index):
        
        data = self.data[index]
        label = self.label[index]
        return torch.from_numpy(np.array(data)), torch.from_numpy(np.array(label))
        
    def __len__(self):
        return self.data.shape[0]
    
