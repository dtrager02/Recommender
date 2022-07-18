import torch
from torch.utils.data import Dataset
from time import perf_counter
import numpy as np
class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training
    
    Args:
        path (str): path to npz file
        num_samples (int): maximum number of samples to read
    
    """

    def __init__(self, path:str, num_samples:int,data:np.ndarray=None):
        self.users, self.items, self.labels = self.get_dataset(path, num_samples,data=data)

    def __len__(self):
        return len(self.users)
  
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, path, n,data):
        if data:
            return torch.tensor(data[:,0]), torch.tensor(data[:,1]), torch.tensor(data[:,2])
        a = np.load(path)
        if n != "all":
            a = a[:n] 
        return torch.tensor(a[:,0]), torch.tensor(a[:,1]), torch.tensor(a[:,2])