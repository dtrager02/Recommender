import torch
from torch.utils.data import Dataset
from collections import defaultdict
from time import perf_counter
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from time import perf_counter
class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training
    
    Args:
        path (str): path to npz file
        num_samples (int): maximum number of samples to read
    
    """

    def __init__(self,data:np.ndarray=None):
        self.users, self.items, self.times, self.labels = self.get_dataset(data)
        self.n_users = int(self.users.max())+1
        self.n_items = int(self.items.max())+1
        for vec in (self.users,self.items,self.times,self.labels):
            vec = torch.from_numpy(vec).to(self.device)
        self.device = torch.device("cpu")
        if torch.cuda.is_available():  
          self.device = torch.device("cuda:0")
        print(self.users.get_device())


    def __len__(self):
        return len(self.users)
  
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.times[idx], self.labels[idx]

    def get_dataset(self,data):
        return data[:,0], data[:,1], data[:,3],data[:,2]

    
    