import torch
from collections import defaultdict
from torch.utils.data import Dataset
from time import perf_counter
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from time import perf_counter
class MovieLensTrainDataset(Dataset):
    """This is an implementation of https://www.sciencedirect.com/science/article/pii/S2405959518302029
    The reason it sucks is because the research paper does not account for the high cost of loading data from a sparse data structure. Even a relatively small 100k rating dataset requires roughly 3 minutes of data loading per epoch, and most of this cost is just indexing the sparse scipy matrix. I don't think there exists a more effective way of implementing this research paper at large scale, so this approach is dead.
    
    Args:
        path (str): path to npz file
        num_samples (int): maximum number of samples to read
    
    """

    def __init__(self,data:np.ndarray=None,batch_size=1024):
        self.batch_size = batch_size
        self.users, self.items, self.times, self.labels = self.get_dataset(data)
        self.n_users = int(self.users.max())+1
        self.n_items = int(self.items.max())+1
        self.sparse_ratings = sparse.coo_matrix((self.labels, (self.users.astype(np.int32), self.items.astype(np.int32))),shape=(self.n_users,self.n_items),dtype=np.float32).tocsr()
        self.sparse_times = sparse.coo_matrix((self.times, (self.users.astype(np.int32), self.items.astype(np.int32))),shape=(self.n_users,self.n_items),dtype=np.float32).tocsr()
        print(np.count_nonzero(self.sparse_ratings[[1],:].toarray()[0]),len(self.users[self.users == 1]))
        self.device = torch.device("cpu")
        if torch.cuda.is_available():  
          self.device = torch.device("cuda:0")
        indices = np.arange(len(self.users))
        np.random.shuffle(indices)
        self.users, self.items, self.times, self.labels = self.users[indices], self.items[indices], self.times[indices], self.labels[indices]
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        for i in range(len(self.users)):
            user_counts[self.users[i]] += 1
            item_counts[self.items[i]] += 1

    def __len__(self):
        return int(len(self.users)/self.batch_size)
  
    def get_batch(self, idx):
        u = self.users[idx:idx+self.batch_size].astype("int32")
        i = self.items[idx:idx+self.batch_size].astype("int32")
        # print(u.shape,i.shape)
        user_row = self.sparse_ratings[u,:].toarray()
        # print(user_row.shape)
        user_row = user_row/(np.count_nonzero(user_row,axis=1)[:,None])/10000.0*(self.n_items)
        # print(user_row.shape)
        item_col = self.sparse_ratings[:,i].toarray().T
        item_col = item_col/(np.count_nonzero(item_col,axis=1)[:,None])/10000.0*(self.n_users)
        # print(item_col.shape)
        user__time_row = self.sparse_times[u,:].toarray()
        # print(user__time_row.shape)
        for k in range(len(u)):
            user_row[k,i[k]],user__time_row[k,i[k]],item_col[k,u[k]] = 0,0,0 #remove labels from training
        user_row = torch.from_numpy(user_row).to(self.device)
        item_col = torch.from_numpy(item_col).to(self.device)
        user__time_row = torch.from_numpy(user__time_row).to(self.device)
        label = torch.tensor(self.labels[idx:idx+self.batch_size])
        if self.__len__ == idx:
            indices = np.arange(len(self.users))
            np.random.shuffle(indices)
            self.users, self.items, self.times, self.labels = self.users[indices], self.items[indices], self.times[indices], self.labels[indices]
        return user_row,item_col, user__time_row,label

    def mask(self):
        return self.sparse_ratings[:,[1,1,1]].toarray().T
    
    def get_dataset(self,data):
        return data[:,0], data[:,1], data[:,3],data[:,2]

    
    