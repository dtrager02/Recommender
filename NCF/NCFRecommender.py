#sbatch -N1 -n1 --gpus=1 --mem-per-gpu=8192 --ntasks=1 --cpus-per-task=16  --constraint=g start.sub
#sacct  --format="JobID,Elapsed,CPUTime,MaxRSS,AveRSS"
#tail -f slurm-146258.out

"""
Notes:
The final model should incorporate a hybrid of MF output and content-user matching
The proportions of these two metrics is determined by how many items the user has rated
the content user matching system will include:
1. Genres of items vs. user genres
2. release dates of items vs typical "era" of user
3. popularity of user-rated items (how niche the user is)
"""
import multiprocessing
import sys
import numpy as np
from time import perf_counter
import Recommender

class ExplicitNCF(Recommender):
    def __init__(self,  
                 alpha=0.006, 
                 beta1=.05,
                 beta2=.015):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.test_grid = np.zeros((9,9))
        self.previous_mse = 0  
        self.timestamp = perf_counter() 
        
        

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    data = np.load("../movielense_27.npy")
    drop_indices = np.random.choice(data.shape[0],size=int(data.shape[0]/10),replace=False)
    test_samples = data[drop_indices,:]
    train_samples = np.delete(data,drop_indices,axis=0)
    train_dataloader = DataLoader(train_samples, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_samples, batch_size=64, shuffle=True)
    