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
from functools import cache
import multiprocessing
import sys
import numpy as np
from time import perf_counter
import pandas as pd
import numba
from numba_methods import *
import csr 
import os.path
from Recommender import Recommender
from SubSample import SubSample
import ray

@ray.remote
class ExplicitMF(Recommender):
    def __init__(self,  
                 n_threads = multiprocessing.cpu_count(),
                 n_factors=40, 
                 alpha=0.006, 
                 beta1=.05,
                 beta2=.015):
        self.n_threads = n_threads
        self.n_factors = n_factors
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.test_grid = np.zeros((9,9))
        self.update_counter = 0
        self.previous_mse = 0  
        self.timestamp = perf_counter() 
        
        
    #@numba.njit(cache=True,,fastmath=True)
    # Initializing user-feature and movie-feature matrix 
    def train(self,iters,multithreaded=True):
        self.P = np.random.normal(scale=1./self.n_factors, size=(self.n_users, self.n_factors)) #users
        self.Q = np.random.normal(scale=1./self.n_factors, size=(self.n_items, self.n_factors)) # items
        self.y = np.random.normal(scale=1./self.n_factors, size=(self.n_items, self.n_factors)) # implicit items
        self.y_owner = False
        # Initializing the bias terms
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        self.b = np.mean(self.samples[:,2])
        row_ptrs,col_inds = self.get_rated_by_user()
        #all_groups = self.generate_indpendent_samples()
        print("items,users:",self.n_users,self.n_items)
        # Stochastic gradient descent for given number of iterations
        if not multithreaded:
            previous_mse = 0
            for i in range(1,iters+1):
                #np.random.shuffle(self.samples)
                sgd_time = perf_counter()
                self.P,self.Q,self.y,self.b_u,self.b_i = sgd(self.P,self.Q,self.b_u,self.b_i,self.b,self.y,self.samples,row_ptrs,col_inds,self.alpha,self.beta1,self.beta2)
                # self.P,self.Q,self.y,self.b_u,self.b_i = sgd2(self.P,self.Q,self.b_u,self.b_i,self.b,self.y,self.samples,row_ptrs,col_inds,self.alpha,self.beta1,self.beta2)
                print(f"SGD time: {perf_counter()-sgd_time}")
                if i % 2:
                    train_mse = mse(self.samples,self.ratings.rowptrs,self.ratings.colinds,self.P,self.Q,self.b_u,self.b_i,self.b,self.y)
                    if train_mse > previous_mse and previous_mse:
                        self.alpha*=.5
                    else:
                        self.alpha*=1.05
                    print(f"Changed alpha to {self.alpha}")
                    previous_mse = train_mse - .0001
                    print("Iteration: %d ; train error = %.4f" % (i,train_mse))
            print("Test error = %.4f" % (mse(self.test_samples,self.ratings.rowptrs,self.ratings.colinds,self.P,self.Q,self.b_u,self.b_i,self.b,self.y)))

    
    def predict(row,col,user_vecs,item_vecs):
        return user_vecs[row, :].dot(item_vecs[col,:].T) 
    
    """
    Execute upon receiving modified subsample
    """
    def update_params(self,subsample:SubSample) -> SubSample:
        block_pos = subsample.block_pos
        row_range = (self.u_bins[block_pos[0]],self.u_bins[block_pos[0]+1])
        col_range = (self.i_bins[block_pos[1]],self.i_bins[block_pos[1]+1])
        self.P[row_range[0]:row_range[1]] = subsample.P
        self.Q[col_range[0]:col_range[1]] = subsample.Q
        self.b_u[row_range[0]:row_range[1]] = subsample.b_u
        self.b_i[col_range[0]:col_range[1]] = subsample.b_i
        if subsample.y:
            self.y = subsample.y
        return subsample
    
    """
    Execute to create a subsample for a grid block
    """
    def make_subsample(self,block_pos) -> SubSample:
        if self.y_owner is False:
            y = self.y
        else:
            y = False
        row_range = (self.u_bins[block_pos[0]],self.u_bins[block_pos[0]+1])
        col_range = (self.i_bins[block_pos[1]],self.i_bins[block_pos[1]+1])
        subsample = SubSample(block_pos,
                              (row_range[0],col_range[0]),
                              self.P[row_range[0]:row_range[1]],
                              self.Q[col_range[0]:col_range[1]],
                              self.b_u[row_range[0]:row_range[1]],
                              self.b_i[col_range[0]:col_range[1]],
                              self.b,
                              y, #[col_range[0]:col_range[1]] temporarily removed
                              self.alpha,
                              self.beta1,
                              self.beta2)  
        return subsample 
    
    def get_chunk_breakpoints(self,debug=False):
        # this is imprecise but works for up to 32 threads
        chunk_size = self.n_threads
        #breakpoint = int(self.ratings.nnz/(chunk_size+1))+chunk_size
        u_breakpoint = int(self.n_users/(chunk_size+1))+1
        i_breakpoint = int(self.n_items/(chunk_size+1))+1
        u_offset, i_offset = 0, 0
        self.row_ranges,self.col_ranges = [],[]
        for _ in range(chunk_size):
            self.row_ranges.append((u_offset,u_offset+u_breakpoint))
            self.col_ranges.append((i_offset,i_offset+i_breakpoint))
            i_offset+= i_breakpoint
            u_offset+= u_breakpoint
        self.row_ranges.append((u_offset,self.n_users))
        self.col_ranges.append((i_offset,self.n_items))
        """Redundant code"""
        # self.row_ranges,self.col_ranges = [],[]
        # offset = 0
        # previous_index = 0
        # for i in range(len(self.ratings.rowptrs)):
        #     if self.ratings.rowptrs[i] -offset > breakpoint:
        #         self.row_ranges.append((previous_index,i))
        #         previous_index = i
        #         offset+=breakpoint
        # self.row_ranges.append((previous_index,len(self.ratings.rowptrs)))
        # col_breakpoints = list(range(0,self.n_items,int(self.n_items/(chunk_size+1))+chunk_size))
        # for i in range(len(col_breakpoints)-1):
        #     self.col_ranges.append((col_breakpoints[i],col_breakpoints[i+1]))
        # self.col_ranges.append((col_breakpoints[-1],col_breakpoints[-1]+int(self.n_items/chunk_size)))
        print("Shape of FPSGD grid:",len(self.row_ranges),len(self.col_ranges))
        if debug:
            print(f"n_users,n_items = {self.n_users},{self.n_items}")
            print("row ranges:",self.row_ranges)
            print("col ranges:",self.col_ranges)
        return self.row_ranges,self.col_ranges
    
    def generate_indpendent_samples(self):
        self.random_renumber_samples()
        self.get_chunk_breakpoints(debug=False) #change this for more verbosity
        all_groups = []
        for i in range(len(self.row_ranges)):
            row_groups = []
            for j in range(len(self.col_ranges)):
                """self.col_ranges,self.row_ranges are in form [(a_0,b+0),(a_1,b_1)...] where a_n,b_n represents the span of rows/cols in the nth grid block
                
                The statements below select ratings that belong to each grid block based on that block's row/col ranges, and place them into a equally sized 2d array. In other words, if grid is 9x9, then all_groups is also 9x9
                """
                row_condition = np.logical_and(self.row_ranges[i][1]> self.samples[:,0], self.samples[:,0]>= self.row_ranges[i][0])
                col_condition = np.logical_and(self.col_ranges[j][1]> self.samples[:,1], self.samples[:,1] >= self.col_ranges[j][0])
                group_values = self.samples[np.logical_and(row_condition,col_condition)]
                row_groups.append(group_values)
            all_groups.append(row_groups)
        # self.samples = self.samples[self.samples[:, 0].argsort()] #sort samples after the grid creation in hopes of improving cache locality
        # self.test_samples = self.test_samples[self.test_samples[:, 0].argsort()]
        print("Shape of FPSGD grid:",len(all_groups),len(all_groups[0]))
        print("FPSG Grid Sample Shapes:")
        c = 0
        for i in all_groups:
            for s in i:
                c += s.shape[0] 
            print([s.shape for s in i])
        print(f"Total grid samples: {c}")
        return all_groups
    
    def generate_indpendent_samples_new(self):
        self.random_renumber_samples()
        chunk_size = self.n_threads
        print(self.samples.shape)
        u_breakpoint = int(self.n_users/(chunk_size+1))+1
        i_breakpoint = int(self.n_items/(chunk_size+1))+1
        
        self.u_bins = [*range(0,self.n_users+u_breakpoint,u_breakpoint)]
        #print(self.u_bins,self.n_users)
        u_bin_inds = np.digitize(self.samples[:,0],self.u_bins)
        u_groups = [np.zeros((1,3)) for _ in range(len(self.u_bins))]
        for i in range(1,len(self.u_bins)):
            u_groups[i-1] = self.samples[u_bin_inds==i]
        if np.sum(u_groups[-1]) == 0: #gets rid of edge case where last bin catches nothing
            u_groups.pop()
        #splitting items from users
        all_groups = []
        self.i_bins = [*range(0,self.n_items+i_breakpoint,i_breakpoint)]
        for group in u_groups:
            i_groups = [np.zeros((1,3)) for _ in range(len(self.i_bins))]
            i_bin_inds = np.digitize(group[:,1],self.i_bins)
            for j in range(1,len(self.u_bins)):
                i_groups[j-1] = group[i_bin_inds==j]
            if np.sum(i_groups[-1]) == 0: #gets rid of edge case where last bin catches nothing
                i_groups.pop()
            all_groups.append(i_groups)
        #For debugging purposes
        print("Shape of FPSGD grid:",len(all_groups),len(all_groups[0]))
        print("FPSG Grid Sample Shapes:")
        c = 0
        for i in all_groups:
            for s in i:
                c += s.shape[0] 
            print([s.shape for s in i])
        print(f"Total grid samples: {c}")
        print(f"user_bins:{self.u_bins}")
        print(f"item_bins:{self.i_bins}")
        # print([u_groups[i].shape[0] for i in range(len(u_groups))])
        # print("total users:",sum([u_groups[i].shape[0] for i in range(len(u_groups))]))
        return all_groups
    
    def random_renumber_samples(self):
        col_index = np.arange(self.n_items)
        row_index = np.arange(self.n_users)
        np.random.shuffle(col_index)
        np.random.shuffle(row_index)
        
        self.col_converter = dict(zip([*range(self.n_items)],col_index))
        self.row_converter = dict(zip([*range(self.n_users)],row_index))
        
        temp_users = pd.Series(self.samples[:,0]).map(self.row_converter).to_numpy()
        temp_items = pd.Series(self.samples[:,1]).map(self.col_converter).to_numpy()
        
        sorter = temp_users.argsort()
        a = temp_items[sorter]
        b = temp_users[sorter]
        c = np.split(a, np.unique(b, return_index=True)[1][1:])
        randomized_lengths = sorted([len(s) for s in c])
        sorter = self.samples[:,0].argsort()
        a = self.samples[sorter]
        c = np.split(a[:,1], np.unique(a[:,0], return_index=True)[1][1:])
        original_lengths = sorted([len(s) for s in c])
        assert randomized_lengths == original_lengths
        assert not (0 in original_lengths)
        self.samples[:,0] = pd.Series(self.samples[:,0]).map(self.row_converter).to_numpy()
        self.samples[:,1] = pd.Series(self.samples[:,1]).map(self.col_converter).to_numpy()
        #checking it actually worked
        #fixed bug where test conversion was missing
        self.test_samples[:,0] = pd.Series(self.test_samples[:,0]).map(self.row_converter).to_numpy()
        self.test_samples[:,1] = pd.Series(self.test_samples[:,1]).map(self.col_converter).to_numpy()
        
        #remove test samples with users/items not found in train samples to prevent error calculation bugs
        drop_rows = []
        for i in range(self.test_samples.shape[0]):
            if self.test_samples[i,0] not in self.samples[:,0] or self.test_samples[i,1] not in self.samples[:,1]:
                drop_rows.append(i)
        self.test_samples = np.delete(self.test_samples,drop_rows,axis=0)
        
        self.ratings = csr.CSR.from_coo(self.samples[:,0], self.samples[:,1],self.samples[:,2])
        for i in np.unique(self.samples[:,0]):
            if (self.ratings.rowptrs[i+1] - self.ratings.rowptrs[i]) == 0:
                raise Exception("Samples are missing from ratings CSR")
        
    def unrandomize_samples(self):
        #reverse dict that originally randomized samples
        self.col_converter = dict(zip(self.col_converter.values(),self.col_converter.keys()))
        self.row_converter =  dict(zip(self.row_converter.values(),self.row_converter.keys()))
        
        #map randomized to ordered
        self.samples[:,0] = pd.Series(self.samples[:,0]).map(self.row_converter).to_numpy()
        self.samples[:,1] = pd.Series(self.samples[:,1]).map(self.col_converter).to_numpy()
        
        #rearrange factor matrices to match new user/item samples
        ordered_indices = np.array(list(self.col_converter.values()))
        for arr in (self.P,self.Q,self.y,self.b_u,self.b_i):
            arr = arr[ordered_indices]
        
        self.ratings = csr.CSR.from_coo(self.samples[:,0], self.samples[:,1],self.samples[:,2])
    
    def save_factor(self,factor,base_name):
        path = os.path.join("/factors",f"{base_name},n_factors={self.n_factors},item_reg={self.beta1},user_reg={self.beta2}",".npy")
        np.save(path,factor)
        
    def save_all_factors(self,name):
        try:
            np.save(f"./factors/{name},{self.n_factors}{self.alpha},{self.beta1},{self.beta2}",P=self.P,Q=self.Q,y=self.y,b_u=self.b_u,b_i=self.b_i)
        except:
            print("Factors were not initialized")
    def expirimental_setter(self,x,y):
        self.test_grid[x,y] += 1
        
    def get_ratings(self):
        return self.ratings
    
    def get_rated_by_user(self):
        return (self.ratings.rowptrs,self.ratings.colinds)
    
    def geterrors(self,update_num):
        print(f"Full grid update took {perf_counter()-self.timestamp} s.")
        self.timestamp = perf_counter()
        start = perf_counter()
        print("Starting error calculation threads")
        train_mse = mse2(self.samples,self.ratings.rowptrs,self.ratings.colinds,self.P,self.Q,self.b_u,self.b_i,self.b,self.y)
        test_mse = mse2(self.test_samples,self.ratings.rowptrs,self.ratings.colinds,self.P,self.Q,self.b_u,self.b_i,self.b,self.y)
        print("Iteration: %d ; train error = %.4f ; test error = %.4f ; lr = %.4f" % (update_num,train_mse,test_mse,self.alpha))
        print(f"Calculated error in {perf_counter()-start} s.")
        return train_mse
    
    def increment(self):
        self.update_counter+= 1
        update_number = self.update_counter/((self.n_threads+1)**2)
        if self.update_counter % (((self.n_threads+1)**2 )*2)  == 0:
            train_mse = self.geterrors(update_number)
            if train_mse > self.previous_mse and self.previous_mse:
                self.alpha*=.6
            else:
                self.alpha*=1.06
            self.previous_mse = train_mse - .0001
        return self.update_counter
if __name__ == "__main__":

    if len(sys.argv) == 4:
        n_factors = int(sys.argv[1])
        alpha = float(sys.argv[2])
        beta1 = float(sys.argv[3])
        beta2 = float(sys.argv[4])
        MF_ALS = ExplicitMF(n_factors=n_factors, alpha=alpha, beta1=beta1, beta2=beta2)
    else:
        MF_ALS = ExplicitMF(n_factors=40)
    print(f"Using hyperparams: n_factors={MF_ALS.n_factors},alpha={MF_ALS.alpha},beta1={MF_ALS.beta1},beta2={MF_ALS.beta2}")
    
    MF_ALS.load_samples_from_npy("./movielense_27.npy",200000)
    MF_ALS.train(40,multithreaded=False)
    #print(MF_ALS.mse(MF_ALS.samples,MF_ALS.ratings,MF_ALS.P,MF_ALS.Q,MF_ALS.b_u,MF_ALS.b_i,MF_ALS.b,MF_ALS.y))