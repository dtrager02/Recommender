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
from cgi import print_environ
import multiprocessing
import sys
import numpy as np
import scipy.sparse as sparse
from time import perf_counter
import pandas as pd
import numba
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
        # Initializing the bias terms
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        self.b = np.mean(self.samples[:,2])
        
        #all_groups = self.generate_indpendent_samples()
        
        # Stochastic gradient descent for given number of iterations
        if not multithreaded:
            previous_mse = 0
            for i in range(1,iters+1):
                #np.random.shuffle(self.samples)
                sgd_time = perf_counter()
                self.P,self.Q,self.y,self.b_u,self.b_i = self.sgd(self.P,self.Q,self.b_u,self.b_i,self.b,self.y,self.samples,self.   ratings,self.alpha,self.beta1,self.beta2)
                print(f"SGD time: {perf_counter()-sgd_time}")
                if i % 2:
                    train_mse = self.mse(self.samples,self.ratings,self.P,self.Q,self.b_u,self.b_i,self.b,self.y)
                    if train_mse > previous_mse and previous_mse:
                        self.alpha*=.5
                    else:
                        self.alpha*=1.05
                    print(f"Changed alpha to {self.alpha}")
                    previous_mse = train_mse - .0001
                    print("Iteration: %d ; train error = %.4f" % (i,train_mse))
            print("Test error = %.4f" % (self.mse(self.test_samples,self.ratings,self.P,self.Q,self.b_u,self.b_i,self.b,self.y)))

    # Stochastic gradient descent to get optimized P and Q matrix
    @staticmethod
    @numba.njit(cache=True,fastmath=True)
    def sgd(P,Q,b_u,b_i,b,y,
            samples,ratings:csr.CSR,
            alpha,beta1,beta2):
        for i in range(samples.shape[0]):
            user = samples[i,0]
            item = samples[i,1]
            rated_items = ratings.row_cs(user)
            R_u = np.sqrt(rated_items.shape[0])+1 #temporary division by 0 fix
            y_sum = np.sum(y[rated_items,:],axis=0)
            prediction = Q[item,:].dot((P[user, :]+y_sum/R_u).T) + b_u[user] + b_i[item] + b
            e = (samples[i,2]-prediction)
            
            b_u[user] += alpha * (e - beta1 * b_u[user])
            b_i[item] += alpha * (e - beta1 * b_i[item])
            
            P[user, :] +=alpha * (e *Q[item, :] - beta2 * P[user,:])
            Q[item, :] +=alpha * (e *P[user, :] - beta2 * Q[item,:])
  
            y[rated_items,:] += alpha*(-1.0*beta2*y[rated_items,:]+e/R_u*Q[item,:])
            
        return P,Q,y,b_u,b_i
    @staticmethod
    @numba.njit(cache=True,fastmath=True)
    def mse(samples,ratings,P,Q,b_u,b_i,b,y,exact=False): # samples format : user,item,rating
        if not exact:
            dropsize = int(samples.shape[0]/10) if samples.shape[0]<100000 else 100000
            drop_indices = np.random.choice(samples.shape[0],size=dropsize,replace=False)
        samples = samples[drop_indices,:]
        test_errors = np.zeros(samples.shape[0])
        size = 1000
        for i in range(0,samples.shape[0],size):
            users = samples[i:i+size,0]
            items = samples[i:i+size,1]
            rated_items = []
            R_u = np.empty(shape=len(users))
            y_sum = np.empty(shape=(items.shape[0],P.shape[1]))
            for j in range(len(users)):
                rated_items.append(ratings.row_cs(users[j])) #size x n
                R_u[j] = np.sqrt(rated_items[j].shape[0])+1 #temporary division by 0 fix
                y_sum[j] = np.sum(y[rated_items[j],:],axis=0)
            R_u = R_u.reshape((R_u.shape[0],1))
            predictions = np.sum(Q[items,:]*(P[users, :]+0*(y_sum/R_u)),axis=1) + b_u[users] + b_i[items] + b #temp ignore y 
            test_errors[i:i+size] = samples[i:i+size,2] - predictions
        return np.sqrt(np.sum(np.square(test_errors))/test_errors.shape[0])
    
    def predict(row,col,user_vecs,item_vecs):
        return user_vecs[row, :].dot(item_vecs[col,:].T) 
    
    """
    Execute upon receiving modified subsample
    """
    def update_params(self,subsample:SubSample):
        row_range = self.row_ranges[subsample.block_pos[0]]
        col_range = self.col_ranges[subsample.block_pos[1]]
        self.P[row_range[0]:row_range[1]] = subsample.P
        self.Q[col_range[0]:col_range[1]] = subsample.Q
        self.b_u[row_range[0]:row_range[1]] = subsample.b_u
        self.b_i[col_range[0]:col_range[1]] = subsample.b_i
        #self.y[col_range[0]:col_range[1]] = subsample.y[col_range[0]:col_range[1]]#temporary
        return subsample
    
    """
    Execute to create a subsample for a grid block
    """
    def make_subsample(self,block_pos):
        row_range = self.row_ranges[block_pos[0]]
        col_range = self.col_ranges[block_pos[1]]
        subsample = SubSample(block_pos,
                              (row_range[0],col_range[0]),
                              self.P[row_range[0]:row_range[1]],
                              self.Q[col_range[0]:col_range[1]],
                              self.b_u[row_range[0]:row_range[1]],
                              self.b_i[col_range[0]:col_range[1]],
                              self.b,
                              self.y, #[col_range[0]:col_range[1]] temporarily removed
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
        self.samples = self.samples[self.samples[:, 0].argsort()] #sort samples after the grid creation in hopes of improving cache locality
        self.test_samples = self.test_samples[self.test_samples[:, 0].argsort()]
        print("FPSG Grid Sample Shapes:")
        for i in all_groups:
            print([s.shape for s in i])
        return all_groups
    
    def random_renumber_samples(self):
        col_index = np.arange(self.n_items)
        row_index = np.arange(self.n_users)
        np.random.shuffle(col_index)
        np.random.shuffle(row_index)
        
        self.col_converter = dict(zip([*range(self.n_items)],col_index))
        self.row_converter = dict(zip([*range(self.n_users)],row_index))
        
        self.samples[:,0] = pd.Series(self.samples[:,0]).map(self.row_converter).to_numpy()
        self.samples[:,1] = pd.Series(self.samples[:,1]).map(self.col_converter).to_numpy()
        
        self.ratings = csr.CSR.from_coo(self.samples[:,0], self.samples[:,1],self.samples[:,2])
        
    def unrandomize_samples(self):
        self.col_converter = dict(zip(self.col_converter.values(),self.col_converter.keys()))
        self.row_converter =  dict(zip(self.row_converter.values(),self.row_converter.keys()))
        
        self.samples[:,0] = pd.Series(self.samples[:,0]).map(self.row_converter).to_numpy()
        self.samples[:,1] = pd.Series(self.samples[:,1]).map(self.col_converter).to_numpy()
        
        self.ratings = csr.CSR.from_coo(self.samples[:,0], self.samples[:,1],self.samples[:,2])
        
    @numba.njit(cache=True)
    def add_users_to_sparse(user_data,ratings:csr.CSR):
        # print(self.ratings.colinds)
        # print(self.ratings.values[77:100])
        values_ = user_data[:,2]
        col_idx = user_data[:,1]
        offset = ratings.nnz
        rowptrs = np.concatenate((ratings.rowptrs,np.array([offset+values_.shape[0]])))
        colinds = np.concatenate((ratings.colinds,col_idx))
        values = np.concatenate((ratings.values,values_))
        print(colinds.shape,values.shape)
        nrows = ratings.nrows+1
        ncols = np.unique(colinds).shape[0]
        nnz = ratings.nnz+values.shape[0]
        ratings_new = csr.create(nrows, ncols, nnz, rowptrs, colinds, values)
        print(ratings_new.row_cs(ratings_new.nrows-1))
        print(ratings_new.row_vs(ratings_new.nrows-1))
        #print(ratings_new.row(ratings_new.nrows-1).nonzero())
        return ratings_new, ratings
    @numba.njit(cache=True)
    def update_existing_sparse_ratings(user_data,ratings:csr.CSR):
        values = ratings.values.astype(numba.int64)
        colinds = ratings.colinds.astype(numba.int64)
        rowptrs = ratings.rowptrs
        for i in range(user_data.shape[0]):
            value = user_data[i,2]
            col_idx = user_data[i,1]
            user = user_data[i,0]
            
            colinds=  np.concatenate((colinds[:rowptrs[user]],
                                      np.array([col_idx]),
                                      colinds[rowptrs[user]:]))
            values =  np.concatenate((values[:rowptrs[user]],
                                      np.array([value]),
                                      values[rowptrs[user]:]))
            rowptrs[user+1:] += 1
        nrows = ratings.nrows
        ncols = np.max(colinds)+1
        nnz = colinds.shape[0]
        ratings_new = csr.create(nrows, ncols, nnz, rowptrs, colinds, values)
        return ratings_new, ratings
    
    def save_factor(self,factor,base_name):
        path = os.path.join("/factors",f"{base_name},n_factors={self.n_factors},item_reg={self.beta1},user_reg={self.beta2}",".npy")
        np.save(path,factor)
        
    def save_all_factors(self):
        try:
            self.save_factor(self.P,"user_factor_movielense")
            self.save_factor(self.Q,"item_factor_movielense")
        except:
            print("Factors were not initialized")
    def expirimental_setter(self,x,y):
        self.test_grid[x,y] += 1
        
    def get_ratings(self):
        return self.ratings
    
    def geterrors(self,update_num):
        print(f"Full grid update took {perf_counter()-self.timestamp} s.")
        self.timestamp = perf_counter()
        start = perf_counter()
        print("Starting error calculation threads")
        train_mse = self.mse(self.samples,self.ratings,self.P,self.Q,self.b_u,self.b_i,self.b,self.y)
        test_mse = self.mse(self.test_samples,self.ratings,self.P,self.Q,self.b_u,self.b_i,self.b,self.y)
        print("Iteration: %d ; train error = %.4f ; test error = %.4f ; lr = %.4f" % (update_num,train_mse,test_mse,self.alpha))
        print(f"Calculated error in {perf_counter()-start} s.")
        return train_mse
    
    def increment(self):
        self.update_counter+= 1
        update_number = self.update_counter/((self.n_threads+1)**2)
        if self.update_counter % (((self.n_threads+1)**2 )*1)  == 0:
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
    
    MF_ALS.load_samples_from_npy("./movielense_27.npy",100000)
    MF_ALS.train(1,multithreaded=True)
    #grid = MF_ALS.generate_indpendent_samples()
    print(MF_ALS.samples.shape)
    start = perf_counter()
    print(MF_ALS.mse(MF_ALS.samples,MF_ALS.ratings,MF_ALS.P,MF_ALS.Q,MF_ALS.b_u,MF_ALS.b_i,MF_ALS.b,MF_ALS.y))
    print(perf_counter()-start)