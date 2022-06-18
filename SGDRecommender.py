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
import sys
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from time import perf_counter
import sqlite3
import pandas as pd
import numba
import csr 
import os.path
class ExplicitMF():
    def __init__(self,  
                 n_factors=40, 
                 alpha=0.005, 
                 beta1=.05,
                 beta2=.015):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings
        
        n_factors : (int)
            Number of latent factors to use in matrix 
            factorization model
        
        item_reg : (float)
            Regularization term for item latent factors
        
        user_reg : (float)
            Regularization term for user latent factors
        
        verbose : (bool)
            Whether or not to printout training progress
        """
        
        self.n_factors = n_factors
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
    def link_db(self,path):
        self.con = sqlite3.connect(path)
        
    def renumber_column(df:pd.Series,column:str):
        all_values = df[column].unique()
        all_values.sort()
        #print(all_values)
        converter = dict(zip(all_values,range(all_values.shape[0])))
        df[column] = df[column].map(converter)
        #print(sorted(df[column]))
        return df
    
    def load_samples_from_sql(self,n):
        try:
            start = perf_counter()
            self.samples = pd.read_sql(f"select username,anime_id,score from user_records order by username limit {n}",self.con)
            print(f"Done loading samples in {perf_counter()-start} s.")
            
            self.samples = ExplicitMF.renumber_column(self.samples,"username")
            self.samples = ExplicitMF.renumber_column(self.samples,"anime_id")
            #np.random.shuffle(self.samples.values)
            self.n_users = self.samples["username"].max() + 1
            self.n_items = self.samples["anime_id"].max() + 1
            drop_indices = np.random.choice(self.samples.index,size=int(self.samples.shape[0]/10),replace=False)
            self.test_samples = self.samples.iloc[drop_indices,:]
            self.samples = self.samples[~self.samples.index.isin(drop_indices)]
            #self.train_samples = self.samples.iloc[0:int(self.samples.shape[0]*.8),:]
            print(f"# of train samples: {self.samples.shape[0]}, # of test samples: {self.test_samples.shape[0]}")
            # self.ratings = sparse.coo_matrix((self.samples["score"], (self.samples["username"], self.samples["anime_id"]))).tocsc().astype("float32")
            # sparse.save_npz('sparse_matrix.npz', self.ratings)
            # start = perf_counter()
            # self.ratings = sparse.load_npz("sparse_matrix.npz")
            
            self.ratings = csr.CSR.from_coo(self.samples["username"].to_numpy(), self.samples["anime_id"].to_numpy(),self.samples["score"].to_numpy())
            print(f"Done loading samples from npz file in {perf_counter()-start} s.")

        except (Exception) as e:
            print("No database loaded\n")
            print(e.with_traceback())
    #numba.float32[:,:](numba.float32[:,:],numba.float32[:,:],numba.float32[:,:],numba.float32,numba.typeof('a')) 
    
    def load_samples_from_npy(self,path,n):
        start = perf_counter()
        a = np.load(path)
        a[:,2] *= 2 #adjusts ratings on a 5 pt scale to ints on a 10 pt scale
        a = a.astype(np.int32) #makes it convertible to CSR
        if n != "all":
            a = a[:n] 
        self.n_users = a[:,0].max() + 1
        self.n_items = a[:,1].max() + 1
        drop_indices = np.random.choice(a.shape[0],size=int(a.shape[0]/10),replace=False)
        self.test_samples = a[drop_indices,:]
        self.samples = np.delete(a,drop_indices,axis=0)

        print(f"# of train samples: {self.samples.shape[0]}, # of test samples: {self.test_samples.shape[0]}")
        self.ratings = csr.CSR.from_coo(self.samples[:,0], self.samples[:,1],self.samples[:,2])
        print(f"Done loading samples from npz file in {perf_counter()-start} s.")
        
    #@numba.njit(cache=True,parallel=True,fastmath=True)
    # Initializing user-feature and movie-feature matrix 
    def train(self,iters):
        self.P = np.random.normal(scale=1./self.n_factors, size=(self.n_users, self.n_factors)) #users
        self.Q = np.random.normal(scale=1./self.n_factors, size=(self.n_items, self.n_factors)) # items
        self.y = np.random.normal(scale=1./self.n_factors, size=(self.n_items, self.n_factors)) # implicit items
        # Initializing the bias terms
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        self.b = np.mean(self.samples[:,2])
        

        # Stochastic gradient descent for given number of iterations
        previous_mse = 0
        for i in range(iters):
            np.random.shuffle(self.samples)
            self.P,self.Q,self.y,self.b_u,self.b_i = ExplicitMF.sgd(self.P,self.Q,self.b_u,self.b_i,self.b,self.y,self.samples,self.ratings,self.alpha,self.beta1,self.beta2)
            
            if i % 2:
                train_mse = ExplicitMF.mse(self.samples,self.P,self.Q,self.b_u,self.b_i,self.b)
                if train_mse > previous_mse and previous_mse:
                    self.alpha*=.5
                else:
                    self.alpha*=1.05
                print(f"Changed alpha to {self.alpha}")
                previous_mse = train_mse - .0001
                print("Iteration: %d ; train error = %.4f" % (i,train_mse))
        print("Test error = %.4f" % (ExplicitMF.mse(self.test_samples,self.P,self.Q,self.b_u,self.b_i,self.b)))

    # Stochastic gradient descent to get optimized P and Q matrix
    @numba.njit(cache=True,fastmath=True,parallel=True)
    def sgd(P,Q,b_u,b_i,b,y,
            samples,ratings:csr.CSR,
            alpha,beta1,beta2):
        for i in range(samples.shape[0]):
            user = samples[i,0]
            item = samples[i,1]
            rated_items = ratings.row_cs(user)
            R_u = np.sqrt(rated_items.shape[0])
            y_sum = np.sum(y[rated_items,:],axis=0)
            prediction = Q[item,:].dot((P[user, :]+y_sum/R_u).T) + b_u[user] + b_i[item] + b
            e = (samples[i,2]-prediction)
            
            b_u[user] += alpha * (e - beta1 * b_u[user])
            b_i[item] += alpha * (e - beta1 * b_i[item])
            
            P[user, :] +=alpha * (e *Q[item, :] - beta2 * P[user,:])
            Q[item, :] +=alpha * (e *P[user, :] - beta2 * Q[item,:])
  
            y[rated_items,:] += alpha*(e/R_u*Q[rated_items,:]-beta2*y[rated_items,:])
            
        return P,Q,y,b_u,b_i

    @numba.njit(cache=True,parallel=True,fastmath=True)
    def mse(samples,P,Q,b_u,b_i,b): # samples format : user,item,rating
        test_errors = np.zeros(samples.shape[0])
        for i in numba.prange(samples.shape[0]):
            user = samples[i][0]
            item = samples[i][1]
            prediction = prediction = P[user, :].dot(Q[item,:].T) + b_u[user] + b_i[item] + b
            test_errors[i] = samples[i][2] - prediction
        return np.sqrt(np.sum(np.square(test_errors))/test_errors.shape[0])
    
    def predict(row,col,user_vecs,item_vecs):
        return user_vecs[row, :].dot(item_vecs[col,:].T)
    
    def make_independent_chunks(self):
        chunk_size = numba.config.NUMBA_DEFAULT_NUM_THREADS
        breakpoint = int(self.ratings.nnz/(chunk_size))+chunk_size
        row_ranges,col_ranges = [],[]
        offset = 0
        previous_index = 0
        for i in range(len(self.ratings.rowptrs)):
            if self.ratings.rowptrs[i] -offset > breakpoint:
                row_ranges.append((previous_index,i))
                previous_index = i
                offset+=breakpoint
        row_ranges.append((previous_index,len(self.ratings.rowptrs)))
        col_breakpoints = list(range(0,self.n_items,int(self.n_items/chunk_size)))
        for i in range(len(col_breakpoints)-1):
            col_ranges.append((col_breakpoints[i],col_breakpoints[i+1]))
        col_ranges.append((col_breakpoints[-1],col_breakpoints[-1]+int(self.n_items/chunk_size)))   
        return row_ranges,col_ranges
                
    
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
        path = os.path.join("/factors",f"{base_name},n_factors={self.n_factors},item_reg={self.item_reg},user_reg={self.user_reg}",".npy")
        np.save(path,factor)
        
    def save_all_factors(self):
        try:
            self.save_factor(self.user_vecs,"user_factor_movielense")
            self.save_factor(self.item_vecs,"user_factor_movielense")
        except:
            print("Factors were not initialized")
    
    def predict(self, u, i):
        """ Single user and item prediction. """
        return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
    

if __name__ == "__main__":
    numba.warnings.simplefilter('ignore', category=numba.errors.NumbaDeprecationWarning)
    numba.warnings.simplefilter('ignore', category=numba.errors.NumbaPendingDeprecationWarning)
    if len(sys.argv) == 4:
        n_factors = int(sys.argv[1])
        item_reg = float(sys.argv[2])
        user_reg = float(sys.argv[3])
        MF_ALS = ExplicitMF(n_factors=n_factors, user_reg=user_reg, item_reg=item_reg)
    else:
        MF_ALS = ExplicitMF(n_factors=30)
    print(f"Using hyperparams: n_factors={MF_ALS.n_factors},alpha={MF_ALS.alpha},beta1={MF_ALS.beta1},beta2={MF_ALS.beta2}")
    
    MF_ALS.load_samples_from_npy("./movielense_27.npy",100000)
    #MF_ALS.train(80)
    print(MF_ALS.make_independent_chunks())