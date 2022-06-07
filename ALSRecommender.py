#sbatch -N1 -n1 --gpus=1 --mem-per-gpu=8192 --ntasks=1 --cpus-per-task=16  --constraint=g start.sub
#sacct  --format="JobID,Elapsed,CPUTime,MaxRSS,AveRSS"
#tail -f slurm-146258.out
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
class ExplicitMF():
    def __init__(self,  
                 n_factors=40, 
                 item_reg=0.0, 
                 user_reg=0.0,
                 verbose=False):
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
        self.item_reg = item_reg
        self.user_reg = user_reg
        self._v = verbose
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
    
    def load_samples(self,n):
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
            #print(self.ratings.rowptrs)
            print(f"Done loading samples from npz file in {perf_counter()-start} s.")
            #self.test_samples = self.samples.iloc[int(self.samples.shape[0]*.8):,:]
            #print(f"# of train samples:{self.train_samples.shape[0]}\n# of test samples:{self.test_samples.shape[0]}")
        except (Exception) as e:
            print("No database loaded\n")
            print(e.with_traceback())
    #numba.float32[:,:](numba.float32[:,:],numba.float32[:,:],numba.float32[:,:],numba.float32,numba.typeof('a'))      
    @numba.njit(cache=True,parallel=True,fastmath=True)
    def als_step(
                 latent_vectors,
                 fixed_vecs,
                 ratings : csr.CSR,
                 _lambda,
                 type):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        temp_fixed_vecs = fixed_vecs.copy()
        lambdaI = np.eye(temp_fixed_vecs.shape[1]) * _lambda
        if type == 'user':
            # Precompute
            for u in numba.prange(0,latent_vectors.shape[0]):
                nonzero_items = ratings.row_cs(u)
                fv = temp_fixed_vecs[nonzero_items,:]

                YTY = (fv.T).dot(fv)
                A = YTY + lambdaI*(fv.shape[0]+1)
                b = ratings.row(u)[nonzero_items].dot(fv)

                latent_vectors[u, :] = solve(A, b)
        elif type == 'item':
            ratings_T = ratings.transpose()
            for i in numba.prange(latent_vectors.shape[0]):
                nonzero_items = ratings_T.row_cs(i)
                #print(ratings[0,:].toarray().reshape((-1,)).shape,fixed_vecs.shape,ratings.shape)
                #nonzero_items = np.nonzero(ratings[:,i].toarray().reshape((-1,)))[0]
                fv = temp_fixed_vecs[nonzero_items,:]
                #print(nonzero_items.shape,temp_fixed_vecs.shape)
                XTX = (fv.T).dot(fv)
                A = XTX + lambdaI*(fv.shape[0]+1)
                b = ratings_T.row(i)[nonzero_items].dot(fv) #(1xm)(mxd)
                #print(ratings_T[i,nonzero_items[row_index]].shape,b.shape)
                latent_vectors[i, :] = solve(A,b)
        return latent_vectors

    def train(self, n_iter=30):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors
        self.user_vecs = np.random.random((self.n_users, self.n_factors)).astype("float64")
        self.item_vecs = np.random.random((self.n_items, self.n_factors)).astype("float64")
        
        self.partial_train(n_iter)
    
    def partial_train(self, n_iter):
        """ 
        Train model for n_iter iterations. Can be 
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            # if ctr % 10 == 0 and self._v:
            #     print(f'\tcurrent iteration: {ctr}')
            self.user_vecs = ExplicitMF.als_step(self.user_vecs, 
                                           self.item_vecs, 
                                           self.ratings, 
                                           self.user_reg, 
                                           type='user')
            self.item_vecs = ExplicitMF.als_step(self.item_vecs, 
                                           self.user_vecs, 
                                           self.ratings, 
                                           self.item_reg, 
                                           type='item')
            ctr += 1
            if ctr % 2:
                print("Iteration: %d ; train error = %.4f" % (ctr, ExplicitMF.mse(self.samples.to_numpy(),self.user_vecs,self.item_vecs)))
                print("Threading layer chosen: %s" % numba.threading_layer())
        print("Test error = %.4f" % (ExplicitMF.mse(self.test_samples.to_numpy(),self.user_vecs,self.item_vecs)))
    @numba.njit(cache=True,parallel=True,fastmath=True)
    def mse(samples,user_vecs,item_vecs): # samples format : user,item,rating
        test_errors = np.zeros(samples.shape[0])
        for i in numba.prange(samples.shape[0]):
            user = samples[i][0]
            item = samples[i][1]
            prediction = user_vecs[user, :].dot(item_vecs[item, :].T)
            test_errors[i] = samples[i][2] - prediction
        return np.sqrt(np.sum(np.square(test_errors))/test_errors.shape[0])
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
        values = np.copy(ratings.values).astype(numba.int64)
        colinds = np.copy(ratings.colinds).astype(numba.int64)
        rowptrs = np.copy(ratings.rowptrs).astype(numba.int64)
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
    
    
    def predict(self, u, i):
        """ Single user and item prediction. """
        return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
    

if __name__ == "__main__":
    numba.warnings.simplefilter('ignore', category=numba.errors.NumbaDeprecationWarning)
    numba.warnings.simplefilter('ignore', category=numba.errors.NumbaPendingDeprecationWarning)
    MF_ALS = ExplicitMF(n_factors=80, user_reg=0.05, item_reg=0.05)
    MF_ALS.link_db("./animeDB2.sqlite3")
    MF_ALS.load_samples(10**5)
    d = {'username': [70]*10000, 'anime_id': np.random.choice(range(20000),size=10000,replace=False),"score":[5,8,7,3,9,4,6,7,1,1]*1000}
    df = pd.DataFrame(data=d)
    start = perf_counter()
    ExplicitMF.update_existing_sparse_ratings(df.to_numpy(),MF_ALS.ratings)
    print(perf_counter()-start)
    #MF_ALS.train()