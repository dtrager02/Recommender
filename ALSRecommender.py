#sbatch -N1 -n1 --gpus=1 --mem-per-gpu=8192 --ntasks=1 --cpus-per-task=16  --constraint=g start.sub
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

            #self.train_samples = self.samples.iloc[0:int(self.samples.shape[0]*.8),:]
            
            # self.ratings = sparse.coo_matrix((self.samples["score"], (self.samples["username"], self.samples["anime_id"]))).tocsc().astype("float32")
            # sparse.save_npz('sparse_matrix.npz', self.ratings)
            # start = perf_counter()
            # self.ratings = sparse.load_npz("sparse_matrix.npz")
            
            self.ratings = csr.CSR.from_coo(self.samples["username"], self.samples["anime_id"],self.samples["score"],rpdtype=np.float32)
             
            print(f"Done loading samples from npz file in {perf_counter()-start} s.")
            #self.test_samples = self.samples.iloc[int(self.samples.shape[0]*.8):,:]
            #print(f"# of train samples:{self.train_samples.shape[0]}\n# of test samples:{self.test_samples.shape[0]}")
        except (Exception) as e:
            print("No database loaded\n")
            print(e)
    #"(float32[:,:],float32[:,:],float32[:,:],float32,str)"        
    #@numba.jit(numba.float32[:,:](numba.float32[:,:],numba.float32[:,:],numba.float32[:,:],numba.float32,numba.typeof('a')),nopython=False,cache=True)
    def als_step(
                 latent_vectors,
                 fixed_vecs,
                 ratings,
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
            for u in range(0,latent_vectors.shape[0]):
                row_index = u%5000
                if row_index == 0:
                    nonzero_items = ratings[u:u+5000,:].tolil().rows
                #print(ratings[0,:].toarray().reshape((-1,)).shape,fixed_vecs.shape,ratings.shape)
                #nonzero_items = ratings[u,:].nonzero().reshape((-1,))[0]
                fv = temp_fixed_vecs[nonzero_items[row_index],:]
                #print(nonzero_items.shape,temp_fixed_vecs.shape)
                YTY = np.matmul(fv.T,fv)
                A = YTY + lambdaI
                b = ratings[u,nonzero_items[row_index]].dot(fv)[0]
                #print(ratings[u,nonzero_items[row_index]].shape,b.shape)
                #b = np.matmul(ratings[u, nonzero_items[row_index]].toarray().reshape((-1,)),fv)
                #print((YTY + lambdaI).shape,np.matmul(ratings[u,:].toarray()[0],fixed_vecs).shape)
                latent_vectors[u, :] = solve(A, b)
        elif type == 'item':
            ratings_T = ratings.T
            for i in range(latent_vectors.shape[0]):
                row_index = i%5000
                if row_index == 0:
                    nonzero_items = ratings_T[i:i+5000,:].tolil().rows
                #print(ratings[0,:].toarray().reshape((-1,)).shape,fixed_vecs.shape,ratings.shape)
                #nonzero_items = np.nonzero(ratings[:,i].toarray().reshape((-1,)))[0]
                fv = temp_fixed_vecs[nonzero_items[row_index],:]
                #print(nonzero_items.shape,temp_fixed_vecs.shape)
                XTX = np.matmul(fv.T,fv)
                A = XTX + lambdaI
                b = ratings_T[i,nonzero_items[row_index]].dot(fv)[0] #(1xm)(mxd)
                #print(ratings_T[i,nonzero_items[row_index]].shape,b.shape)
                latent_vectors[i, :] = solve(A,b)
        return latent_vectors

    def train(self, n_iter=5):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors
        self.user_vecs = np.random.random((self.n_users, self.n_factors)).astype("float32")
        self.item_vecs = np.random.random((self.n_items, self.n_factors)).astype("float32")
        
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
                print("Iteration: %d ; train error = %.4f" % (ctr, self.mse()))
    
    def mse(self,test=False):
        train_errors, test_errors = [],[]
        if test:
            for i, j, r in self.test_samples.values:
                prediction = self.predict(i, j)
                test_errors.append(r - prediction)
            test_errors = np.array(test_errors)
            return np.sqrt(np.sum(np.square(test_errors))/test_errors.shape[0])
        
        predictions = self.samples.apply(
            lambda x:self.user_vecs[x[0], :].dot(self.item_vecs[x[1], :].T)
            ,axis=1)
        train_errors = self.samples["score"]-predictions
        return np.sqrt(np.sum(np.square(train_errors))/train_errors.shape[0])
    
    def predict_all(self):
        """ Predict ratings for every user and item. """
        predictions = np.zeros((self.user_vecs.shape[0], 
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)
                
        return predictions
    def predict(self, u, i):
        """ Single user and item prediction. """
        return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
    
    def get_mse(pred, actual):
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual)
    def calculate_learning_curve(self, iter_array, test):
        """
        Keep track of MSE as a function of training iterations.
        
        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of 
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item).
        
        The function creates two new class attributes:
        
        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        iter_array.sort()
        self.train_mse =[]
        self.test_mse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print (f'Iteration: {n_iter}')
            if i == 0:
                self.train(n_iter - iter_diff)
            else:
                self.partial_train(n_iter - iter_diff)

            predictions = self.predict_all()

            self.train_mse += [ExplicitMF.get_mse(predictions, self.ratings)]
            self.test_mse += [ExplicitMF.get_mse(predictions, test)]
            if self._v:
                print( 'Train mse: ' + str(self.train_mse[-1]))
                print ('Test mse: ' + str(self.test_mse[-1]))
            iter_diff = n_iter

if __name__ == "__main__":
    numba.warnings.simplefilter('ignore', category=numba.errors.NumbaDeprecationWarning)
    numba.warnings.simplefilter('ignore', category=numba.errors.NumbaPendingDeprecationWarning)
    MF_ALS = ExplicitMF(n_factors=40, user_reg=0.01, item_reg=0.01)
    MF_ALS.link_db("./animeDB2.sqlite3")
    MF_ALS.load_samples(10**5)
    MF_ALS.train()