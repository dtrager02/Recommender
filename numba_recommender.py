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

spec = [
    ('value', numba.int32),
    ('value', numba.float32),
    ('value', numba.float32)
]

#@numba.experimental.jitclass(spec)
 
def link_db(path):
    con = sqlite3.connect(path)
    return con
    
def renumber_column(df:pd.Series,column:str):
    all_values = df[column].unique()
    all_values.sort()
    #print(all_values)
    converter = dict(zip(all_values,range(all_values.shape[0])))
    df[column] = df[column].map(converter)
    #print(sorted(df[column]))
    return df

def load_samples(n):
    try:
        con = link_db("./animeDB2.sqlite3")
        start = perf_counter()
        samples = pd.read_sql(f"select username,anime_id,score from user_records order by username limit {n}",con)
        print(f"Done loading samples in {perf_counter()-start} s.")
        samples = renumber_column(samples,"username")
        samples = renumber_column(samples,"anime_id")
        #np.random.shuffle(samples.values)
        n_users = samples["username"].max() + 1
        n_items = samples["anime_id"].max() + 1
        drop_indices = np.random.choice(samples.index,size=int(samples.shape[0]/10),replace=False)
        test_samples = samples.iloc[drop_indices,:]
        samples = samples[~samples.index.isin(drop_indices)]
        #train_samples = samples.iloc[0:int(samples.shape[0]*.8),:]
        print(f"# of train samples: {samples.shape[0]}, # of test samples: {test_samples.shape[0]}")
        # ratings = sparse.coo_matrix((samples["score"], (samples["username"], samples["anime_id"])))tocsc().astype("float32")
        # sparse.save_npz('sparse_matrix.npz', ratings)
        # start = perf_counter()
        # ratings = sparse.load_npz("sparse_matrix.npz")
        
        ratings = csr.CSR.from_coo(samples["username"].to_numpy(), samples["anime_id"].to_numpy(),samples["score"].to_numpy())
        
        print(f"Done loading samples from npz file in {perf_counter()-start} s.")
        return ratings, samples, test_samples, n_users, n_items
        #test_samples = samples.iloc[int(samples.shape[0]*.8):,:]
        #print(f"# of train samples:{train_samples.shape[0]}\n# of test samples:{test_samples.shape[0]}")
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
@numba.njit(cache=True)
def train(n_users,n_items,n_iter=30):
    """ Train model for n_iter iterations from scratch."""
    # initialize latent vectors
    user_vecs = np.random.random((n_users, n_factors)).astype("float64")
    item_vecs = np.random.random((n_items, n_factors)).astype("float64")
    
    partial_train(n_iter,user_vecs,item_vecs)
@numba.njit(cache=True)
def partial_train(n_iter,user_vecs,item_vecs):
    """ 
    Train model for n_iter iterations. Can be 
    called multiple times for further training.
    """
    ctr = 1
    while ctr <= n_iter:
        # if ctr % 10 == 0 and _v:
        #     print(f'\tcurrent iteration: {ctr}')
        user_vecs = als_step(user_vecs, 
                                       item_vecs, 
                                       ratings, 
                                       user_reg,
                                       type='user')
        item_vecs = als_step(item_vecs, 
                                       user_vecs, 
                                       ratings, 
                                       item_reg,
                                       type='item')
        ctr += 1
        if ctr % 2:
            print("Iteration:"+str(ctr)+"; train error = "+ str(int(mse(samples,user_vecs,item_vecs)*100)))
            #print("Threading layer chosen: "+numba.threading_layer())
    print("Test error = ",str(int(mse(test_samples,user_vecs,item_vecs)*100)))

@numba.njit(cache=True,parallel=True,fastmath=True)
def mse(samples,user_vecs,item_vecs): # samples format : user,item,rating
    test_errors = np.zeros(samples.shape[0])
    for i in numba.prange(samples.shape[0]):
        user = samples[i][0]
        item = samples[i][1]
        prediction = user_vecs[user, :].dot(item_vecs[item, :].T)
        test_errors[i] = samples[i][2] - prediction
    return np.sqrt(np.sum(np.square(test_errors))/test_errors.shape[0])

if __name__ == "__main__":
    numba.warnings.simplefilter('ignore', category=numba.errors.NumbaDeprecationWarning)
    numba.warnings.simplefilter('ignore', category=numba.errors.NumbaPendingDeprecationWarning)
    n_factors=20
    user_reg=0.05
    item_reg=0.05
    ratings, samples, test_samples, n_users, n_items = load_samples(10**5)
    samples,test_samples = samples.to_numpy(),test_samples.to_numpy()
    print("Data loading and preprocessing done")
    train(n_users,n_items)