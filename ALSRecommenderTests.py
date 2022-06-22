import pandas as pd
import numpy as np
from ALSRecommender import *
from numba import config
from time import perf_counter
"""
    Test for updating the sparse matrix using update_existing_sparse_ratings()
    """
def update_test():
    d = {'username': [70]*100, 'anime_id': np.random.choice(range(20000),size=100,replace=False),"score":[5,8,7,3,9,4,6,7,  1,1]*10}
    df = pd.DataFrame(data=d)
    start = perf_counter()
    ExplicitMF.update_existing_sparse_ratings(df.to_numpy(),MF_ALS.ratings)
    MF_ALS.ratings.__setattr__("rowptrs",np.arange(1000)) 
    print(perf_counter()-start)
def transpose_test():
    MF_ALS = ExplicitMF(n_factors=4, user_reg=.05, item_reg=.05)
    print(f"Using hyperparams: n_factors={MF_ALS.n_factors},item_reg={MF_ALS.item_reg},user_reg={MF_ALS.user_reg}")
    MF_ALS.load_samples_from_npy("./movielense_27.npy",50000)
    print("done")
#transpose_test()
a= np.arange(100).reshape(5,20)
# print(np.sum(a,axis=0))
# print(numba.config.NUMBA_DEFAULT_NUM_THREADS)
# print(list(range(1,100,35)))
config.THREADING_LAYER = 'default'
@numba.njit(parallel=True)
def func():
    b = np.zeros((100,100000))
    a = np.ones((100,100000))
    sum = 0
    for i in numba.prange(100):
        for j in range(100000):
            sum+= a[i,j]
            b[i,j] = a[i,j]
    return sum, np.sum(b), 1
print(func())
print(numba.threading_layer())