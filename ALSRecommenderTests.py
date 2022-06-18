import pandas as pd
import numpy as np
from ALSRecommender import *
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
k = 500000

start = perf_counter()
a = np.empty(k)
for i in range(k):
    a[i]  = np.log(i+1)
end = perf_counter()
print(end-start)

start = perf_counter()
b = []
for i in range(k):
    b.append(np.log(i+1))
end = perf_counter()
print(end-start)

start = perf_counter()
c = [0]*k
for i in range(k):
    c[i]  = np.log(i+1)
end = perf_counter()
print(end-start)

start = perf_counter()
c = np.random.random(size=k)
c = np.log(c+1) 
end = perf_counter()
print(end-start)