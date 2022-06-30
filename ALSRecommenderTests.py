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
# config.THREADING_LAYER = 'default'
# @numba.njit(parallel=True)
# def func():
#     b = np.zeros((100,100000))
#     a = np.ones((100,100000))
#     sum = 0
#     for i in numba.prange(100):
#         for j in range(100000):
#             sum+= a[i,j]
#             b[i,j] = a[i,j]
#     return sum, np.sum(b), 1
# print(func())
# print(numba.threading_layer())

num = 89099
def func(num):
    s = str(num)
    value_queue = [s[0]]
    index_queue = [0]
    max_int = num
    index = 0
    for i in range(1,len(s)):
        print(value_queue,max_int)
        if int(s[i]) >= int(value_queue[0]):
            value_queue.insert(0,s[i])
            index_queue.insert(0,int(i))
        elif int(s[i]) < int(value_queue[0]):
            temp = int(s[:index]+"".join(value_queue)+s[len(value_queue)+index:])
            if temp>max_int:
                max_int = temp
                index = i
            value_queue = [s[i]]
            index_queue = [i]
        else:
            ...
    temp = int(s[:index]+"".join(value_queue)+s[len(value_queue)+index:])
    if temp>max_int:
        max_int = temp
        index = i
    return max_int,index

num = 61111116611
#num = int("".join([str(i) for i in range(10**2)]))
def func2(num):
    s = str(num)
    l = 0
    r = len(s)-1 
    max = -1
    while l != r:
        #print(int(s[:l]+s[l:r+1][::-1]+s[r+1:]),l,r,max)
        if(int(s[:l]+s[l:r+1][::-1]+s[r+1:]) > max):
            max = int(s[:l]+s[l:r+1][::-1]+s[r+1:])
        if int(s[l]) < int(s[r]):
            l+=1
        elif int(s[l]) == int(s[r]):
            #print("flag")
            temp_l,temp_r = l,r
            while int(s[temp_l]) == int(s[temp_r]):
                temp_l += 1
                temp_r -=1
                if temp_l >= temp_r:
                    return max
            if int(s[temp_l]) < int(s[temp_r]):
                r = temp_r
            else:
                l = temp_l
        else:
            r-=1
    return max
print(func2(num))


