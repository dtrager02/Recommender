import pandas as pd
import numpy as np
from ALSRecommender import *
from time import perf_counter
"""
    Test for updating the sparse matrix using update_existing_sparse_ratings()
    """
d = {'username': [70]*100, 'anime_id': np.random.choice(range(20000),size=100,replace=False),"score":[5,8,7,3,9,4,6,7,1,1]*10}
df = pd.DataFrame(data=d)
start = perf_counter()
ExplicitMF.update_existing_sparse_ratings(df.to_numpy(),MF_ALS.ratings)
MF_ALS.ratings.__setattr__("rowptrs",np.arange(1000)) 
print(perf_counter()-start)