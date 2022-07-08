import sys
from numpy.linalg import solve
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from time import perf_counter
import sqlite3
import pandas as pd
import csr

class Recommender:
    def __init__(self) -> None:
        pass
    
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
        start = perf_counter()
        self.samples = pd.read_sql(f"select username,anime_id,score from user_records order by username limit {n}",self.con)
        print(f"Done loading samples in {perf_counter()-start} s.")
        
        self.samples = Recommender.renumber_column(self.samples,"username")
        self.samples = Recommender.renumber_column(self.samples,"anime_id")
        #np.random.shuffle(self.samples.values)
        self.n_users = self.samples["username"].max() + 1
        self.n_items = self.samples["anime_id"].max() + 1
        drop_indices = np.random.choice(self.samples.index,size=int(self.samples.shape[0]/10),replace=False)
        self.test_samples = self.samples.iloc[drop_indices,:]
        self.samples = self.samples[~self.samples.index.isin(drop_indices)]
        #self.train_samples = self.samples.iloc[0:int(self.samples.shape[0]*.8),:]
        print(f"# of train samples: {self.samples.shape[0]}, # of test samples: {self.test_samples.shape[0]}")
        # self.ratings = sparse.coo_matrix((self.samples["score"], (self.samples["username"], self.samples["anime_id"]))).tocsc(.astype("float32")
        # sparse.save_npz('sparse_matrix.npz', self.ratings)
        # start = perf_counter()
        # self.ratings = sparse.load_npz("sparse_matrix.npz")
        
        self.ratings = csr.CSR.from_coo(self.samples["username"].to_numpy(), self.samples["anime_id"].to_numpy(),self.sample["score"].to_numpy())
        print(f"Done loading samples from npz file in {perf_counter()-start} s.")
        
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