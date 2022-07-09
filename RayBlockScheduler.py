import multiprocessing
import queue
import time
import os
import random
import asyncio
import ray
import numba
import numpy as np
import csr
from multiprocessing import Process, Queue, Lock, Manager
from collections import defaultdict
from SubSample import SubSample
from SGDRecommender import ExplicitMF
        
@ray.remote
class BlockScheduler:
    def __init__(self,width,iters) -> None:
        self.width = width
        self.n_threads = self.width -1
        temp = [0]*self.width
        self.update_counter = []
        for i in range(self.width):
            self.update_counter.append(temp.copy())
        self.unused_rows,self.unused_cols=[*range(self.width)],[*range(self.width)]
        self.completed_rows,self.completed_cols=set(),set()
        self.iters  = iters
        self.test = 0
    
    def check_completion(self):
        #print(self.unused_rows,self.unused_cols)
        if len(self.completed_rows) == self.width and len(self.completed_cols)==self.width:
            return True
        else:
            return False
    
    def get_next(self,completed=None):
        # print("Unused rows",self.unused_rows,"Unused cols: ",self.unused_cols)
        # print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
        if completed:
            self.completed_chunk(completed)
        # if len(self.unused_cols) != len(self.unused_rows):
        #     raise Exception(f"Desync occurred, # of unused rows and cols is not the same\n{self.unused_rows}\n{self.unused_cols}")
        if not len(self.unused_cols):
            return None
        min = 10**10
        min_idx = (-1,-1)
        for i in self.unused_rows:
            for j in self.unused_cols:
                if self.update_counter[i][j] <min and self.update_counter[i][j] <self.iters:
                    min = self.update_counter[i][j]
                    min_idx = (i,j)
        if min_idx != (-1,-1):
            output = min_idx # min_idx is the block_pos of subsample
            row_idx = self.unused_rows.index(min_idx[0])
            col_idx = self.unused_cols.index(min_idx[1])
            self.unused_cols.pop(col_idx)
            self.unused_rows.pop(row_idx)
        else:
            output = None
        return output
            
    def completed_chunk(self,idx):
        self.test += 1
        row = idx[0]
        col = idx[1]
        if idx[0] in self.unused_rows or idx[1] in self.unused_cols:
            raise Exception("Desync occurred, completed chunk was still in unused chunks")
        self.update_counter[idx[0]][idx[1]] += 1
        r_count,c_count = 0,0
        for i in range(self.width):
            if self.update_counter[row][i] == self.iters:
                r_count += 1
            if self.update_counter[i][col] == self.iters:
                c_count += 1
        # if r_count > self.iters or c_count>self.iters:
        #     raise Exception(f"Too many iterations. updates:\n{self.update_counter}")
        
        if c_count == self.width:
            self.completed_cols.add(col)
        else:
            self.unused_cols.append(col)
            
        if r_count == self.width:
            self.completed_rows.add(row)
        
        else:
            self.unused_rows.append(row)                
        
    def get_update_counter(self):
        return self.update_counter
    
    def get_test(self):
        return self.test
    
    def __self__(self):
        return self
    
    

# The consumer function takes data off of the Queue
@ray.remote
def consumer(scheduler : BlockScheduler,trainer: ExplicitMF, groups: list,ratings: csr.CSR):
    block = ray.get(scheduler.get_next.remote())
    subsample = ray.get(trainer.make_subsample.remote(block))
    while True:
        # start = time.perf_counter()
        # time.sleep(1)
        # print(time.perf_counter()-start)
        print(subsample.block_pos,block)
        time.sleep(random.random()/5.0)
        block = ray.get(scheduler.get_next.remote(completed=subsample.block_pos))
        if block is None:
            return 1
        subsample = ray.get(trainer.make_subsample.remote(block))
    

scheduler = BlockScheduler.remote(multiprocessing.cpu_count()+1,3)
trainer = ExplicitMF.remote(n_factors=40) #trainer is an actor so it can be writeable

ray.get(trainer.load_samples_from_npy.remote("./movielense_27.npy",100000))
groups = trainer.generate_indpendent_samples.remote() #groups are read-only in shared memmory
ratings = trainer.get_ratings.remote()
ray.get(trainer.train.remote(1))
scheduler = BlockScheduler.remote(multiprocessing.cpu_count()+1,3)
results= ray.get([consumer.remote(scheduler,trainer,groups,ratings) for _ in range(multiprocessing.cpu_count())])
print("results",results)
counter = ray.get(scheduler.get_update_counter.remote())
print(counter)

