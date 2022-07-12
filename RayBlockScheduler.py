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
import copy
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
        min_idx = []
        for i in self.unused_rows:
            for j in self.unused_cols:
                if self.update_counter[i][j] <min and self.update_counter[i][j] <self.iters:
                    min = self.update_counter[i][j]
                    min_idx = [(i,j)]
                elif self.update_counter[i][j] == min and self.update_counter[i][j] <self.iters:
                    min_idx.append((i,j))
        if len(min_idx):
            output = random.choice(min_idx) # min_idx is the block_pos of subsample. we pick randomly to mimick true SGD
            row_idx = self.unused_rows.index(output[0])
            col_idx = self.unused_cols.index(output[1])
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
    
@numba.njit(cache=True,fastmath=True)
def sgd(idx_offset,P,Q,b_u,b_i,b,y,
        samples,ratings:csr.CSR,
        alpha,beta1,beta2):
    #adjust for the offset in row/col indexes for this block
    #"0*" are my way of ignoring a variable
    samples = np.copy(samples) # avoiding read-only
    samples[:,0] -= idx_offset[0]
    samples[:,1] -= idx_offset[1]
    for i in range(samples.shape[0]):
        user = samples[i,0]
        item = samples[i,1]
        rated_items = ratings.row_cs(user)
        R_u = np.sqrt(rated_items.shape[0])+1 #temporary division by 0 fix
        y_sum = np.sum(y[rated_items,:],axis=0)
        prediction = Q[item,:].dot((P[user, :]+0*(y_sum/R_u)).T) + b_u[user] + b_i[item] + b
        e = (samples[i,2]-prediction)
        
        b_u[user] += alpha * (e - beta1 * b_u[user])
        b_i[item] += alpha * (e - beta1 * b_i[item])
        
        P[user, :] +=alpha * (e *Q[item, :] - beta2 * P[user,:])
        Q[item, :] +=alpha * (e *(P[user, :]+(y_sum/np.sqrt(P.shape[0]))*0) - beta2 * Q[item,:])

        y[rated_items,:] += alpha*(-1.0*beta2*y[rated_items,:]+e/R_u*Q[item,:])
        
    return P,Q,y,b_u,b_i

# The consumer function takes data off of the Queue
@ray.remote
def consumer(scheduler : BlockScheduler,trainer: ExplicitMF, groups: list,ratings: csr.CSR):
    count = 0
    print(len(groups))
    print(ratings.nnz)
    print(groups[0][1].shape)
    block = scheduler.get_next.remote() #returns tuple representing grid block
    subsample = ray.get(trainer.make_subsample.remote(block)) #gets params related to that grid block
    count += 1
    print("Got first subsample",count)
    while True:
        """updating subsample with SGD results
        numpy arrays must be copied, otherwise they are read-only in shared storage
        """
        subsample = copy.deepcopy(subsample)
        subsample.P,subsample.Q,subsample.y,subsample.b_u,subsample.b_i = \
        sgd(subsample.idx_offset,
            subsample.P, 
            subsample.Q,
            subsample.b_u,
            subsample.b_i,
            subsample.b,
            subsample.y,
            groups[subsample.block_pos[0]][subsample.block_pos[1]],ratings,
            subsample.alpha,subsample.beta1,subsample.beta2)
        trainer.update_params.remote(
            subsample
            ) #this redundant copy is to prevent using object storage for SGD output
        trainer.increment.remote()
        block = ray.get(scheduler.get_next.remote(completed=subsample.block_pos))
        if block is None: # as long as training takes longer than 1s, first block cannot be none
            return 1
        subsample = ray.get(trainer.make_subsample.remote(block))
        count += 1
        #print("Got subsample",count)
n_threads = 8
scheduler = BlockScheduler.remote(n_threads+1,4)
trainer = ExplicitMF.remote(n_factors=80,n_threads=n_threads) #trainer is an actor so it can be writeable

ray.get(trainer.load_samples_from_npy.remote("./movielense_27.npy",600000))
groups = trainer.generate_indpendent_samples.remote() #groups are read-only in shared memmory
ratings = trainer.get_ratings.remote()
ray.get(trainer.train.remote(1,multithreaded=True))
results= ray.get([consumer.remote(scheduler,trainer,groups,ratings) for _ in range(n_threads)])
print("results",results)
counter = ray.get(scheduler.get_update_counter.remote())
print(counter)

