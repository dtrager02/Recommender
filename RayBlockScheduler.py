import multiprocessing
import sys
import random
from time import perf_counter
import ray
import numba
import numpy as np
import csr
import copy
from collections import defaultdict
from SubSample import SubSample
from SGDRecommender import ExplicitMF

@ray.remote(num_cpus=.5)
class BlockScheduler:
    def __init__(self,width,iters,verbose=False) -> None:
        self.width = width
        self.verbose = verbose
        self.n_threads = self.width -1
        temp = [0]*self.width
        self.update_counter = []
        for i in range(self.width):
            self.update_counter.append(temp.copy())
        self.unused_rows,self.unused_cols=[*range(self.width)],[*range(self.width)]
        self.completed_rows,self.completed_cols=set(),set()
        self.iters  = iters
        self.test = 0
        self.delays = [0]*self.width
    
    def check_completion(self):
        #print(self.unused_rows,self.unused_cols)
        return len(self.completed_rows) == self.width and len(self.completed_cols)==self.width

    
    def get_next(self,completed=None):
        # print("Unused rows",self.unused_rows,"Unused cols: ",self.unused_cols)
        # print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
        if completed:
            if self.verbose:
                print("Completed",completed)
            self.completed_chunk(completed)
        # if len(self.unused_cols) != len(self.unused_rows):
        #     raise Exception(f"Desync occurred, # of unused rows and cols is not the same\n{self.unused_rows}\n{self.unused_cols}")
        if not len(self.unused_cols):
            return None
        min = 10**10
        min_idx = []
        if self.verbose:
            print("Unused rows/cols",self.unused_rows,self.unused_cols)
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
        if self.verbose:
            print("Chose",output)
            print("--------------------------")
        return output

    @staticmethod 
    def pretty_2d(arr):
        return "\n".join(str(s) for s in arr)
    
    def completed_chunk(self,idx):
        self.test += 1
        if self.test % (self.width**2) == 0:  
            print("Total:",sum([sum(i) for i in self.update_counter]))
            print("Update Counter:\n",self.pretty_2d(self.update_counter))
        row = idx[0]
        col = idx[1]
        if self.verbose:
            for i in range(len(self.delays)):
                self.delays[i]+= 1
            print("Delays",self.delays)
            self.delays[row] =0
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
        samples,row_ptrs:np.ndarray,col_inds:np.ndarray,
        alpha,beta1,beta2):
    #adjust for the offset in row/col indexes for this block
    #"0*" are my way of ignoring a variable
    samples = np.copy(samples) # avoiding read-only
    samples[:,0] -= idx_offset[0]
    samples[:,1] -= idx_offset[1]
    for i in range(samples.shape[0]):
        user = samples[i,0]
        item = samples[i,1]
        #must add offsets back because CSR was made without offset
        rated_items = col_inds[row_ptrs[user+idx_offset[0]]:row_ptrs[user+idx_offset[0]+1]]
        R_u = np.sqrt(rated_items.shape[0]) #temporary division by 0 fix
        y_sum = np.sum(y[rated_items,:],axis=0)
        prediction = Q[item,:].dot((P[user, :]+ y_sum/R_u).T) + b_u[user] + b_i[item] + b
        e = samples[i,2]-prediction
        
        b_u[user] += alpha * (e - beta1 * b_u[user])
        b_i[item] += alpha * (e - beta1 * b_i[item])
        
        P[user, :] +=alpha * (e *Q[item, :] - beta2 * P[user,:])
        Q[item, :] +=alpha * (e *(P[user, :]+y_sum/R_u) - beta2 * Q[item,:]) #

        y[rated_items,:] += alpha*(-1.0*beta2*y[rated_items,:]+e/R_u*Q[item,:])
        
    return P,Q,y,b_u,b_i

# The consumer function takes data off of the Queue
@ray.remote
def consumer(scheduler : BlockScheduler,trainer: ExplicitMF, groups: list,row_ptrs:np.ndarray,col_inds:np.ndarray):
    count = 0
    block = ray.get(scheduler.get_next.remote()) #returns tuple representing grid block
    subsample = ray.get(trainer.make_subsample.remote(block)) #gets params related to that grid block
    count += 1
    # start = perf_counter()
    # print(rated_by_user[1:3])
    # print(perf_counter()-start)
    print("Got first subsample",count)
    while True:
        """updating subsample with SGD results
        numpy arrays must be copied, otherwise they are read-only in shared storage
        """
        subsample = copy.deepcopy(subsample)
        #print(f"Got {subsample.block_pos},idx_offset: {subsample.idx_offset}")
        subsample.P,subsample.Q,subsample.y,subsample.b_u,subsample.b_i = \
        sgd(subsample.idx_offset,
            subsample.P, 
            subsample.Q,
            subsample.b_u,
            subsample.b_i,
            subsample.b,
            subsample.y,
            groups[subsample.block_pos[0]][subsample.block_pos[1]],row_ptrs,col_inds,
            subsample.alpha,subsample.beta1,subsample.beta2)
        trainer.update_params.remote(subsample) #this redundant copy is to prevent using object storage for SGD output
        trainer.increment.remote()
        block = ray.get(scheduler.get_next.remote(completed=subsample.block_pos))
        if block is None: # as long as training takes longer than 1s, first block cannot be none
            return 1
        subsample = ray.get(trainer.make_subsample.remote(block))
        count += 1
        #print("Got subsample",count)

n_threads = int(sys.argv[1]) if len(sys.argv) > 1 else multiprocessing.cpu_count()
n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else "all"
scheduler = BlockScheduler.remote(n_threads+1,5)
trainer = ExplicitMF.remote(n_factors=80,n_threads=n_threads) #trainer is an actor so it can be writeable

ray.get(trainer.load_samples_from_npy.remote("./movielense_27.npy",n_samples))
groups = trainer.generate_indpendent_samples_new.remote() #groups are read-only in shared memmory
indices = ray.get(trainer.get_rated_by_user.remote()) #changed to a list of variable-length numpy arrays instead of
row_ptrs = indices[0]
col_inds = indices[1]
ray.get(trainer.train.remote(1,multithreaded=True))
results= ray.get([consumer.remote(scheduler,trainer,groups,row_ptrs,col_inds) for _ in range(n_threads)])
print("results",results)
counter = ray.get(scheduler.get_update_counter.remote())
print(counter)
trainer.unrandomize_sampls.remote()
trainer.save_all_factors.remote("SGD_factors")


