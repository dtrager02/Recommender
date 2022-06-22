import multiprocessing
import queue
import time
import os
import random
from multiprocessing import Process, Queue, Lock, Manager
from collections import defaultdict
class BlockScheduler:
    def __init__(self,width,iters) -> None:
        self.width = width
        self.n_threads = width -1
        temp = [0]*width
        self.update_counter = []
        for i in range(width):
            self.update_counter.append(temp.copy())
        self.unused_rows,self.unused_cols=[*range(width)],[*range(width)]
        self.completed_rows,self.completed_cols=set(),set()
        self.iters  = iters
        print(self.update_counter)
    
    def check_completion(self):
        #print(self.unused_rows,self.unused_cols)
        if len(self.completed_rows) == self.width and len(self.completed_cols)==self.width:
            return True
        else:
            return False
    
    def get_next(self,completed=None):
        print(self.unused_rows,self.unused_cols)
        print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
      for row in self.update_counter]))
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
        output =  min_idx if min_idx != (-1,-1) else None
        if output == None:
            # self.unused_rows.clear()
            # self.unused_cols.clear()
            pass
        else:
            
            row_idx = self.unused_rows.index(min_idx[0])
            col_idx = self.unused_cols.index(min_idx[1])
            self.unused_cols.pop(col_idx)
            self.unused_rows.pop(row_idx)
        return output
            
    def completed_chunk(self,idx):
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
            print(self.unused_rows,self.completed_rows)
            print("\n\n")
        
        else:
            self.unused_rows.append(row)
            
 
 
# The consumer function takes data off of the Queue
def consumer(in_queue,out_queue, lock):
    # Synchronize access to the console
    with lock:
        print('Starting consumer => {}'.format(os.getpid()))
     
    # Run indefinitely
    while True:    
        # If the queue is empty, queue.get() will block until the queue has data
        block = in_queue.get()
        # Synchronize access to the console
        with lock:
            print('{} got {}'.format(os.getpid(), block))
        if block == "000":
            return 0
        elif block:
            time.sleep(random.randint(1,4)/1000.0)
            out_queue.put(block)
        else:
            continue
 
if __name__ == '__main__':
     
    # Some lists with our favorite characters
 
    # Create the Queue object
    m = multiprocessing.Manager()
    in_queue = m.Queue()
    out_queue = m.Queue()
    scheduler = BlockScheduler(multiprocessing.cpu_count()+2,6)
    # Create a lock object to synchronize resource access
    lock = Lock()
 
    consumers = []
 
    # Create consumer processes
    for i in range(multiprocessing.cpu_count()):
        p = Process(target=consumer, args=(in_queue,out_queue, lock))
        consumers.append(p)
        in_queue.put(scheduler.get_next())
 
    for c in consumers:
        c.start()
    
    #scheduler will handle putting and retreiving items from queues
    done = False
    while not done:
        updated_block = out_queue.get(timeout=20)
        next = scheduler.get_next(completed=updated_block)
        done = scheduler.check_completion()
        if done:
            print("Done!")
            for i in range(multiprocessing.cpu_count()):
                in_queue.put("000")
            break
        in_queue.put(next)

    
    # Like threading, we have a join() method that synchronizes our program
    for c in consumers:
        c.join()
 
    print('Parent process exiting...')
    