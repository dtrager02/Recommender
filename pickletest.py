import pickle
import numpy as np
from time import perf_counter
import os
b = np.arange(1000_000_000).reshape(1000,1000_000)
start = perf_counter()
print(b.nbytes)
a = pickle.dumps(b)
c = pickle.loads(a)
print(perf_counter()-start)