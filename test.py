import numpy as np
import time
# a = np.ones((40)).reshape(10,4)
# b = np.arange(10)[:,None]
# print(np.sum(b*a,axis=0))
from numpy.linalg import solve
from scipy import sparse
a = np.eye(40)
b = np.arange(40)
#print(np.nonzero(b)[0])
# l = np.arange(95)
# for i in range(0,len(l),10):
#     print(l[i:i+10])
sparse_arr = sparse.rand(5,5,density=.3,random_state=5).tocsc()
print(sparse_arr.toarray())
#nz = sparse_arr.nonzero().T
#temp = np.argwhere(nz[:,0]==2)




start = time.perf_counter()
print(sparse_arr.tolil().data)
print(time.perf_counter()-start)
start = time.perf_counter()
for i in range(5):
    sparse_arr[i].nonzero()
print(time.perf_counter()-start)
def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 ratings,
                 _lambda,
                 type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'user':
            # Precompute
            YTY = np.matmul(fixed_vecs.T,fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in range(latent_vectors.shape[0]):
                #print((YTY + lambdaI).shape,np.matmul(ratings[u,:].toarray()[0],fixed_vecs).shape)
                latent_vectors[u, :] = solve(YTY + lambdaI, 
                                             np.matmul(ratings[u, :].toarray().reshape((-1,)),fixed_vecs))
        elif type == 'item':
            # Precompute
            XTX = np.matmul(fixed_vecs.T,fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda
            
            for i in range(latent_vectors.shape[0]):
                #print((XTX + lambdaI).shape,ratings[:, i].toarray().reshape((-1,)).shape,fixed_vecs.shape)
                latent_vectors[i, :] = solve(XTX + lambdaI, 
                                             np.matmul(ratings[:, i].toarray().reshape((-1,)),fixed_vecs))
        return latent_vectors