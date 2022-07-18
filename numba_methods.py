import numpy as np
import numba
import csr

# Stochastic gradient descent to get optimized P and Q matrix  

@numba.njit(cache=True,fastmath=True)
def sgd(P,Q,b_u,b_i,b,y,
    samples,row_ptrs:np.ndarray,col_inds:np.ndarray,
    alpha,beta1,beta2):
    for i in range(samples.shape[0]):
        user = samples[i,0]
        item = samples[i,1]
        rated_items = col_inds[row_ptrs[user]:row_ptrs[user+1]]#np.array([1,123,45])#ratings.row_cs(user)
        R_u = np.sqrt(rated_items.shape[0]) #temporary division by 0 fix
        y_sum = np.sum(y[rated_items,:],axis=0)
        prediction = Q[item,:].dot((P[user, :]+ y_sum/R_u).T) + b_u[user] + b_i[item] + b
        e = samples[i,2]-prediction
        b_u[user] += alpha * (e - beta1 * b_u[user])
        b_i[item] += alpha * (e - beta1 * b_i[item])
        P[user] +=alpha * (e *Q[item, :] - beta2 * P[user,:])
        Q[item, :] +=alpha * (e *(P[user, :]+y_sum/R_u) - beta2 * Q[item,:]) #
        y[rated_items,:] += alpha*(-1.0*beta2*y[rated_items,:]+e/R_u*Q[item,:])
    return P,Q,y,b_u,b_i
@staticmethod
@numba.njit(cache=True,fastmath=True)
def sgd2(P,Q,b_u,b_i,b,y,
        samples,row_ptrs:np.ndarray,col_inds:np.ndarray,
        alpha,beta1,beta2):
    n_factors = P.shape[1]
    for i in range(samples.shape[0]):
        y_sum = np.zeros(n_factors)
        user = samples[i,0]
        item = samples[i,1]
        rated_items = col_inds[row_ptrs[user]:row_ptrs[user+1]]#np.array([1,123,45])#ratings.row_cs(user)
        R_u = np.sqrt(rated_items.shape[0])+1.0 #temporary division by 0 fix
        for j in range(rated_items.shape[0]):
            for k in range(n_factors):
                y_sum[k] += y[rated_items[j],k]
        pred = b + b_u[user] + b_i[item]
        
        for factor in range(n_factors):
            pred += (P[user, factor]+y_sum[factor]/R_u) * Q[item, factor]
        err = samples[i,2] - pred
        # Update biases
        b_u[user] += alpha * (err - beta1 * b_u[user])
        b_i[item] += alpha * (err - beta1 * b_i[item])
        # Update latent factors
        for factor in range(n_factors):
            puf = P[user, factor]
            qif = Q[item, factor]
            P[user, factor] += alpha * (err * qif - beta2 * puf)
            Q[item, factor] += alpha * (err * (puf+y_sum[factor]/R_u) - beta2 * qif)
            for j in range(rated_items.shape[0]):
                y[rated_items[j],factor] += alpha*(err/R_u*qif - beta2 * y[rated_items[j],factor])
            
    return P,Q,y,b_u,b_i

@numba.njit(cache=True,fastmath=True)
def mse(samples,row_ptrs:np.ndarray,col_inds:np.ndarray,
        P,Q,b_u,b_i,b,y,exact=False): # samples format : user,item,rating
    if not exact:
        #reduces computation cost if exactness is not needed
        dropsize = int(samples.shape[0]/10) if samples.shape[0]<10000 else 10000
        drop_indices = np.random.choice(samples.shape[0],size=dropsize,replace=False)
    samples = samples[drop_indices,:]
    test_errors = np.zeros(samples.shape[0])
    size = 100
    for i in range(0,samples.shape[0],size):
        users = samples[i:i+size,0]
        items = samples[i:i+size,1]
        rated_items = []
        R_u = np.empty(shape=len(users))
        y_sum = np.empty(shape=(items.shape[0],P.shape[1]))
        for j in range(len(users)):
            rated_items.append(col_inds[row_ptrs[users[j]]:row_ptrs[users[j+1]]]) #size x n
            R_u[j] = np.sqrt(rated_items[j].shape[0])+1 #temporary division by 0 fix
            y_sum[j] = np.sum(y[rated_items[j],:],axis=0)
        R_u = R_u.reshape((R_u.shape[0],1))
        predictions = np.sum(Q[items,:]*(P[users, :]+ y_sum/R_u),axis=1) + b_u[users] + b_i[items] + b #does multiple preductionsat once
        test_errors[i:i+size] = samples[i:i+size,2] - predictions
    return np.sqrt(np.sum(np.square(test_errors))/test_errors.shape[0])

@numba.njit(cache=True,fastmath=True)
def mse2(samples,row_ptrs:np.ndarray,col_inds:np.ndarray,
        P,Q,b_u,b_i,b,y,exact=False):
    if not exact:
        #reduces computation cost if exactness is not needed
        dropsize = int(samples.shape[0]/10) if samples.shape[0]<50000 else 50000
        drop_indices = np.random.choice(samples.shape[0],size=dropsize,replace=False)
    samples = samples[drop_indices,:]
    residuals = np.empty(samples.shape[0])

    for i in range(samples.shape[0]):
        user, item, rating = int(samples[i, 0]), int(samples[i, 1]), samples[i, 2]
        rated_items = col_inds[row_ptrs[user]:row_ptrs[user+1]]#np.array([1,123,45])#ratings.row_cs(user)
        R_u = np.sqrt(rated_items.shape[0]) #temporary division by 0 fix
        y_sum = np.sum(y[rated_items],axis=0)
        pred = Q[item,:].dot((P[user, :]+ y_sum/R_u).T) + b + b_u[user] + b_i[item]
        residuals[i] = (rating - pred)
    loss = np.square(residuals).mean()
    rmse = np.sqrt(loss)
    return rmse
@numba.njit(cache=True)
def add_users_to_sparse(user_data,ratings:csr.CSR):
    # print(self.ratings.colinds)
    # print(self.ratings.values[77:100])
    values_ = user_data[:,2]
    col_idx = user_data[:,1]
    offset = ratings.nnz
    rowptrs = np.concatenate((ratings.rowptrs,np.array([offset+values_.shape[0]])))
    colinds = np.concatenate((ratings.colinds,col_idx))
    values = np.concatenate((ratings.values,values_))
    print(colinds.shape,values.shape)
    nrows = ratings.nrows+1
    ncols = np.unique(colinds).shape[0]
    nnz = ratings.nnz+values.shape[0]
    ratings_new = csr.create(nrows, ncols, nnz, rowptrs, colinds, values)
    print(ratings_new.row_cs(ratings_new.nrows-1))
    print(ratings_new.row_vs(ratings_new.nrows-1))
    #print(ratings_new.row(ratings_new.nrows-1).nonzero())
    return ratings_new, ratings
@numba.njit(cache=True)
def update_existing_sparse_ratings(user_data,ratings:csr.CSR):
    values = ratings.values.astype(numba.int64)
    colinds = ratings.colinds.astype(numba.int64)
    rowptrs = ratings.rowptrs
    for i in range(user_data.shape[0]):
        value = user_data[i,2]
        col_idx = user_data[i,1]
        user = user_data[i,0]
        
        colinds=  np.concatenate((colinds[:rowptrs[user]],
                                  np.array([col_idx]),
                                  colinds[rowptrs[user]:]))
        values =  np.concatenate((values[:rowptrs[user]],
                                  np.array([value]),
                                  values[rowptrs[user]:]))
        rowptrs[user+1:] += 1
    nrows = ratings.nrows
    ncols = np.max(colinds)+1
    nnz = colinds.shape[0]
    ratings_new = csr.create(nrows, ncols, nnz, rowptrs, colinds, values)
    return ratings_new, ratings