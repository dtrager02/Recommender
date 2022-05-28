import numpy as np
import pandas as pd
import sqlite3
from time import perf_counter
import tensorflow.compat.v1 as tf
class MF():

    # Initializing the user-movie rating matrix, no. of latent features, alpha and beta.
    def __init__(self, K, alpha, beta, iterations):
        #self.R = Rscipy
        #self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        
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
    
    def load_samples(self):
        try:
            start = perf_counter()
            self.samples = pd.read_sql("select username,anime_id,score from user_records order by username limit 500000",self.con)
            print(f"Done loading samples in {perf_counter()-start} s.")
            self.samples = MF.renumber_column(self.samples,"username")
            self.samples = MF.renumber_column(self.samples,"anime_id")
            np.random.shuffle(self.samples.values)
            self.train_samples = self.samples.iloc[0:int(self.samples.shape[0]*.8),:]
            self.test_samples = self.samples.iloc[int(self.samples.shape[0]*.8):,:]
            self.train_samples_grouped = self.train_samples.groupby("anime_id")
            print(f"# of train samples:{self.train_samples.shape[0]}\n# of test samples:{self.test_samples.shape[0]}")
        except (Exception) as e:
            print("No database loaded\n")
            print(e)
    

    # Initializing user-feature and movie-feature matrix 
    def train(self):
        self.num_users = self.samples["username"].max() + 1
        self.num_items = self.samples["anime_id"].max() + 1
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.samples["score"])

        # List of training samples
        # self.samples = [
        # (i, j, self.R[i, j])
        # for i in range(self.num_users)
        # for j in range(self.num_items)
        # if self.R[i, j] > 0
        # ]

        # Stochastic gradient descent for given number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.train_samples.values)
            self.sgd()
            if (i+1) % 2 == 0:
                train_mse = self.mse()
                training_process.append((i, train_mse))
                print("Iteration: %d ; train error = %.4f" % (i+1, train_mse))
        print(f"Final test RMSE:{self.mse(test=True)}")
        return training_process

    # Computing total mean squared error
    def mse(self,test=False):
        train_errors, test_errors = [],[]
        if test:
            for i, j, r in self.test_samples.values:
                prediction = self.get_rating(i, j)
                test_errors.append(r - prediction)
            test_errors = np.array(test_errors)
            return np.sqrt(np.sum(np.square(test_errors))/test_errors.shape[0])
        
        for i, j, r in self.train_samples.values:
            prediction = self.get_rating(i, j)
            train_errors.append(r - prediction)
        train_errors = np.array(train_errors)
        return np.sqrt(np.sum(np.square(train_errors))/train_errors.shape[0])

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        total_time,op_time = 0,0
        total_start = perf_counter()
        indices = self.train_samples_grouped.indices
        for i, j, r in self.train_samples.values:
            
            prediction = self.get_rating(i, j)
            
            
            e = (r - prediction)
            op_start = perf_counter()
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])
            op_end = perf_counter()
            op_time += (op_end-op_start)
        total_end = perf_counter()
        print(f"Operation time: {op_time}; total time: {total_end-total_start}")      

    def sgd2(self):
        total_time,op_time = 0,0
        total_start = perf_counter()
        indices = self.train_samples_grouped.indices
        for j in indices.keys():
            prediction = self.get_item_ratings(j)
            r = np.copy(prediction)
            # for idx in indices[j]:
            #     row = self.train_samples.iloc[idx,[0,2]]
            #     r[row[0]] = row[1] #index is username, value is score
            e = (r - prediction)
            op_start = perf_counter()
            self.b_u += self.alpha * (e - self.beta * self.b_u)
            self.b_i[j] += np.sum(self.alpha * (e - self.beta * self.b_i[j])) #TEMP IMPERFECT SOLUTION

            self.P += self.alpha * (e[:,None] * self.Q[j, :] - self.beta * self.P) #e=users x 1, Q[j,:] = 1 x d
            self.Q[j, :] += self.alpha * (np.sum(e[:,None] * self.P,axis=0) - self.beta * self.Q[j,:]) #e=users x 1, P = users x d
            op_end = perf_counter()
            op_time += (op_end-op_start)
        total_end = perf_counter()
        print(f"Operation time: {op_time}; total time: {total_end-total_start}")      
    # Ratings for user i and moive j
    
    def train2(self):
        user_ids = np.array(self.train_samples['username'].tolist())
        movie_ids = np.array(self.train_samples['anime_id'].tolist())
        user_ratings = np.array(self.train_samples['score'].tolist())
        graph = tf.Graph()
        n_movie = self.train_samples["anime_id"].max()+1
        n_user = self.train_samples["username"].max() + 1 
        embedding_size = 40

        lr = 0.0005
        reg = 0.01

        with graph.as_default():
            user = tf.compat.v1.placeholder(tf.int32, name="user_id") 
            movie = tf.compat.v1.placeholder(tf.int32, name="movie_id") 
            rating = tf.compat.v1.placeholder(tf.float32, name="rating") 

            movie_embedding = tf.Variable(tf.compat.v1.truncated_normal([n_movie, embedding_size], stddev=0.02, mean=0.) ,        name="movie_embedding")
            user_embedding = tf.Variable(tf.compat.v1.truncated_normal([n_user, embedding_size], stddev=0.02, mean=0.) ,name="user_embedding")

            movie_bias_embedding = tf.Variable(tf.compat.v1.truncated_normal([n_movie], stddev=0.02, mean=0.) ,name="movie_bias_embedding")
            user_bias_embedding = tf.Variable(tf.compat.v1.truncated_normal([n_user], stddev=0.02, mean=0.) ,name="user_bias_embedding")


            global_bias = tf.Variable(tf.compat.v1.truncated_normal([], stddev=0.02, mean=0.) ,name="global_bias")

            u = tf.nn.embedding_lookup(user_embedding, user)
            m = tf.nn.embedding_lookup(movie_embedding, movie)

            u_bias = tf.nn.embedding_lookup(user_bias_embedding, user)
            m_bias = tf.nn.embedding_lookup(movie_bias_embedding, movie)


            predicted_rating = tf.reduce_sum(tf.multiply(u, m), 1) + u_bias + m_bias + global_bias

            rmse = tf.sqrt(tf.reduce_mean(tf.square(predicted_rating - rating))) # RMSE
            cost = tf.nn.l2_loss(predicted_rating - rating)
            regularization = reg * (tf.nn.l2_loss(movie_embedding) + tf.nn.l2_loss(user_embedding)
                                    + tf.nn.l2_loss(movie_bias_embedding) + tf.nn.l2_loss(user_bias_embedding))

            loss = cost + regularization

            optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
            
            batch_size = 5
            n_epoch = 30


            with tf.Session(graph=graph) as sess:
                tf.initialize_all_variables().run()
                for _ in range(n_epoch):
                    for start in range(0, user_ratings.shape[0] - batch_size, batch_size):
                        end = start + batch_size
                        _, cost_value = sess.run([optimizer, rmse], feed_dict={user: user_ids[start:end],
                                                              movie: movie_ids[start: end],
                                                              rating: user_ratings[start: end]})

                    print ("RMSE", cost_value)
                embeddings = movie_embedding.eval()

    
    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction
    
    def get_item_ratings(self, j): #returns nx1 matrix representing all user ratings for item j
        prediction = self.b + self.b_u + self.b_i[j] + np.matmul(self.P, self.Q[j, :].T)
        return prediction

    # Full user-movie rating matrix
    #def full_matrix(self):
    #    return mf.b + mf.b_u[:,np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)
    
if __name__ == "__main__":
    mf = MF(40,.01,.001,20)
    mf.link_db("./animeDB2.sqlite3")
    mf.load_samples()
    mf.train2()