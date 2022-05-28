import implicit
from time import perf_counter
import sqlite3
import pandas as pd
import scipy.sparse as sparse
con = sqlite3.connect("./animeDB2.sqlite3")
def renumber_column(df:pd.Series,column:str):
        all_values = df[column].unique()
        all_values.sort()
        #print(all_values)
        converter = dict(zip(all_values,range(all_values.shape[0])))
        df[column] = df[column].map(converter)
        #print(sorted(df[column]))
        return df
start = perf_counter()
samples = pd.read_sql("select username,anime_id,score from user_records order by username limit 100000",con)
print(f"Done loading samples in {perf_counter()-start} s.")
samples = renumber_column(samples,"username")
samples = renumber_column(samples,"anime_id")
#np.random.shuffle(samples.values)
n_users = samples["username"].nunique()
n_items = samples["anime_id"].nunique()
#train_samples = samples.iloc[0:int(samples.shape[0]*.8),:]
ratings = sparse.coo_matrix((samples["score"], (samples["username"], samples["anime_id"]))).tocsr()
# initialize a model
print(ra)