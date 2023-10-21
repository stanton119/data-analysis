"""
Based on ratings only, no meta info from the movies
ratings.csv - userId,movieId,rating,timestamp

To start:
Limit the file to movieId < number or userId < number

Measure cosine similarity across different users
Maybe use dask for large file size

recommender_env

dataset - https://grouplens.org/datasets/movielens/25m/

"""
# %% [markdown]
#
#
# # Exploring multi-armed bandit baseline strategies
# Our problem:
# For the next 100 days, we will have 1 hour to play a video games.
# We have 20 games, but have no idea which one we will enjoy the most.
# How do we decide what to play each day?
# We assume that the enjoyment we get from a single hour is random and comes from a beta distribution.
# Each game has a different distribution.
# Each hour we play of a game gives us an enjoyment value and helps build our knowledge of that game.
#
# We need to compromise exploring which games are enjoyable and sticking to the games we know so far to be good.
# This can be represented by a multi-armed bandit problem.
#
# Here I will explore some simple benchmark solutions.
# %%
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use("seaborn-whitegrid")

np.random.seed(0)
# %%
df_ratings_pd = pd.read_csv(
    Path(__file__).parent / "ml-25m" / "ratings.csv", nrows=1000
)
df_ratings_pd
# %%
import dask.dataframe as dd

df_ratings = dd.read_csv(Path(__file__).parent / "ml-25m" / "ratings.csv")

# %%
# preprocess
# limit userId
max_users = 100
df_ratings = df_ratings.loc[df_ratings["userId"] <= max_users]
df_ratings = df_ratings.compute()
df_ratings

# %%
# convert user/movie IDs into a unique mapping
def create_mapping(df):
    mapping = df.drop_duplicates().reset_index(drop=True)
    mapping.index = mapping.index.rename(df.name + "U")
    return mapping.reset_index()


movie_mapping = create_mapping(df_ratings["movieId"])
user_mapping = create_mapping(df_ratings["userId"])

df_ratings = df_ratings.merge(movie_mapping, on="movieId").merge(
    user_mapping, on="userId"
)

# %%
# estimate SVD
# based on unpivoted matrix?
import tqdm

n_users = user_mapping.shape[0]
n_movies = movie_mapping.shape[0]
n_latent_features = 20
p = np.random.normal(0, 0.1, (n_users, n_latent_features))
q = np.random.normal(0, 0.1, (n_movies, n_latent_features))
alpha = 0.01

def train_svd(df_ratings, p, q, alpha):
    for idx, row in tqdm.tqdm(df_ratings[["userIdU","movieIdU","rating"]].iterrows()):
        u, i, r_ui = int(row["userIdU"]), int(row["movieIdU"]), row["rating"]
        err = r_ui - np.dot(p[u], q[i])
        # Update vectors p_u and q_i
        p[u] += alpha * err * q[i]
        q[i] += alpha * err * p[u]
    return p, q

def get_rating_est(df_ratings, p, q):
    ratings_est = []
    for idx, row in tqdm.tqdm(df_ratings.iterrows()):
        u, i, r_ui = int(row["userIdU"]), int(row["movieIdU"]), row["rating"]
        ratings_est.append(np.dot(p[u], q[i]))
    ratings_est = np.array(ratings_est)
    return ratings_est

def get_error(y, y_est):
    return np.mean(np.power(y-y_est, 2))

err = []
for epoch in range(20):
    p, q = train_svd(df_ratings, p, q, alpha)
    df_ratings.loc[:, f"ratings_est_{epoch}"] = get_rating_est(df_ratings, p, q)
    err.append(get_error(df_ratings.loc[:, "rating"], df_ratings.loc[:, f"ratings_est_{epoch}"]))

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(err)
ax.set_xlabel('epoch')
ax.set_ylabel('MSE')

fig, ax = plt.subplots(figsize=(10,6))
df_ratings['rating_p'] = df_ratings['rating'] + np.random.normal(0, 0.05, df_ratings.shape[0])
df_ratings.plot(x='rating_p', y=df_ratings.columns[-2], kind='scatter', ax=ax)
# %%
# error against epoch - should converge
# error against n_latent variables - will start to overfit
# %%
# build with minibatches instead of stochastic gradient descent
# build in pytorch

def train_svd(df_ratings, p, q, alpha):
    for idx, row in tqdm.tqdm(df_ratings.iterrows()):
        u, i, r_ui = int(row["userIdU"]), int(row["movieIdU"]), row["rating"]
        err = r_ui - np.dot(p[u], q[i])
        # Update vectors p_u and q_i
        p[u] += alpha * err * q[i]
        q[i] += alpha * err * p[u]
    return p, q

import timeit

t1 = timeit.default_timer()
for _ in range(1000):
    df_temp = df_ratings[["userIdU","movieIdU","rating"]].iloc[:10]
    pred = np.diag(np.dot(p[df_temp["userIdU"]], q[df_temp["movieIdU"]].transpose()))

    err = df_temp["rating"].to_numpy() - pred
    p[df_temp["userIdU"]] += alpha * err[:, np.newaxis] * q[df_temp["movieIdU"]]
    q[df_temp["movieIdU"]] += alpha * err[:, np.newaxis] * p[df_temp["userIdU"]]
print(timeit.default_timer() - t1)

p[df_temp["userIdU"]]

err[:, np.newaxis] * q[df_temp["movieIdU"]]

t1 = timeit.default_timer()
for _ in range(1000):
    df_temp = df_ratings[["userIdU","movieIdU","rating"]].iloc[:10]
    pred = np.sum(p[df_temp["userIdU"]] * q[df_temp["movieIdU"]], axis=1)

    err = df_temp["rating"].to_numpy() - pred
    p[df_temp["userIdU"]] += alpha * err[:, np.newaxis] * q[df_temp["movieIdU"]]
    q[df_temp["movieIdU"]] += alpha * err[:, np.newaxis] * p[df_temp["userIdU"]]
print(timeit.default_timer() - t1)

t1 = timeit.default_timer()
for _ in range(1000):
    for idx, row in df_ratings.iloc[:10].iterrows():
        u, i, r_ui = int(row["userIdU"]), int(row["movieIdU"]), row["rating"]
        pred = np.dot(p[u], q[i])
        
        err = r_ui - np.dot(p[u], q[i])
        # Update vectors p_u and q_i
        p[u] += alpha * err * q[i]
        q[i] += alpha * err * p[u]
print(timeit.default_timer() - t1)

# %%
# single latent factor = highest averaging scoring ratings?


# %%
# build with pytorch embedding layers
# https://medium.com/@rinabuoy13/explicit-recommender-system-matrix-factorization-in-pytorch-f3779bb55d74
import torch
embedding = torch.nn.Embedding(10, 3)
# a batch of 2 samples of 4 indices each
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
embedding(input)
# %%
