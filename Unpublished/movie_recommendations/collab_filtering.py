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
df_ratings_pd = pd.read_csv(Path(__file__).parent / "ml-25m"/"ratings.csv", nrows=1000)
df_ratings_pd
# %%
import dask.dataframe as dd
df_ratings = dd.read_csv(Path(__file__).parent / "ml-25m"/"ratings.csv")
# df_ratings['userId'].max().compute()
# df_ratings['userId'].drop_duplicates().compute()

# %%
# preprocess
# limit userId
df_ratings = df_ratings.loc[df_ratings['userId']<10000]
df_ratings = df_ratings.compute()
# df_ratings['movieId'].max()
# df_ratings['movieId'].drop_duplicates()

# pivot

# %%
# find cosine similarity
import tqdm

def get_combined_ratings(ref_ratings, user_ratings):
    combined_ratings = pd.concat([ref_ratings, user_ratings], axis=1)
    combined_ratings = combined_ratings.loc[combined_ratings.notna().all(axis=1)]
    return combined_ratings

def get_cosine_similarity(combined_ratings):
    return combined_ratings.prod(axis=1).sum() / combined_ratings.pow(2).sum().pow(0.5).prod()

ref_ratings = df_ratings.loc[df_ratings['userId']==1]
ref_ratings = ref_ratings[['movieId','rating']].set_index('movieId')
user_similarity = {}

for user_id in tqdm.tqdm(df_ratings['userId'].drop_duplicates()):
    user_ratings = df_ratings.loc[df_ratings['userId']==user_id]

    user_ratings = user_ratings[['movieId','rating']].set_index('movieId')

    combined_ratings = get_combined_ratings(ref_ratings, user_ratings)
    cosine_similarity = get_cosine_similarity(combined_ratings)
    user_similarity[user_id] = [cosine_similarity, combined_ratings.shape[0]]

user_similarity = pd.DataFrame.from_dict(user_similarity, orient='index', columns=['similarity','length'])
user_similarity = user_similarity.sort_values(by=['length', 'similarity'], ascending=False)

user_similarity_filt = user_similarity.loc[user_similarity['length']>20]


# %%
user_ratings = df_ratings.loc[df_ratings['userId']==user_similarity_filt.index[1]]
user_ratings = user_ratings[['movieId','rating']].set_index('movieId')
combined_ratings = get_combined_ratings(ref_ratings, user_ratings)
combined_ratings



# %%
df_genome_scores = pd.read_csv(Path(__file__).parent / "ml-25m"/"genome-scores.csv", nrows=100)
df_genome_tags = pd.read_csv(Path(__file__).parent / "ml-25m"/"genome-tags.csv", nrows=100)
df_links = pd.read_csv(Path(__file__).parent / "ml-25m"/"links.csv", nrows=100)
df_movies = pd.read_csv(Path(__file__).parent / "ml-25m"/"movies.csv", nrows=100)
df_ratings = pd.read_csv(Path(__file__).parent / "ml-25m"/"ratings.csv", nrows=100)
df_tags = pd.read_csv(Path(__file__).parent / "ml-25m"/"tags.csv", nrows=100)
