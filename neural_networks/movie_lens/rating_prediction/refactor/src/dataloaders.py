import pathlib
from typing import Tuple
import zipfile

import polars as pl
import pandas as pd
import requests
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = pathlib.Path(__file__).absolute().parents[1] / "data" / "ml-25m"
DATA_ZIP_PATH = pathlib.Path(__file__).absolute().parents[1] / "data" / "ml-25m.zip"


def load_movielens_data(
    holdout: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    _download_movielens_data()

    ratings_df = _load_ratings()

    # separate out a holdout set
    ratings_df, holdout_df = train_test_split(
        ratings_df, test_size=0.1, random_state=42
    )
    if holdout:
        ratings_df = holdout_df

    top_movie_ids = _get_most_frequent_movies(ratings_df)
    ratings_df = ratings_df.join(top_movie_ids, on="movieId", how="inner")
    ratings_df, user_id_mapping, movie_id_mapping = _map_users_and_movies(ratings_df)
    return ratings_df, user_id_mapping, movie_id_mapping


def _download_movielens_data():
    # download to disk if not present already
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        if not DATA_ZIP_PATH.is_file():
            print("downloading data")
            data_url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
            with open(DATA_ZIP_PATH, "wb") as f:
                f.write(requests.get(data_url).content)
        print("unzipping data")
        with zipfile.ZipFile(DATA_ZIP_PATH, "r") as zip_file:
            zip_file.extractall(DATA_PATH.parent)


def _load_movies():
    return pl.read_csv(DATA_PATH / "movies.csv")


def _load_ratings():
    df = pl.read_csv(DATA_PATH / "ratings.csv")
    df = df.sort(["userId", "timestamp"])
    return df


def _get_most_frequent_movies(ratings_df: pl.DataFrame, n: int = 50) -> pl.Series:
    top_movie_ids = (
        ratings_df.group_by("movieId")
        .count()
        .sort("count", descending=True)
        .head(n)[["movieId"]]
    )
    return top_movie_ids


def _map_users_and_movies(
    ratings_df: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Maps movie and user IDs to incremental integers.
    """

    user_id_mapping = (
        ratings_df[["userId"]]
        .unique()
        .sort("userId")
        .with_row_index(name="userIdMapped")
    )
    movie_id_mapping = (
        ratings_df[["movieId"]]
        .unique()
        .sort("movieId")
        .with_row_index(name="movieIdMapped")
    )
    ratings_df = ratings_df.join(user_id_mapping, on="userId", how="left").join(
        movie_id_mapping, on="movieId", how="left"
    )
    return ratings_df, user_id_mapping, movie_id_mapping


def get_dataloaders(
    name: str, batch_size=32, test_size=0.2, subset_ratio=1.0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns train and test DataLoaders for the MovieLens dataset using TensorDataset.
    :param file_path: Path to the MovieLens CSV file
    :param batch_size: Batch size for the DataLoader
    :param test_size: Proportion of the dataset to use for testing
    :param subset_ratio: Fraction of the dataset to use (e.g., 0.1 for 10%)
    :return: train_loader, test_loader
    """

    if name == "movielens":
        data, _, _ = load_movielens_data()

    if subset_ratio < 1.0:
        data = data.sample(frac=subset_ratio, random_state=42)

    # Convert columns to tensors
    user_ids = torch.tensor(data["userIdMapped"], dtype=torch.long)
    movie_ids = torch.tensor(data["movieIdMapped"], dtype=torch.long)
    ratings = torch.tensor(data["rating"], dtype=torch.float)

    # Split into train and test sets
    train_idx, test_idx = train_test_split(
        range(len(data)), test_size=test_size, random_state=42
    )

    train_dataset = TensorDataset(
        user_ids[train_idx], movie_ids[train_idx], ratings[train_idx]
    )
    test_dataset = TensorDataset(
        user_ids[test_idx], movie_ids[test_idx], ratings[test_idx]
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader
