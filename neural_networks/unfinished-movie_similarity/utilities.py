import polars as pl
import requests
import zipfile
import pathlib
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")


DATA_PATH = pathlib.Path().absolute() / "data" / "ml-25m"
DATA_ZIP_PATH = pathlib.Path().absolute() / "data" / "ml-25m.zip"


def download_data():
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


def load_movies():
    return pl.read_csv(DATA_PATH / "movies.csv")


def load_ratings():
    df = pl.read_csv(DATA_PATH / "ratings.csv")
    df = df.sort(["userId", "timestamp"])
    return df


def get_most_frequent_movies(ratings_df: pl.DataFrame, n: int = 50) -> pl.Series:
    top_movie_ids = (
        ratings_df.group_by("movieId")
        .count()
        .sort("count", descending=True)
        .head(n)[["movieId"]]
    )
    return top_movie_ids


def plot_similarities(similarity_matrix: np.array, labels: List[str]):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        data=similarity_matrix,
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
    )
    fig.show()


def get_top_similarities(
    similarity_matrix: np.array, top_n: int = 5, highest: bool = True
) -> List[Tuple[int, int, float]]:
    """
    Get the indices of the highest or lowest similarity values in a similarity matrix, ignoring NaNs.
    """
    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("The similarity matrix must be square (n x n).")

    # Mask diagonal values (self-similarities) and NaNs
    sim_matrix = similarity_matrix.copy()
    np.fill_diagonal(sim_matrix, np.nan)  # Ignore self-similarities
    valid_mask = ~np.isnan(sim_matrix)  # Mask for non-NaN values

    # Flatten valid similarities and their indices
    valid_indices = np.where(valid_mask)
    valid_values = sim_matrix[valid_indices]

    # Sort valid values by similarity
    sorted_indices = np.argsort(valid_values)
    if highest:
        sorted_indices = sorted_indices[::-1]  # Reverse for descending order

    # Select top_n
    top_indices = sorted_indices[:top_n]

    # Get corresponding row/column indices and similarity values
    row_indices = valid_indices[0][top_indices]
    col_indices = valid_indices[1][top_indices]
    similarities = valid_values[top_indices]

    return list(zip(row_indices, col_indices, similarities))


def get_extreme_similarities(similarity_matrix: np.array, labels: List[str]):
    print("Most similar:")
    indices = get_top_similarities(similarity_matrix, top_n=5, highest=True)
    for result in indices:
        print(f"{result[2]:.2f} - {labels[result[0]]}, {labels[result[1]]}")

    print("")
    print("Least similar:")
    indices = get_top_similarities(similarity_matrix, top_n=5, highest=False)
    for result in indices:
        print(f"{result[2]:.2f} - {labels[result[0]]}, {labels[result[1]]}")
