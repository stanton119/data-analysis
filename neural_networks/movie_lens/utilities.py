import polars as pl
import requests
import zipfile
import pathlib
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")


DATA_PATH = pathlib.Path(__file__).absolute().parent / "data" / "ml-25m"
DATA_ZIP_PATH = pathlib.Path(__file__).absolute().parent / "data" / "ml-25m.zip"


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


def map_users_and_movies(
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


# ######### similarity functions #########
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


def get_extreme_similarities(
    similarity_matrix: np.array, labels: List[str], top_n: int = 5
):
    print("Most similar:")
    indices = get_top_similarities(similarity_matrix, top_n=top_n, highest=True)
    for result in indices:
        print(f"{result[2]:.2f} - {labels[result[0]]}, {labels[result[1]]}")

    print("")
    print("Least similar:")
    indices = get_top_similarities(similarity_matrix, top_n=top_n, highest=False)
    for result in indices:
        print(f"{result[2]:.2f} - {labels[result[0]]}, {labels[result[1]]}")


# ######### ratings prediction #########
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split


def load_torch_data() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    ratings_df = load_ratings()
    top_movie_ids = get_most_frequent_movies(ratings_df)
    ratings_df = ratings_df.join(top_movie_ids, on="movieId", how="inner")
    ratings_df, user_id_mapping, movie_id_mapping = map_users_and_movies(ratings_df)
    return ratings_df, user_id_mapping, movie_id_mapping


def split_train_test(
    ratings_df: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train_data, _val_data = train_test_split(ratings_df, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(_val_data, test_size=1 / 3, random_state=42)
    return train_data, val_data, test_data


def _create_tensor_dataset(data):
    user_ids = torch.tensor(data["userIdMapped"], dtype=torch.long)
    item_ids = torch.tensor(data["movieIdMapped"], dtype=torch.long)
    ratings = torch.tensor(data["rating"], dtype=torch.float)
    return TensorDataset(user_ids, item_ids, ratings)


def get_data_loaders(
    ratings_df: pl.DataFrame, sample: int = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_data, val_data, test_data = split_train_test(ratings_df=ratings_df)

    dataset_train = _create_tensor_dataset(train_data)
    dataset_val = _create_tensor_dataset(val_data)
    dataset_test = _create_tensor_dataset(test_data)

    if sample:
        dataset_train = _create_tensor_dataset(train_data.head(sample))
        dataset_val = _create_tensor_dataset(val_data.head(sample))

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=2**12,
        shuffle=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=2**12,
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=2**12,
    )
    return dataloader_train, dataloader_val, dataloader_test


def training_logs_to_df(run_id, name: str = None) -> pl.DataFrame:
    """
    Convert MLFlow run id training logs to a dataframe
    """
    import mlflow

    client = mlflow.tracking.MlflowClient()

    logger_train_loss = client.get_metric_history(run_id, "train_loss_epoch")
    logger_eval_loss = client.get_metric_history(run_id, "val_loss_epoch")

    df = (
        pl.concat(
            [
                pl.DataFrame(
                    [(m.step, m.value) for m in logger_train_loss],
                    orient="row",
                    schema=["step", "train_loss"],
                ),
                pl.DataFrame(
                    [(m.step, m.value) for m in logger_eval_loss],
                    orient="row",
                    schema=["step", "val_loss"],
                ).drop("step"),
            ],
            how="horizontal",
        )
        .with_row_index(name="epoch", offset=1)
        .unpivot(index=["epoch", "step"], variable_name="dataset", value_name="loss")
    )
    if name:
        df = df.with_columns(pl.lit(name).alias("name"))
    return df


def get_training_logs_for_experiment(experiment_name: str) -> pl.DataFrame:
    """
    Get training logs dataframe for all runs in an experiment
    """
    import mlflow

    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    return pl.concat(
        [
            training_logs_to_df(run_id=run["run_id"], name=run["tags.mlflow.runName"])
            for _, run in runs.iterrows()
        ],
        how="vertical",
    )
