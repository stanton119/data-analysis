import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
import zipfile
import urllib.request
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"


class MovieLensDataset(Dataset):
    def __init__(self, interactions, num_users, num_items, num_negatives=4):
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives

        # Create user-item interaction matrix for fast negative sampling
        self.user_items = {}
        for row in interactions.iter_rows(named=True):
            user_id = row["user_id"]
            if user_id not in self.user_items:
                self.user_items[user_id] = set()
            self.user_items[user_id].add(row["item_id"])

    def __len__(self):
        return len(self.interactions) * (1 + self.num_negatives)

    def __getitem__(self, idx):
        # Determine if this is a positive or negative sample
        interaction_idx = idx // (1 + self.num_negatives)
        sample_type = idx % (1 + self.num_negatives)

        row = self.interactions.row(interaction_idx, named=True)
        user_id = row["user_id"]

        if sample_type == 0:  # Positive sample
            return {
                "user_id": torch.tensor(user_id, dtype=torch.long),
                "item_id": torch.tensor(row["item_id"], dtype=torch.long),
                "rating": torch.tensor(1.0, dtype=torch.float32),
            }
        else:  # Negative sample
            # Sample random item not interacted with by user
            while True:
                neg_item = np.random.randint(0, self.num_items)
                if neg_item not in self.user_items[user_id]:
                    break

            return {
                "user_id": torch.tensor(user_id, dtype=torch.long),
                "item_id": torch.tensor(neg_item, dtype=torch.long),
                "rating": torch.tensor(0.0, dtype=torch.float32),
            }


DATASET_CONFIGS = {
    "100k": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "folder": "ml-100k",
        "ratings_file": "u.data",
        "separator": "\t",
        "columns": ["user_id", "item_id", "rating", "timestamp"],
    },
    "1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "folder": "ml-1m",
        "ratings_file": "ratings.dat",
        "separator": "::",
        "columns": ["user_id", "item_id", "rating", "timestamp"],
    },
    "25m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "folder": "ml-25m",
        "ratings_file": "ratings.csv",
        "separator": ",",
        "columns": ["userId", "movieId", "rating", "timestamp"],
    },
}


def download_movielens(version="100k", data_dir=None):
    """Download MovieLens dataset"""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if version not in DATASET_CONFIGS:
        raise ValueError(
            f"Version {version} not supported. Choose from: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[version]
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_filename = f"ml-{version}.zip"
    zip_path = data_dir / zip_filename
    extract_path = data_dir / config["folder"]

    if not extract_path.exists():
        print(f"Downloading MovieLens {version.upper()}...")
        urllib.request.urlretrieve(config["url"], zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

        zip_path.unlink()
        print("Download complete!")

    return extract_path


def load_movielens(version="100k", data_dir=None, min_rating=4.0):
    """Load and preprocess MovieLens data"""
    config = DATASET_CONFIGS[version]
    extract_path = download_movielens(version, data_dir)

    # Load ratings
    ratings_path = extract_path / config["ratings_file"]
    ratings = pl.read_csv(
        ratings_path, separator=config["separator"], new_columns=config["columns"]
    )

    # Normalize column names for 25m dataset
    if version == "25m":
        ratings = ratings.rename({"userId": "user_id", "movieId": "item_id"})

    # Convert to binary (positive if rating >= min_rating)
    ratings = ratings.with_columns(
        (pl.col("rating") >= min_rating).cast(pl.Int32).alias("rating")
    )

    # Keep only positive interactions for training
    positive_ratings = ratings.filter(pl.col("rating") == 1)

    # Encode users and items
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    user_ids = user_encoder.fit_transform(positive_ratings["user_id"].to_numpy())
    item_ids = item_encoder.fit_transform(positive_ratings["item_id"].to_numpy())
    
    positive_ratings = positive_ratings.with_columns([
        pl.Series("user_id", user_ids),
        pl.Series("item_id", item_ids)
    ])

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    print(f"Loaded {len(positive_ratings)} positive interactions")
    print(f"Users: {num_users}, Items: {num_items}")

    return positive_ratings, num_users, num_items, user_encoder, item_encoder


def get_dataloaders(version="100k", batch_size=1024, num_negatives=4, test_split=0.2):
    """Get train/test dataloaders with negative sampling"""
    interactions, num_users, num_items, user_enc, item_enc = load_movielens(version)

    # Train/test split
    interactions = interactions.sample(fraction=1, shuffle=True)  # Shuffle
    split_idx = int(len(interactions) * (1 - test_split))

    train_interactions = interactions[:split_idx]
    test_interactions = interactions[split_idx:]

    # Create datasets
    train_dataset = MovieLensDataset(
        train_interactions, num_users, num_items, num_negatives
    )
    test_dataset = MovieLensDataset(
        test_interactions, num_users, num_items, num_negatives
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, num_users, num_items


if __name__ == "__main__":
    # Test the module
    train_loader, test_loader, num_users, num_items = get_dataloaders(version="100k")

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Sample batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"User IDs shape: {batch['user_id'].shape}")
    print(f"Item IDs shape: {batch['item_id'].shape}")
    print(f"Ratings shape: {batch['rating'].shape}")
    print(f"Sample ratings: {batch['rating'][:10]}")
