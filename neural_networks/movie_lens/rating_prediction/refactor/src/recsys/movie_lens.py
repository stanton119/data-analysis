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
    def __init__(
        self,
        interactions,
        num_users,
        num_items,
        item_features=None,
        user_features=None,
        num_negatives=4,
    ):
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items
        self.item_features = item_features  # Genre features
        self.user_features = user_features  # Age, gender, occupation
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
            item_id = row["item_id"]
            rating = 1.0
        else:  # Negative sample
            # Sample random item not interacted with by user
            while True:
                item_id = np.random.randint(0, self.num_items)
                if item_id not in self.user_items[user_id]:
                    break
            rating = 0.0

        result = {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "item_id": torch.tensor(item_id, dtype=torch.long),
            "rating": torch.tensor(rating, dtype=torch.float32),
        }

        # Add user features if available
        if self.user_features:
            user_feat_dict = {}
            # Continuous features
            if "continuous" in self.user_features:
                user_feat_dict["continuous"] = torch.tensor(
                    self.user_features["continuous"][user_id], dtype=torch.float32
                )
            # Embedding features
            if "categorical" in self.user_features:
                user_feat_dict["categorical"] = {
                    k: torch.tensor(v[user_id], dtype=torch.long)
                    for k, v in self.user_features["categorical"].items()
                }
            result["user_features"] = user_feat_dict

        # Add item features if available
        if self.item_features:
            item_feat_dict = {}
            if "continuous" in self.item_features:
                item_feat_dict["continuous"] = torch.tensor(
                    self.item_features["continuous"][item_id], dtype=torch.float32
                )
            if "categorical" in self.item_features:
                item_feat_dict["categorical"] = {
                    k: torch.tensor(v[item_id], dtype=torch.long)
                    for k, v in self.item_features["categorical"].items()
                }
            result["item_features"] = item_feat_dict

        return result


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


def load_user_features(version="100k", user_encoder=None, data_dir=None):
    """Load user features (age, gender, occupation) for MovieLens dataset"""
    if version not in ["100k", "1m"]:
        print(f"User features not available for {version}")
        return None, None

    config = DATASET_CONFIGS[version]
    extract_path = download_movielens(version, data_dir)

    if version == "100k":
        # Load u.user file: user_id | age | gender | occupation | zip_code
        users_path = extract_path / "u.user"
        users = pl.read_csv(users_path, separator="|", has_header=False)
        users.columns = ["user_id", "age", "gender", "occupation", "zip_code"]

        # Encode categorical features
        gender_map = {"M": 0, "F": 1}
        users = users.with_columns(
            [
                pl.col("gender").map_elements(
                    lambda x: gender_map.get(x, 0), return_dtype=pl.Int32
                ),
                pl.col("age").cast(pl.Float32) / 100.0,
            ]
        )

        # Handle occupation encoding
        occupation_encoder = LabelEncoder()
        occupation_encoded = occupation_encoder.fit_transform(
            users["occupation"].to_numpy()
        )
        # Encode zip codes for embedding
        zip_encoder = LabelEncoder()
        zip_encoded = zip_encoder.fit_transform(users["zip_code"].to_numpy())

        continuous_features = users[["age", "gender"]].to_numpy()
        categorical_features = {
            "occupation": occupation_encoded,
            "zip_code": zip_encoded,
        }

        dims = {
            "continuous_dim": continuous_features.shape[1],
            "categorical_dims": {
                "occupation": len(occupation_encoder.classes_),
                "zip_code": len(zip_encoder.classes_),
            },
        }

        if user_encoder is not None:
            num_users = len(user_encoder.classes_)

            # Align features with encoded user IDs
            aligned_continuous = np.zeros((num_users, continuous_features.shape[1]))
            aligned_categorical = {
                k: np.zeros(num_users, dtype=int) for k in categorical_features
            }

            orig_user_ids = users["user_id"].to_numpy()
            encoded_user_map = {
                orig_id: enc_id for enc_id, orig_id in enumerate(user_encoder.classes_)
            }

            for i, orig_id in enumerate(orig_user_ids):
                if orig_id in encoded_user_map:
                    enc_id = encoded_user_map[orig_id]
                    aligned_continuous[enc_id] = continuous_features[i]
                    for k, v in categorical_features.items():
                        aligned_categorical[k][enc_id] = v[i]

            features = {
                "continuous": aligned_continuous,
                "categorical": aligned_categorical,
            }
            return features, dims

    return None, None


def load_item_features(version="100k", item_encoder=None, data_dir=None):
    """Load item features (genres) for MovieLens dataset"""
    if version not in ["100k", "1m"]:
        print(f"Genre features not available for {version}")
        return None, None

    config = DATASET_CONFIGS[version]
    extract_path = download_movielens(version, data_dir)

    if version == "100k":
        # Load u.item file
        items_path = extract_path / "u.item"
        items = pl.read_csv(
            items_path, separator="|", has_header=False, encoding="latin1"
        )

        # Extract genre columns (last 19 columns)
        genre_cols = [f"column_{i}" for i in range(6, 25)]
        genres = items.select(genre_cols).to_numpy()

        # Extract release year from date (column_3: release_date)
        release_dates = items["column_3"].to_numpy()
        year_encoder = LabelEncoder()
        # Parse date format like "01-Jan-1995"
        years = [
            int(d.split("-")[-1]) if isinstance(d, str) and "-" in d else 1995
            for d in release_dates
        ]
        years_encoded = year_encoder.fit_transform(years)

        continuous_features = genres
        categorical_features = {"year": years_encoded}

        dims = {
            "continuous_dim": continuous_features.shape[1],
            "categorical_dims": {"year": len(year_encoder.classes_)},
        }

        # Map original item IDs to encoded IDs
        if item_encoder is not None:
            num_items = len(item_encoder.classes_)

            aligned_continuous = np.zeros((num_items, continuous_features.shape[1]))
            aligned_categorical = {"year": np.zeros(num_items, dtype=int)}

            orig_item_ids = items["column_1"].to_numpy()
            encoded_item_map = {
                orig_id: enc_id for enc_id, orig_id in enumerate(item_encoder.classes_)
            }

            for i, orig_id in enumerate(orig_item_ids):
                if orig_id in encoded_item_map:
                    enc_id = encoded_item_map[orig_id]
                    aligned_continuous[enc_id] = continuous_features[i]
                    aligned_categorical["year"][enc_id] = categorical_features["year"][
                        i
                    ]

            features = {
                "continuous": aligned_continuous,
                "categorical": aligned_categorical,
            }
            return features, dims

    return None, None


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

    positive_ratings = positive_ratings.with_columns(
        [pl.Series("user_id", user_ids), pl.Series("item_id", item_ids)]
    )

    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    print(f"Loaded {len(positive_ratings)} positive interactions")
    print(f"Users: {num_users}, Items: {num_items}")

    return positive_ratings, num_users, num_items, user_encoder, item_encoder


def get_dataloaders(
    version="100k",
    batch_size=1024,
    num_negatives=4,
    val_split=0.1,
    test_split=0.1,
    use_features=True,
):
    """Get train/val/test dataloaders with negative sampling (80/10/10 split)"""
    interactions, num_users, num_items, user_enc, item_enc = load_movielens(version)

    user_features, user_feature_dims = None, None
    item_features, item_feature_dims = None, None
    if use_features:
        user_features, user_feature_dims = load_user_features(version, user_enc)
        item_features, item_feature_dims = load_item_features(version, item_enc)
        if user_features:
            print(f"Loaded user features with dims: {user_feature_dims}")
        if item_features:
            print(f"Loaded item features with dims: {item_feature_dims}")

    # Train/val/test split (80/10/10)
    interactions = interactions.sample(fraction=1, seed=42)
    val_idx = int(len(interactions) * (1 - val_split - test_split))
    test_idx = int(len(interactions) * (1 - test_split))

    train_interactions = interactions[:val_idx]
    val_interactions = interactions[val_idx:test_idx]
    test_interactions = interactions[test_idx:]

    # Create datasets
    train_dataset = MovieLensDataset(
        train_interactions,
        num_users,
        num_items,
        item_features,
        user_features,
        num_negatives,
    )
    val_dataset = MovieLensDataset(
        val_interactions,
        num_users,
        num_items,
        item_features,
        user_features,
        num_negatives,
    )
    test_dataset = MovieLensDataset(
        test_interactions,
        num_users,
        num_items,
        item_features,
        user_features,
        num_negatives,
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        num_users,
        num_items,
        user_feature_dims,
        item_feature_dims,
    )


if __name__ == "__main__":
    # Test the module
    (
        train_loader,
        val_loader,
        test_loader,
        num_users,
        num_items,
        user_dims,
        item_dims,
    ) = get_dataloaders(version="100k")

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"User feature dimensions: {user_dims}")
    print(f"Item feature dimensions: {item_dims}")

    # Sample batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"User IDs shape: {batch['user_id'].shape}")
    if "user_features" in batch:
        if "continuous" in batch["user_features"]:
            print(
                f"User continuous features shape: {batch['user_features']['continuous'].shape}"
            )
        if "categorical" in batch["user_features"]:
            for k, v in batch["user_features"]["categorical"].items():
                print(f"User categorical '{k}' shape: {v.shape}")
    print(f"Sample ratings: {batch['rating'][:10]}")
