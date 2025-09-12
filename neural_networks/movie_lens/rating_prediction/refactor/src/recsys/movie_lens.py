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
    def __init__(self, interactions, num_users, num_items, item_features=None, user_features=None, num_negatives=4):
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
        else:  # Negative sample
            # Sample random item not interacted with by user
            while True:
                item_id = np.random.randint(0, self.num_items)
                if item_id not in self.user_items[user_id]:
                    break

        result = {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "item_id": torch.tensor(item_id, dtype=torch.long),
            "rating": torch.tensor(1.0 if sample_type == 0 else 0.0, dtype=torch.float32),
        }
        
        # Add user features if available
        if self.user_features is not None:
            # Split continuous and embedding features
            result["user_continuous"] = torch.tensor(self.user_features[user_id][:2], dtype=torch.float32)  # age, gender
            result["user_occupation"] = torch.tensor(self.user_features[user_id][2], dtype=torch.long)  # occupation_id
            if self.user_features.shape[1] > 3:  # has zip_code
                result["user_zip"] = torch.tensor(self.user_features[user_id][3], dtype=torch.long)
        
        # Add item features if available
        if self.item_features is not None:
            result["item_genres"] = torch.tensor(self.item_features[item_id][:19], dtype=torch.float32)  # 19 genres
            if self.item_features.shape[1] > 19:  # has release_year
                result["item_year"] = torch.tensor(self.item_features[item_id][19], dtype=torch.long)
            
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


def load_user_features(version="100k", user_encoder=None, data_dir=None, use_occupation_embedding=False, use_zip_embedding=False):
    """Load user features (age, gender, occupation) for MovieLens dataset"""
    if version not in ["100k", "1m"]:
        print(f"User features not available for {version}")
        return None
        
    config = DATASET_CONFIGS[version]
    extract_path = download_movielens(version, data_dir)
    
    if version == "100k":
        # Load u.user file: user_id | age | gender | occupation | zip_code
        users_path = extract_path / "u.user"
        users = pl.read_csv(users_path, separator="|", has_header=False)
        users.columns = ["user_id", "age", "gender", "occupation", "zip_code"]
        
        # Encode categorical features
        gender_map = {"M": 0, "F": 1}
        users = users.with_columns([
            pl.col("gender").map_elements(lambda x: gender_map.get(x, 0), return_dtype=pl.Int32),
            pl.col("age").cast(pl.Float32) / 100.0,  # Normalize age
        ])
        
        # Handle occupation encoding
        occupation_encoder = LabelEncoder()
        occupation_encoded = occupation_encoder.fit_transform(users["occupation"].to_numpy())
        
        features = [users["age"].to_numpy(), users["gender"].to_numpy()]
        
        if use_occupation_embedding:
            features.append(occupation_encoded.reshape(-1, 1))  # Single column for embedding
        else:
            # One-hot encode occupation
            occupation_onehot = np.eye(len(occupation_encoder.classes_))[occupation_encoded]
            features.append(occupation_onehot)
            
        if use_zip_embedding:
            # Encode zip codes for embedding
            zip_encoder = LabelEncoder()
            zip_encoded = zip_encoder.fit_transform(users["zip_code"].to_numpy())
            features.append(zip_encoded.reshape(-1, 1))
        
        user_features = np.column_stack(features)
        
        # Map to encoded user IDs
        if user_encoder is not None:
            encoded_features = np.zeros((len(user_encoder.classes_), user_features.shape[1]))
            for i, orig_id in enumerate(user_encoder.classes_):
                if orig_id <= len(user_features):
                    encoded_features[i] = user_features[orig_id - 1]
            return encoded_features
            
    return None


def load_item_features(version="100k", item_encoder=None, data_dir=None, use_year_embedding=False):
    """Load item features (genres) for MovieLens dataset"""
    if version not in ["100k", "1m"]:
        print(f"Genre features not available for {version}")
        return None
        
    config = DATASET_CONFIGS[version]
    extract_path = download_movielens(version, data_dir)
    
    if version == "100k":
        # Load u.item file
        items_path = extract_path / "u.item"
        items = pl.read_csv(items_path, separator="|", has_header=False, encoding="latin1")
        
        features = []
        
        # Extract genre columns (last 19 columns)
        genre_cols = [f"column_{i}" for i in range(6, 25)]
        genres = items.select([pl.col(col) for col in genre_cols]).to_numpy()
        features.append(genres)
        
        if use_year_embedding:
            # Extract release year from date (column_3: release_date)
            release_dates = items.select(pl.col("column_3")).to_numpy().flatten()
            years = []
            for date_str in release_dates:
                try:
                    # Parse date format like "01-Jan-1995"
                    year = int(date_str.split('-')[-1]) if date_str else 1995
                    years.append(year - 1900)  # Normalize to start from 0
                except:
                    years.append(95)  # Default to 1995
            features.append(np.array(years).reshape(-1, 1))
        
        item_features = np.column_stack(features)
        
        # Map original item IDs to encoded IDs
        if item_encoder is not None:
            encoded_features = np.zeros((len(item_encoder.classes_), item_features.shape[1]))
            item_ids = items.select(pl.col("column_1")).to_numpy().flatten()
            for i, orig_id in enumerate(item_encoder.classes_):
                idx = np.where(item_ids == orig_id)[0]
                if len(idx) > 0:
                    encoded_features[i] = item_features[idx[0]]
            return encoded_features
            
    return None


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
    version="100k", batch_size=1024, num_negatives=4, val_split=0.1, test_split=0.1, use_features=True
):
    """Get train/val/test dataloaders with negative sampling (80/10/10 split)"""
    interactions, num_users, num_items, user_enc, item_enc = load_movielens(version)
    
    # Load features if requested
    item_features = user_features = None
    if use_features:
        item_features = load_item_features(version, item_enc, data_dir=None, use_year_embedding=True)
        user_features = load_user_features(version, user_enc, data_dir=None, use_occupation_embedding=True, use_zip_embedding=True)
        if item_features is not None:
            print(f"Loaded item features with shape: {item_features.shape}")
        if user_features is not None:
            print(f"Loaded user features with shape: {user_features.shape}")

    # Train/val/test split (80/10/10)
    interactions = interactions.sample(fraction=1, seed=42)
    val_idx = int(len(interactions) * (1 - val_split - test_split))
    test_idx = int(len(interactions) * (1 - test_split))

    train_interactions = interactions[:val_idx]
    val_interactions = interactions[val_idx:test_idx]
    test_interactions = interactions[test_idx:]

    # Create datasets
    train_dataset = MovieLensDataset(
        train_interactions, num_users, num_items, item_features, user_features, num_negatives
    )
    val_dataset = MovieLensDataset(
        val_interactions, num_users, num_items, item_features, user_features, num_negatives
    )
    test_dataset = MovieLensDataset(
        test_interactions, num_users, num_items, item_features, user_features, num_negatives
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_users, num_items


if __name__ == "__main__":
    # Test the module
    train_loader, val_loader, test_loader, num_users, num_items = get_dataloaders(version="100k")

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Sample batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"User IDs shape: {batch['user_id'].shape}")
    print(f"Item IDs shape: {batch['item_id'].shape}")
    print(f"Ratings shape: {batch['rating'].shape}")
    if 'user_features' in batch:
        print(f"User features shape: {batch['user_features'].shape}")
    if 'item_features' in batch:
        print(f"Item features shape: {batch['item_features'].shape}")
    print(f"Sample ratings: {batch['rating'][:10]}")
