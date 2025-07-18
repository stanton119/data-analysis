"""
PyTorch dataset classes and utilities for multi-task learning.

This module provides:
1. DataPreprocessor class for consistent feature preprocessing
2. Dataset class for tabular data with multiple tasks
3. Utility functions to create dataloaders for training and evaluation
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any


class TabularDataPreprocessor:
    """Preprocessor for tabular data with consistent handling of categorical and numerical features."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.categorical_mappings = {}
        self.one_hot_sizes = {}
        self.feature_mean = None
        self.feature_std = None
        self.categorical_cols = []
        self.numerical_cols = []
        self.feature_cols = []
        self.fitted = False

    def fit(
        self,
        dataframe: pl.DataFrame,
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        normalize_features: bool = True,
    ):
        """
        Fit the preprocessor on training data.

        Args:
            dataframe (pl.DataFrame): DataFrame with features
            feature_cols (List[str], optional): List of feature column names
            categorical_cols (List[str], optional): List of categorical column names
            normalize_features (bool): Whether to normalize numerical features
        """
        self.feature_cols = (
            feature_cols.copy() if feature_cols is not None else list(dataframe.columns)
        )
        self.categorical_cols = categorical_cols or []

        # Identify categorical and numerical columns
        self.numerical_cols = []

        for col in self.feature_cols:
            # Check if column is in categorical_cols list
            if col in self.categorical_cols:
                pass  # Already in categorical_cols
            # Check if column is numeric
            elif dataframe[col].dtype in [
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
                pl.Float32,
                pl.Float64,
            ]:
                self.numerical_cols.append(col)
            # If not explicitly categorical and not numeric, treat as categorical
            else:
                if col not in self.categorical_cols:
                    self.categorical_cols.append(col)

        # Fit categorical features
        for col in self.categorical_cols:
            unique_values = dataframe[col].unique().to_list()
            self.categorical_mappings[col] = {
                val: i for i, val in enumerate(unique_values)
            }
            self.one_hot_sizes[col] = len(unique_values)

        # Fit numerical features
        if self.numerical_cols and normalize_features:
            numerical_features = (
                dataframe.select(self.numerical_cols).to_numpy().astype(float)
            )
            self.feature_mean = numerical_features.mean(axis=0)
            self.feature_std = numerical_features.std(axis=0)
            # Replace zero std with 1 to avoid division by zero
            self.feature_std = np.where(self.feature_std == 0, 1.0, self.feature_std)

        self.fitted = True
        return self

    def transform(self, dataframe: pl.DataFrame) -> torch.Tensor:
        """
        Transform data using fitted preprocessor.

        Args:
            dataframe (pl.DataFrame): DataFrame to transform

        Returns:
            torch.Tensor: Processed features
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Process numerical features
        if self.numerical_cols:
            numerical_features = (
                dataframe.select(self.numerical_cols).to_numpy().astype(float)
            )

            # Normalize if mean and std are available
            if self.feature_mean is not None and self.feature_std is not None:
                numerical_features = (
                    numerical_features - self.feature_mean
                ) / self.feature_std
        else:
            numerical_features = np.empty((len(dataframe), 0))

        # Process categorical features
        encoded_categorical_features = []

        for col in self.categorical_cols:
            # Extract the column as numpy array
            col_values = dataframe[col].to_numpy()

            # Create one-hot encoding
            one_hot = np.zeros((len(col_values), self.one_hot_sizes[col]))
            for i, val in enumerate(col_values):
                # Use the mapping from training, defaulting to 0 for unseen values
                idx = self.categorical_mappings[col].get(val, 0)
                one_hot[i, idx] = 1

            encoded_categorical_features.append(one_hot)

        # Combine numerical and categorical features
        if encoded_categorical_features:
            features = np.concatenate(
                [numerical_features] + encoded_categorical_features, axis=1
            )
        else:
            features = numerical_features

        return torch.tensor(features, dtype=torch.float32)


class MultiTaskTabularDataset(Dataset):
    """PyTorch Dataset for tabular data with multiple tasks."""

    def __init__(
        self,
        dataframe: pl.DataFrame,
        preprocessor: Optional[TabularDataPreprocessor] = None,
        feature_cols: Optional[List[str]] = None,
        task_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        normalize_features: bool = True,
        fit_preprocessor: bool = False,
    ):
        """
        Args:
            dataframe (pl.DataFrame): DataFrame with features and task targets
            preprocessor (TabularDataPreprocessor, optional): Preprocessor for features
            feature_cols (List[str], optional): List of feature column names
            task_cols (List[str], optional): List of task target column names
            categorical_cols (List[str], optional): List of categorical column names
            normalize_features (bool): Whether to normalize numerical features
            fit_preprocessor (bool): Whether to fit the preprocessor on this data
        """
        self.dataframe = dataframe

        # Identify task columns if not provided
        if task_cols is None:
            task_cols = [
                col
                for col in dataframe.columns
                if ("task_" in col or col.endswith("_binary"))
            ]
            if not task_cols:
                raise ValueError("No task columns found. Please specify task_cols.")
        self.task_cols = task_cols

        # Identify feature columns if not provided
        if feature_cols is None:
            feature_cols = [col for col in dataframe.columns if col not in task_cols]
        self.feature_cols = feature_cols.copy()

        # Process features using preprocessor
        if preprocessor is None:
            self.preprocessor = TabularDataPreprocessor()
            fit_preprocessor = True
        else:
            self.preprocessor = preprocessor

        if fit_preprocessor:
            self.preprocessor.fit(
                dataframe,
                feature_cols=self.feature_cols,
                categorical_cols=categorical_cols,
                normalize_features=normalize_features,
            )

        self.features = self.preprocessor.transform(dataframe)

        # Process targets
        self.targets = {
            task: torch.tensor(dataframe[task].to_numpy(), dtype=torch.float32)
            for task in task_cols
        }

        # Store feature and target dimensions
        self.feature_dim = self.features.shape[1]
        self.num_tasks = len(task_cols)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Dict containing features and targets for each task
        """
        sample = {
            "features": self.features[idx],
        }

        # Add targets for each task
        for task in self.task_cols:
            sample[task] = self.targets[task][idx]

        return sample


def create_train_test_dataloaders(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    random_seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Creates training, validation, and test dataloaders from separate train and test dataframes
    with consistent preprocessing.

    Args:
        train_df (pl.DataFrame): The training dataframe
        test_df (pl.DataFrame): The test dataframe
        batch_size (int): The batch size for dataloaders
        val_ratio (float): Proportion of training data to use for validation
        dataset_kwargs (Dict): Additional keyword arguments to pass to the dataset class
        random_seed (int): Random seed for reproducibility

    Returns:
        Dict: A dictionary containing the training, validation, and test dataloaders and preprocessor
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)

    # Create datasets with consistent preprocessing
    dataset_kwargs = dataset_kwargs or {}

    # Create and fit preprocessor on training data
    preprocessor = TabularDataPreprocessor()

    # Create training dataset with preprocessor fitting
    train_full_dataset = MultiTaskTabularDataset(
        train_df, preprocessor=preprocessor, fit_preprocessor=True, **dataset_kwargs
    )

    # Create test dataset using the same preprocessor (no fitting)
    test_dataset = MultiTaskTabularDataset(
        test_df, preprocessor=preprocessor, fit_preprocessor=False, **dataset_kwargs
    )

    # Split training dataset into train and validation
    train_size = int((1 - val_ratio) * len(train_full_dataset))
    val_size = len(train_full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        train_full_dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "preprocessor": preprocessor,
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    from data_sources import create_synthetic_dataset, load_uci_census_dataset

    print("Testing with synthetic multi-task data...")
    # Create synthetic dataset with multiple tasks
    synthetic_data = create_synthetic_dataset(
        num_samples=200, num_features=10, num_tasks=3, task_correlations=[1.0, 0.8, 0.2]
    )

    # Create dataset and dataloaders
    dataset_kwargs = {
        "feature_cols": [f"feature_{i}" for i in range(10)],
        "task_cols": [f"task_{i}" for i in range(3)],
    }

    dataloaders = create_train_test_dataloaders(
        synthetic_data["train"],
        synthetic_data["test"],
        batch_size=32,
        dataset_kwargs=dataset_kwargs,
    )

    print("\nSynthetic data dataloader batch shapes:")
    for split, loader in dataloaders.items():
        if split != "preprocessor":  # Skip preprocessor in the loop
            batch = next(iter(loader))
            print(f"{split}:")
            for key, value in batch.items():
                print(f"  {key}: {value.shape}")

    # Extract preprocessor for later use
    preprocessor = dataloaders["preprocessor"]
    print("\nPreprocessor is fitted:", preprocessor.fitted)

    print("\nTesting with UCI Census dataset...")
    # Load UCI Census dataset
    census_data = load_uci_census_dataset(create_multi_task=True)

    # Define categorical columns
    categorical_cols = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    # Define feature columns (exclude target columns)
    feature_cols = [
        col
        for col in census_data["train"].columns
        if col not in ["income", "income_binary", "marital_status_binary"]
    ]

    # Create dataset and dataloaders
    dataset_kwargs = {
        "feature_cols": feature_cols,
        "task_cols": ["income_binary", "marital_status_binary"],
        "categorical_cols": categorical_cols,
        "normalize_features": True,
    }

    dataloaders = create_train_test_dataloaders(
        census_data["train"],
        census_data["test"],
        batch_size=64,
        dataset_kwargs=dataset_kwargs,
    )

    print("\nUCI Census dataset dataloader batch shapes:")
    for split, loader in dataloaders.items():
        if split != "preprocessor":  # Skip preprocessor in the loop
            batch = next(iter(loader))
            print(f"{split}:")
            for key, value in batch.items():
                print(f"  {key}: {value.shape}")

    # Extract preprocessor for later use
    preprocessor = dataloaders["preprocessor"]

    # INFERENCE EXAMPLE
    print("\n=== Inference Example ===")

    # Simulate new data (using a few samples from test data for demonstration)
    new_data = census_data["test"].head(5)
    print(f"New data shape: {new_data.shape}")

    # Preprocess new data using the same preprocessor (no fitting)
    features = preprocessor.transform(new_data)
    print(f"Processed features shape: {features.shape}")

    # Example of using the preprocessed features with a model
    print("\nSimulating model inference:")
    print("1. Create a model instance")
    print("2. Load trained model weights")
    print("3. Set model to evaluation mode")
    print("4. Run inference:")
    print("   with torch.no_grad():")
    print("       outputs = model(features)")

    # Example of saving and loading the preprocessor
    print("\nExample of saving and loading the preprocessor:")
    print("import pickle")
    print("\n# Save preprocessor")
    print("with open('preprocessor.pkl', 'wb') as f:")
    print("    pickle.dump(preprocessor, f)")
    print("\n# Load preprocessor")
    print("with open('preprocessor.pkl', 'rb') as f:")
    print("    loaded_preprocessor = pickle.load(f)")
    print("\n# Use loaded preprocessor for inference")
    print("features = loaded_preprocessor.transform(new_data)")
