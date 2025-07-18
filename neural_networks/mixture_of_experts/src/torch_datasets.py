"""
PyTorch dataset classes and utilities for multi-task learning.

This module provides:
1. Dataset class for tabular data with multiple tasks
2. Utility functions to create dataloaders for training and evaluation
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any


class MultiTaskTabularDataset(Dataset):
    """PyTorch Dataset for tabular data with multiple tasks."""

    def __init__(
        self,
        dataframe: pl.DataFrame,
        feature_cols: Optional[List[str]] = None,
        task_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        normalize_features: bool = True,
    ):
        """
        Args:
            dataframe (pl.DataFrame): DataFrame with features and task targets
            feature_cols (List[str], optional): List of feature column names. If None, all columns
                                               except task_cols will be used as features.
            task_cols (List[str], optional): List of task target column names. If None, columns
                                            with names containing 'task_' or ending with '_binary'
                                            will be used as targets.
            categorical_cols (List[str], optional): List of categorical column names to be one-hot encoded.
            normalize_features (bool): Whether to normalize numerical features
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
        self.feature_cols = (
            feature_cols.copy()
        )  # Make a copy to avoid modifying the original

        # Process categorical features if any
        self.categorical_cols = categorical_cols or []
        self.categorical_mappings = {}
        self.one_hot_sizes = {}

        # Process features
        self.features = self._process_features(normalize_features)

        # Process targets
        self.targets = {
            task: torch.tensor(dataframe[task].to_numpy(), dtype=torch.float32)
            for task in task_cols
        }

        # Store feature and target dimensions
        self.feature_dim = self.features.shape[1]
        self.num_tasks = len(task_cols)

    def _process_features(self, normalize: bool) -> torch.Tensor:
        """Process features including categorical encoding and normalization."""
        df = self.dataframe.clone()
        numerical_feature_cols = []
        categorical_feature_cols = []

        # First, identify which columns are categorical and which are numerical
        for col in self.feature_cols:
            # Check if column is in categorical_cols list
            if col in self.categorical_cols:
                categorical_feature_cols.append(col)
            # Check if column is numeric
            elif df[col].dtype in [
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
                numerical_feature_cols.append(col)
            # If not explicitly categorical and not numeric, treat as categorical
            else:
                categorical_feature_cols.append(col)
                if col not in self.categorical_cols:
                    self.categorical_cols.append(col)

        # Process categorical features
        encoded_categorical_features = []

        for col in categorical_feature_cols:
            # Create mapping for categorical values
            unique_values = df[col].unique().to_list()
            self.categorical_mappings[col] = {
                val: i for i, val in enumerate(unique_values)
            }
            self.one_hot_sizes[col] = len(unique_values)

            # Extract the column as numpy array
            col_values = df[col].to_numpy()

            # Create one-hot encoding
            one_hot = np.zeros((len(col_values), self.one_hot_sizes[col]))
            for i, val in enumerate(col_values):
                one_hot[i, self.categorical_mappings[col].get(val, 0)] = 1

            encoded_categorical_features.append(one_hot)

        # Process numerical features
        if numerical_feature_cols:
            numerical_features = (
                df.select(numerical_feature_cols).to_numpy().astype(float)
            )

            # Normalize numerical features if requested
            if normalize:
                # Calculate mean and std for each feature
                self.feature_mean = numerical_features.mean(axis=0)
                self.feature_std = numerical_features.std(axis=0)

                # Replace zero std with 1 to avoid division by zero
                self.feature_std = np.where(
                    self.feature_std == 0, 1.0, self.feature_std
                )

                # Normalize
                numerical_features = (
                    numerical_features - self.feature_mean
                ) / self.feature_std
        else:
            numerical_features = np.empty((len(df), 0))

        # Combine numerical and categorical features
        if encoded_categorical_features:
            features = np.concatenate(
                [numerical_features] + encoded_categorical_features, axis=1
            )
        else:
            features = numerical_features

        return torch.tensor(features, dtype=torch.float32)

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
    Creates training, validation, and test dataloaders from separate train and test dataframes.

    Args:
        train_df (pl.DataFrame): The training dataframe
        test_df (pl.DataFrame): The test dataframe
        batch_size (int): The batch size for dataloaders
        val_ratio (float): Proportion of training data to use for validation
        dataset_kwargs (Dict): Additional keyword arguments to pass to the dataset class
        random_seed (int): Random seed for reproducibility

    Returns:
        Dict: A dictionary containing the training, validation, and test dataloaders
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)

    # Create datasets
    dataset_kwargs = dataset_kwargs or {}
    train_full_dataset = MultiTaskTabularDataset(train_df, **dataset_kwargs)
    test_dataset = MultiTaskTabularDataset(test_df, **dataset_kwargs)

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

    return {"train": train_loader, "val": val_loader, "test": test_loader}


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
        batch = next(iter(loader))
        print(f"{split}:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")

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
        batch = next(iter(loader))
        print(f"{split}:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")
