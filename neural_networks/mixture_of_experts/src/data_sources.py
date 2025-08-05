"""
Data loader module for loading and creating experiment data.

This module provides functions to:
1. Load the UCI Census Income dataset
2. Create synthetic multi-task datasets with controlled correlation levels
3. Generate simulated experiment data as described in the MMoE paper
"""

import os
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union


def load_uci_census_dataset(
    data_dir: str = "data/census_income",
    remove_unknowns: bool = True,
    binary_income: bool = True,
    create_multi_task: bool = True
) -> Dict[str, pl.DataFrame]:
    """
    Load the UCI Census Income dataset and return as polars dataframes.
    
    The dataset is used in the MMoE paper for multi-task learning with two tasks:
    1. Income prediction (>50K or <=50K)
    2. Marital status prediction (married or not married)
    
    Args:
        data_dir: Directory containing the UCI Census Income dataset files
        remove_unknowns: Whether to remove rows with unknown values (?)
        binary_income: Whether to convert income to binary (1 for >50K, 0 for <=50K)
        create_multi_task: Whether to create a second task (marital status prediction)
        
    Returns:
        Dictionary containing 'train' and 'test' polars dataframes
    """
    # Define column names based on the adult.names file
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    
    # Get absolute path to data directory
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = base_dir / data_dir
    
    # Load training data
    train_path = data_path / "adult.data"
    train_df = pl.read_csv(
        train_path,
        has_header=False,
        separator=",",
        skip_rows=0,
        new_columns=column_names,
        truncate_ragged_lines=True
    )
    
    # Load test data (skip the first line which contains metadata)
    test_path = data_path / "adult.test"
    test_df = pl.read_csv(
        test_path,
        has_header=False,
        separator=",",
        skip_rows=1,
        new_columns=column_names,
        truncate_ragged_lines=True
    )
    
    # Clean the data
    train_df = clean_census_data(train_df)
    test_df = clean_census_data(test_df)
    
    # Remove rows with unknown values if specified
    if remove_unknowns:
        # Filter out rows with "?" in any string column
        for col in train_df.columns:
            if train_df[col].dtype == pl.Utf8:
                train_df = train_df.filter(~pl.col(col).str.contains("\\?"))
                test_df = test_df.filter(~pl.col(col).str.contains("\\?"))
    
    # Create binary income target if specified
    if binary_income:
        train_df = train_df.with_columns(
            pl.col("income").str.contains(">50K").cast(pl.Int8).alias("income_binary")
        )
        test_df = test_df.with_columns(
            pl.col("income").str.contains(">50K").cast(pl.Int8).alias("income_binary")
        )
    
    # Create marital status target if specified (for multi-task learning)
    if create_multi_task:
        # Create binary marital status target (1 for married, 0 for not married)
        train_df = train_df.with_columns(
            pl.col("marital_status").str.starts_with("Married").cast(pl.Int8).alias("marital_status_binary")
        )
        test_df = test_df.with_columns(
            pl.col("marital_status").str.starts_with("Married").cast(pl.Int8).alias("marital_status_binary")
        )
    
    return {"train": train_df, "test": test_df}


def clean_census_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean the UCI Census Income dataset.
    
    Args:
        df: Input dataframe
        
    Returns:
        Cleaned dataframe
    """
    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).str.strip_chars())
    
    # Remove period from income in test set (e.g., ">50K." -> ">50K")
    df = df.with_columns(pl.col("income").str.replace_all("\\.", ""))
    
    return df


def create_synthetic_dataset(
    num_samples: int = 10000,
    num_features: int = 10,
    num_tasks: int = 2,
    task_correlations: Optional[List[float]] = None,
    test_size: float = 0.2,
    random_seed: int = 42
) -> Dict[str, pl.DataFrame]:
    """
    Create a synthetic multi-task dataset with controlled correlation levels between tasks.
    
    Args:
        num_samples: Number of samples to generate
        num_features: Number of features to generate
        num_tasks: Number of tasks (target variables)
        task_correlations: List of correlation coefficients between task 0 and other tasks
                          (if None, defaults to [1.0, 0.8, 0.5, 0.2, 0.0, -0.2, -0.5, -0.8])
        test_size: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing 'train' and 'test' polars dataframes
    """
    np.random.seed(random_seed)
    
    # Default correlations if not provided
    if task_correlations is None:
        task_correlations = [1.0, 0.8, 0.5, 0.2, 0.0, -0.2, -0.5, -0.8][:num_tasks]
    
    # Ensure we have enough correlations for all tasks
    if len(task_correlations) < num_tasks:
        raise ValueError(f"Not enough correlations provided. Need {num_tasks}, got {len(task_correlations)}")
    
    # Generate features (standard normal distribution)
    X = np.random.randn(num_samples, num_features)
    
    # Generate task weights (different for each task)
    task_weights = [np.random.randn(num_features) for _ in range(num_tasks)]
    
    # Generate targets with controlled correlations
    y = np.zeros((num_samples, num_tasks))
    
    # First task is just a linear combination of features with noise
    y[:, 0] = X @ task_weights[0] + 0.1 * np.random.randn(num_samples)
    
    # Generate other tasks with controlled correlation to the first task
    for i in range(1, num_tasks):
        # Calculate how much of task 0 to mix in to achieve desired correlation
        task_i_raw = X @ task_weights[i] + 0.1 * np.random.randn(num_samples)
        
        # Standardize both signals
        task_0_std = (y[:, 0] - np.mean(y[:, 0])) / np.std(y[:, 0])
        task_i_std = (task_i_raw - np.mean(task_i_raw)) / np.std(task_i_raw)
        
        # Mix the signals to achieve desired correlation
        correlation = task_correlations[i]
        alpha = correlation  # Mixing coefficient
        
        # Create correlated signal
        y[:, i] = alpha * task_0_std + np.sqrt(1 - alpha**2) * task_i_std
    
    # Convert to binary classification tasks
    y_binary = (y > 0).astype(np.int32)
    
    # Split into train and test sets
    test_idx = int(num_samples * (1 - test_size))
    X_train, X_test = X[:test_idx], X[test_idx:]
    y_train, y_test = y_binary[:test_idx], y_binary[test_idx:]
    
    # Create column names
    feature_cols = [f"feature_{i}" for i in range(num_features)]
    target_cols = [f"task_{i}" for i in range(num_tasks)]
    
    # Create polars dataframes
    train_df = pl.DataFrame(
        {**{feature_cols[i]: X_train[:, i] for i in range(num_features)},
         **{target_cols[i]: y_train[:, i] for i in range(num_tasks)}}
    )
    
    test_df = pl.DataFrame(
        {**{feature_cols[i]: X_test[:, i] for i in range(num_features)},
         **{target_cols[i]: y_test[:, i] for i in range(num_tasks)}}
    )
    
    return {"train": train_df, "test": test_df}


def generate_mmoe_synthetic_data(
    num_samples: int = 10000,
    num_features: int = 100,
    task_correlation: float = 0.5,
    num_sinusoidal_components: int = 5,
    test_size: float = 0.2,
    random_seed: int = 42,
    regression: bool = False
) -> Dict[str, Union[pl.DataFrame, Dict, np.ndarray]]:
    """
    Generate simulated experiment data as described in the MMoE paper.
    
    This function implements the synthetic data generation process from the paper:
    "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts"
    
    The data generation process is as follows:
    1. Generate two orthogonal unit vectors u1, u2 in d-dimensional space
    2. Create weight vectors w1, w2 with controlled correlation
    3. Generate input data points x randomly
    4. Generate two regression targets y1, y2 using sinusoidal functions
    5. Convert regression targets to binary classification tasks (unless regression=True)
    
    Args:
        num_samples: Number of samples to generate
        num_features: Dimension of the feature space
        task_correlation: Cosine similarity between weight vectors (controls task relatedness)
        num_sinusoidal_components: Number of sinusoidal components to add
        test_size: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
        regression: If True, return regression targets instead of binary classification
        
    Returns:
        Dictionary containing 'train' and 'test' polars dataframes
    """
    np.random.seed(random_seed)
    
    # Step 1: Generate two orthogonal unit vectors
    u1 = np.random.randn(num_features)
    u1 = u1 / np.linalg.norm(u1)  # Normalize to unit vector
    
    # Generate a random vector
    u2_temp = np.random.randn(num_features)
    # Make u2 orthogonal to u1 using Gram-Schmidt process
    u2 = u2_temp - np.dot(u2_temp, u1) * u1
    u2 = u2 / np.linalg.norm(u2)  # Normalize to unit vector
    
    # Verify orthogonality
    assert np.abs(np.dot(u1, u2)) < 1e-10, "Vectors are not orthogonal"
    
    # Step 2: Generate weight vectors with controlled correlation
    c = 1.0  # Scale constant
    w1 = c * u1
    w2 = c * (task_correlation * u1 + np.sqrt(1 - task_correlation**2) * u2)
    
    # Verify the cosine similarity is as expected
    cos_sim = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
    assert np.abs(cos_sim - task_correlation) < 1e-10, f"Expected correlation {task_correlation}, got {cos_sim}"
    
    # Step 3: Generate input data
    X = np.random.randn(num_samples, num_features)
    
    # Step 4: Generate sinusoidal parameters
    alphas = np.random.uniform(0.5, 2.0, num_sinusoidal_components)
    betas = np.random.uniform(0, 2*np.pi, num_sinusoidal_components)
    
    # Step 5: Generate regression targets with sinusoidal components
    y1_reg = np.dot(X, w1)
    y2_reg = np.dot(X, w2)
    
    for i in range(num_sinusoidal_components):
        y1_reg += np.sin(alphas[i] * np.dot(X, w1) + betas[i])
        y2_reg += np.sin(alphas[i] * np.dot(X, w2) + betas[i])
    
    # Add small Gaussian noise
    y1_reg += 0.1 * np.random.randn(num_samples)
    y2_reg += 0.1 * np.random.randn(num_samples)
    
    # Calculate the Pearson correlation between the regression targets
    reg_correlation = np.corrcoef(y1_reg, y2_reg)[0, 1]
    
    if regression:
        # Use regression targets directly
        y1 = y1_reg
        y2 = y2_reg
        label_correlation = reg_correlation
    else:
        # Convert to binary classification tasks
        y1 = (y1_reg > 0).astype(np.int32)
        y2 = (y2_reg > 0).astype(np.int32)
        
        # Calculate the actual Pearson correlation between the binary labels
        label_correlation = np.corrcoef(y1, y2)[0, 1]
    
    # Split into train and test sets
    test_idx = int(num_samples * (1 - test_size))
    X_train, X_test = X[:test_idx], X[test_idx:]
    y1_train, y1_test = y1[:test_idx], y1[test_idx:]
    y2_train, y2_test = y2[:test_idx], y2[test_idx:]
    
    # Create column names
    feature_cols = [f"feature_{i}" for i in range(num_features)]
    
    # Create polars dataframes
    train_df = pl.DataFrame(
        {**{feature_cols[i]: X_train[:, i] for i in range(num_features)},
         "task_0": y1_train,
         "task_1": y2_train}
    )
    
    test_df = pl.DataFrame(
        {**{feature_cols[i]: X_test[:, i] for i in range(num_features)},
         "task_0": y1_test,
         "task_1": y2_test}
    )
    
    # Add metadata about the generation process
    metadata = {
        "task_correlation_input": task_correlation,
        "label_correlation_actual": label_correlation,
        "regression_correlation": reg_correlation,
        "num_features": num_features,
        "num_samples": num_samples,
        "num_sinusoidal_components": num_sinusoidal_components,
        "is_regression": regression
    }
    
    return {
        "train": train_df, 
        "test": test_df, 
        "metadata": metadata,
        "weights": {"w1": w1, "w2": w2}
    }


def generate_correlation_experiment_datasets(
    correlations: List[float] = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0, -0.1, -0.3, -0.5],
    num_samples: int = 10000,
    num_features: int = 100,
    num_sinusoidal_components: int = 5,
    test_size: float = 0.2,
    random_seed: int = 42,
    regression: bool = False
) -> Dict[float, Dict[str, Union[pl.DataFrame, Dict, np.ndarray]]]:
    """
    Generate multiple datasets with different correlation levels for comparing multi-task architectures.
    
    This function creates a series of datasets with varying task correlations to evaluate
    how different multi-task learning approaches perform under different levels of task relatedness.
    
    Args:
        correlations: List of correlation values to generate datasets for
        num_samples: Number of samples per dataset
        num_features: Number of features per dataset
        num_sinusoidal_components: Number of sinusoidal components to add
        test_size: Fraction of data to use for testing
        random_seed: Base random seed (will be incremented for each dataset)
        regression: If True, generate regression targets instead of binary classification
        
    Returns:
        Dictionary mapping correlation values to dataset dictionaries
    """
    datasets = {}
    
    for i, correlation in enumerate(correlations):
        # Use a different seed for each dataset to ensure independence
        dataset_seed = random_seed + i
        
        # Generate dataset with the specified correlation
        dataset = generate_mmoe_synthetic_data(
            num_samples=num_samples,
            num_features=num_features,
            task_correlation=correlation,
            num_sinusoidal_components=num_sinusoidal_components,
            test_size=test_size,
            random_seed=dataset_seed,
            regression=regression
        )
        
        datasets[correlation] = dataset
    
    return datasets


def save_synthetic_dataset(
    dataset: Dict[str, Union[pl.DataFrame, Dict, np.ndarray]],
    output_dir: str,
    dataset_name: str
) -> None:
    """
    Save a synthetic dataset to disk.
    
    Args:
        dataset: Dataset dictionary containing 'train', 'test', 'metadata', and 'weights'
        output_dir: Directory to save the dataset to
        dataset_name: Name of the dataset (will be used as a subdirectory)
    """
    # Create output directory if it doesn't exist
    dataset_dir = Path(output_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train and test dataframes
    dataset["train"].write_parquet(dataset_dir / "train.parquet")
    dataset["test"].write_parquet(dataset_dir / "test.parquet")
    
    # Save metadata as JSON
    import json
    with open(dataset_dir / "metadata.json", "w") as f:
        # Convert numpy values to Python types for JSON serialization
        metadata = {k: float(v) if isinstance(v, np.number) else v 
                   for k, v in dataset["metadata"].items()}
        json.dump(metadata, f, indent=2)
    
    # Save weights as numpy arrays
    for weight_name, weight_array in dataset["weights"].items():
        np.save(dataset_dir / f"{weight_name}.npy", weight_array)


def load_synthetic_dataset(
    dataset_dir: str
) -> Dict[str, Union[pl.DataFrame, Dict, np.ndarray]]:
    """
    Load a synthetic dataset from disk.
    
    Args:
        dataset_dir: Directory containing the dataset files
        
    Returns:
        Dataset dictionary containing 'train', 'test', 'metadata', and 'weights'
    """
    dataset_path = Path(dataset_dir)
    
    # Load train and test dataframes
    train_df = pl.read_parquet(dataset_path / "train.parquet")
    test_df = pl.read_parquet(dataset_path / "test.parquet")
    
    # Load metadata
    import json
    with open(dataset_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load weights
    weights = {}
    for weight_file in dataset_path.glob("*.npy"):
        weight_name = weight_file.stem
        weights[weight_name] = np.load(weight_file)
    
    return {
        "train": train_df,
        "test": test_df,
        "metadata": metadata,
        "weights": weights
    }


def save_correlation_experiment_datasets(
    datasets: Dict[float, Dict[str, Union[pl.DataFrame, Dict, np.ndarray]]],
    output_dir: str,
    experiment_name: str = "correlation_experiment"
) -> None:
    """
    Save multiple datasets with different correlation levels.
    
    Args:
        datasets: Dictionary mapping correlation values to dataset dictionaries
        output_dir: Directory to save the datasets to
        experiment_name: Name of the experiment (will be used as a subdirectory)
    """
    experiment_dir = Path(output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    for correlation, dataset in datasets.items():
        # Format correlation as a string for the directory name
        corr_str = f"corr_{correlation:.2f}".replace("-", "neg_").replace(".", "_")
        save_synthetic_dataset(dataset, experiment_dir, corr_str)


def load_correlation_experiment_datasets(
    experiment_dir: str
) -> Dict[float, Dict[str, Union[pl.DataFrame, Dict, np.ndarray]]]:
    """
    Load multiple datasets with different correlation levels.
    
    Args:
        experiment_dir: Directory containing the experiment datasets
        
    Returns:
        Dictionary mapping correlation values to dataset dictionaries
    """
    experiment_path = Path(experiment_dir)
    datasets = {}
    
    for dataset_dir in experiment_path.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name.startswith("corr_"):
            # Extract correlation value from directory name
            corr_str = dataset_dir.name.replace("corr_", "").replace("neg_", "-").replace("_", ".")
            correlation = float(corr_str)
            
            # Load dataset
            dataset = load_synthetic_dataset(dataset_dir)
            datasets[correlation] = dataset
    
    return datasets


if __name__ == "__main__":
    # Example usage
    print("Loading UCI Census Income dataset...")
    census_data = load_uci_census_dataset()
    
    print("\nTrain data shape:", census_data["train"].shape)
    print("Test data shape:", census_data["test"].shape)
    
    print("\nTrain data schema:")
    print(census_data["train"].schema)
    
    print("\nTrain data sample:")
    print(census_data["train"].head(5))
    
    print("\nCreating synthetic dataset...")
    synthetic_data = create_synthetic_dataset(
        num_samples=5000,
        num_features=10,
        num_tasks=3,
        task_correlations=[1.0, 0.8, 0.2]
    )
    
    print("\nSynthetic train data shape:", synthetic_data["train"].shape)
    print("Synthetic test data shape:", synthetic_data["test"].shape)
    
    print("\nSynthetic train data sample:")
    print(synthetic_data["train"].head(5))
    
    print("\nGenerating MMoE synthetic data...")
    mmoe_data = generate_mmoe_synthetic_data(
        num_samples=5000,
        num_features=100,
        task_correlation=0.5
    )
    
    print("\nMMoE synthetic train data shape:", mmoe_data["train"].shape)
    print("MMoE synthetic test data shape:", mmoe_data["test"].shape)
    print(f"Input task correlation: {mmoe_data['metadata']['task_correlation_input']}")
    print(f"Actual label correlation: {mmoe_data['metadata']['label_correlation_actual']}")
    
    print("\nMMoE synthetic train data sample:")
    print(mmoe_data["train"].head(5))
    
    print("\nGenerating correlation experiment datasets...")
    # Generate a smaller set for the example
    correlation_datasets = generate_correlation_experiment_datasets(
        correlations=[0.9, 0.5, 0.1],
        num_samples=1000,
        num_features=20
    )
    
    for corr, dataset in correlation_datasets.items():
        print(f"\nCorrelation {corr}:")
        print(f"  Input correlation: {corr}")
        print(f"  Actual label correlation: {dataset['metadata']['label_correlation_actual']}")
        print(f"  Train shape: {dataset['train'].shape}")
        print(f"  Test shape: {dataset['test'].shape}")
    
    # Example of saving and loading datasets
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nSaving datasets to {temp_dir}...")
        save_correlation_experiment_datasets(correlation_datasets, temp_dir)
        
        print("\nLoading datasets...")
        loaded_datasets = load_correlation_experiment_datasets(Path(temp_dir) / "correlation_experiment")
        
        print("\nVerifying loaded datasets...")
        for corr, dataset in loaded_datasets.items():
            print(f"\nCorrelation {corr}:")
            print(f"  Input correlation: {dataset['metadata']['task_correlation_input']}")
            print(f"  Actual label correlation: {dataset['metadata']['label_correlation_actual']}")
            print(f"  Train shape: {dataset['train'].shape}")
            print(f"  Test shape: {dataset['test'].shape}")
            
    # Generate regression datasets example
    print("\nGenerating regression datasets...")
    regression_datasets = generate_correlation_experiment_datasets(
        correlations=[0.9, 0.5, 0.1],
        num_samples=1000,
        num_features=20,
        regression=True
    )
    
    for corr, dataset in regression_datasets.items():
        print(f"\nCorrelation {corr} (Regression):")
        print(f"  Input correlation: {corr}")
        print(f"  Actual correlation: {dataset['metadata']['label_correlation_actual']}")
        print(f"  Train shape: {dataset['train'].shape}")
        print(f"  Test shape: {dataset['test'].shape}")
