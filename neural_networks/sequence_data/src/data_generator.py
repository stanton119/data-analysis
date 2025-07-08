import polars as pl
import numpy as np

def generate_dummy_data(num_samples=1000, sequence_length=50, num_features=10):
    """
    Generates a dummy dataset with tabular features, sequence data, and a target variable.

    Args:
        num_samples (int): The number of samples to generate.
        sequence_length (int): The length of the sequences.
        num_features (int): The number of tabular features.

    Returns:
        pl.DataFrame: A DataFrame with the generated data.
    """
    # Generate tabular data
    tabular_data = pl.DataFrame(np.random.rand(num_samples, num_features),
                                schema=[f'feature_{i}' for i in range(num_features)])

    # Generate sequence data
    sequences = [np.random.rand(sequence_length).tolist() for _ in range(num_samples)]
    tabular_data = tabular_data.with_columns(pl.Series("sequence", sequences))

    # Generate target variable
    # The target is a combination of the mean of the tabular features and the sum of the sequence
    feature_cols = [f'feature_{i}' for i in range(num_features)]
    tabular_data = tabular_data.with_columns(
        (pl.mean_horizontal(feature_cols) + pl.col("sequence").list.sum()).alias("target")
    )

    return tabular_data

if __name__ == '__main__':
    data = generate_dummy_data()
    print(data.head())
