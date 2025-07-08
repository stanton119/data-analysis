import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np

class TabularSequenceDataset(Dataset):
    """PyTorch Dataset for tabular and sequence data."""

    def __init__(self, dataframe):
        """
        Args:
            dataframe (pl.DataFrame): DataFrame with tabular features, sequences, and target.
        """
        self.dataframe = dataframe
        self.tabular_features = torch.tensor(dataframe.select(pl.exclude(["sequence", "target"])).to_numpy(), dtype=torch.float32)
        self.sequences = torch.tensor(np.array(dataframe["sequence"].to_list()), dtype=torch.float32)
        self.targets = torch.tensor(dataframe["target"].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return {
            "tabular": self.tabular_features[idx],
            "sequence": self.sequences[idx],
            "target": self.targets[idx]
        }

def create_dataloaders(dataframe, batch_size=32, train_split=0.8):
    """
    Creates training and validation dataloaders.

    Args:
        dataframe (pl.DataFrame): The input dataframe.
        batch_size (int): The batch size.
        train_split (float): The proportion of data to use for training.

    Returns:
        tuple: A tuple containing the training and validation dataloaders.
    """
    train_size = int(train_split * len(dataframe))
    train_df = dataframe[:train_size]
    val_df = dataframe[train_size:]

    train_dataset = TabularSequenceDataset(train_df)
    val_dataset = TabularSequenceDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == '__main__':
    from data_generator import generate_dummy_data

    # Generate dummy data
    dummy_data = generate_dummy_data()

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(dummy_data)

    # Print a batch from the train loader
    for batch in train_loader:
        print(batch)
        break
