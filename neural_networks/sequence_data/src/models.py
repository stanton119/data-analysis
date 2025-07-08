import torch
import torch.nn as nn
from typing import Protocol


class ModelProtocol(Protocol):
    def forward(
        self, tabular: torch.Tensor, sequence: torch.Tensor
    ) -> torch.Tensor: ...


class StatisticalAggregationModel(nn.Module, ModelProtocol):
    """A model that combines tabular data with statistical aggregations of sequence data."""

    def __init__(self, num_tabular_features, hidden_dim=64):
        """
        Args:
            num_tabular_features (int): The number of features in the tabular data.
            hidden_dim (int): The dimension of the hidden layer.
        """
        super().__init__()
        # The input dimension will be the number of tabular features plus 3 for the sequence aggregations (mean, max, sum)
        self.fc1 = nn.Linear(num_tabular_features + 3, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, tabular, sequence):
        """
        Args:
            tabular (torch.Tensor): The tabular data.
            sequence (torch.Tensor): The sequence data.

        Returns:
            torch.Tensor: The model\'s prediction.
        """
        # Calculate statistical aggregations of the sequence
        seq_mean = torch.mean(sequence, dim=1, keepdim=True)
        seq_max, _ = torch.max(sequence, dim=1, keepdim=True)
        seq_sum = torch.sum(sequence, dim=1, keepdim=True)

        # Concatenate the tabular data with the sequence aggregations
        combined_features = torch.cat([tabular, seq_mean, seq_max, seq_sum], dim=1)

        # Pass the combined features through the feed-forward network
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FixedPositionalWeightingModel(nn.Module, ModelProtocol):
    """A model that combines tabular data with fixed positional weighting of sequence data."""

    def __init__(self, num_tabular_features, sequence_length, hidden_dim=64):
        """
        Args:
            num_tabular_features (int): The number of features in the tabular data.
            sequence_length (int): The length of the input sequence.
            hidden_dim (int): The dimension of the hidden layer.
        """
        super().__init__()
        self.sequence_length = sequence_length
        # Create fixed linear decay weights
        self.weights = torch.arange(1, sequence_length + 1, dtype=torch.float32).flip(
            dims=[0]
        )
        self.weights = (
            self.weights / self.weights.sum()
        )  # Normalize weights to sum to 1

        # The input dimension will be the number of tabular features + 1 (for the weighted sequence)
        self.fc1 = nn.Linear(num_tabular_features + 1, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, tabular: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tabular (torch.Tensor): The tabular data.
            sequence (torch.Tensor): The sequence data.

        Returns:
            torch.Tensor: The model\'s prediction.
        """  # Apply fixed positional weights to the sequence
        # Ensure weights are broadcastable: [1, sequence_length]
        weighted_sequence = (sequence * self.weights.to(sequence.device)).sum(
            dim=1, keepdim=True
        )

        # Concatenate the tabular data with the weighted sequence
        combined_features = torch.cat([tabular, weighted_sequence], dim=1)

        # Pass the combined features through the feed-forward network
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class LearnedStaticPositionalWeightingModel(nn.Module, ModelProtocol):
    """A model that combines tabular data with learned static positional weighting of sequence data."""

    def __init__(self, num_tabular_features, sequence_length, hidden_dim=64):
        """
        Args:
            num_tabular_features (int): The number of features in the tabular data.
            sequence_length (int): The length of the input sequence.
            hidden_dim (int): The dimension of the hidden layer.
        """
        super().__init__()
        self.sequence_length = sequence_length

        # Learnable weights for each position in the sequence
        # Initialized to ones and normalized to sum to 1, similar to fixed weights but learnable
        initial_weights = torch.ones(sequence_length, dtype=torch.float32)
        self.weights = nn.Parameter(initial_weights / initial_weights.sum())

        # The input dimension will be the number of tabular features + 1 (for the weighted sequence)
        self.fc1 = nn.Linear(num_tabular_features + 1, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, tabular: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tabular (torch.Tensor): The tabular data.
            sequence (torch.Tensor): The sequence data.

        Returns:
            torch.Tensor: The model\'s prediction.
        """  # Apply learned static positional weights to the sequence
        # Ensure weights are broadcastable: [1, sequence_length]
        weighted_sequence = (sequence * self.weights).sum(dim=1, keepdim=True)

        # Concatenate the tabular data with the weighted sequence
        combined_features = torch.cat([tabular, weighted_sequence], dim=1)

        # Pass the combined features through the feed-forward network
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class LearnedWeightedAverageModel(nn.Module, ModelProtocol):
    """A model that combines tabular data with a learned weighted average (simple attention) of sequence data."""

    def __init__(self, num_tabular_features, sequence_length, hidden_dim=64):
        """
        Args:
            num_tabular_features (int): The number of features in the tabular data.
            sequence_length (int): The length of the input sequence.
            hidden_dim (int): The dimension of the hidden layer.
        """
        super().__init__()
        self.sequence_length = sequence_length

        # Layer to learn attention scores for each element in the sequence
        # Input: sequence_length (each element in the sequence)
        # Output: sequence_length (a score for each element)
        self.attention_weights_layer = nn.Linear(sequence_length, sequence_length)

        # The input dimension for the final FC layer will be num_tabular_features + 1 (for the weighted sequence)
        self.fc1 = nn.Linear(num_tabular_features + 1, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, tabular: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tabular (torch.Tensor): The tabular data.
            sequence (torch.Tensor): The sequence data.

        Returns:
            torch.Tensor: The model\'s prediction.
        """  # Learn attention scores
        # The input to attention_weights_layer should be (batch_size, sequence_length)
        attention_scores = self.attention_weights_layer(sequence)

        # Apply softmax to get attention weights
        # dim=1 ensures softmax is applied across the sequence elements for each sample
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Compute the weighted sum of the sequence
        # (batch_size, sequence_length) * (batch_size, sequence_length) -> (batch_size, sequence_length)
        # sum(dim=1, keepdim=True) -> (batch_size, 1)
        weighted_sequence_sum = (sequence * attention_weights).sum(dim=1, keepdim=True)

        # Concatenate the tabular data with the weighted sequence sum
        combined_features = torch.cat([tabular, weighted_sequence_sum], dim=1)

        # Pass the combined features through the feed-forward network
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def get_model(
    model_name: str, num_tabular_features: int, sequence_length: int
) -> ModelProtocol:
    """
    Factory function to get a model instance based on its name.
    """
    if model_name == "StatisticalAggregationModel":
        return StatisticalAggregationModel(num_tabular_features=num_tabular_features)
    elif model_name == "FixedPositionalWeightingModel":
        return FixedPositionalWeightingModel(
            num_tabular_features=num_tabular_features, sequence_length=sequence_length
        )
    elif model_name == "LearnedStaticPositionalWeightingModel":
        return LearnedStaticPositionalWeightingModel(
            num_tabular_features=num_tabular_features, sequence_length=sequence_length
        )
    elif model_name == "LearnedWeightedAverageModel":
        return LearnedWeightedAverageModel(
            num_tabular_features=num_tabular_features, sequence_length=sequence_length
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    # Example usage of the factory function
    num_tabular_features = 10
    sequence_length = 50
    batch_size = 32

    tabular_data = torch.randn(batch_size, num_tabular_features)
    sequence_data = torch.randn(batch_size, sequence_length)

    # Get models using the factory function
    stat_agg_model = get_model(
        "StatisticalAggregationModel", num_tabular_features, sequence_length
    )
    fixed_weight_model = get_model(
        "FixedPositionalWeightingModel", num_tabular_features, sequence_length
    )
    learned_static_weight_model = get_model(
        "LearnedStaticPositionalWeightingModel", num_tabular_features, sequence_length
    )
    learned_weight_model = get_model(
        "LearnedWeightedAverageModel", num_tabular_features, sequence_length
    )

    # Test forward pass for each model
    print(
        f"StatisticalAggregationModel output shape: {stat_agg_model(tabular_data, sequence_data).shape}"
    )
    print(
        f"FixedPositionalWeightingModel output shape: {fixed_weight_model(tabular_data, sequence_data).shape}"
    )
    print(
        f"LearnedStaticPositionalWeightingModel output shape: {learned_static_weight_model(tabular_data, sequence_data).shape}"
    )
    print(
        f"LearnedWeightedAverageModel output shape: {learned_weight_model(tabular_data, sequence_data).shape}"
    )
