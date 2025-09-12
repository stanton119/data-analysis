"""
Factorization Machines (FM) - 2010
https://ieeexplore.ieee.org/document/5694074

Factorization Machines model all interactions between variables using factorized parameters.
For recommendation systems, FM captures user-item interactions plus any additional features.

The model equation is composed of a linear weighting of features combined with sparse interactions:
y = w0 + Σ(wi * xi) + Σ(Σ(<vi, vj> * xi * xj))

Where:
- w0: global bias
- wi: linear weights  
- vi: embedding vectors for feature interactions
- <vi, vj>: dot product of embeddings

The input features, xi, are one hot encoding of user/items concatenated together.

For user-item recommendation, this reduces to matrix factorization when only 
user and item features are present. The linear weights become the user/item bias terms:

• FM: y = w0 + wu + wi + <vu, vi>
• MF: y = μ + bu + bi + <pu, qi>

Where:
• w0 = global bias μ
• wu, wi = user/item biases bu, bi 
• <vu, vi> = user-item interaction <pu, qi>
"""

import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=10,
        avg_rating: float = None,
        include_bias: bool = True,
    ):
        super().__init__()

        # Total number of features (user + item)
        self.num_features = num_users + num_items
        self.num_users = num_users

        # FM parameters
        self.global_bias = nn.Parameter(torch.zeros(1)) if include_bias else None
        self.linear_weights = nn.Embedding(self.num_features, 1)
        self.embeddings = nn.Embedding(self.num_features, embedding_dim)

        # Initialize embeddings
        nn.init.normal_(self.embeddings.weight, std=0.01)
        nn.init.normal_(self.linear_weights.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        batch_size = user_ids.size(0)

        # Convert to feature indices
        # Users: 0 to num_users-1, Items: num_users to num_users+num_items-1
        item_feature_ids = item_ids + self.num_users

        # Stack user and item feature ids
        feature_ids = torch.stack(
            [user_ids, item_feature_ids], dim=1
        )  # [batch_size, 2]

        # Linear terms
        linear_part = 0
        if self.global_bias is not None:
            linear_part += self.global_bias.expand(batch_size, 1)

        linear_weights = self.linear_weights(feature_ids)  # [batch_size, 2, 1]
        linear_part += torch.sum(linear_weights, dim=1)  # [batch_size, 1]

        # Interaction terms: 0.5 * (sum_square - square_sum)
        embeddings = self.embeddings(feature_ids)  # [batch_size, 2, embedding_dim]

        sum_square = torch.sum(embeddings, dim=1) ** 2  # [batch_size, embedding_dim]
        square_sum = torch.sum(embeddings**2, dim=1)  # [batch_size, embedding_dim]

        interaction_part = 0.5 * torch.sum(
            sum_square - square_sum, dim=1, keepdim=True
        )  # [batch_size, 1]

        return linear_part + interaction_part
