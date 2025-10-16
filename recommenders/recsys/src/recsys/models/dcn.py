"""
Deep & Cross Network for Ad Click Predictions - 2017
https://arxiv.org/abs/1708.05123

Linear weighting of cross and deep networks.

Cross network - models interactions through multiplying by input layer at each stage
1. Concat user/item embeddings to give the layer input features
1. linear weights applied to the input features to give a scalar
2. multiply that scalar by the input features again to give an interaction effect
1. x_{l+1} = x_0 ⊙ (w_l^T * x_l + b_l) + x_l
3. Each cross layer adds one degree of polynomial interaction
4. E.g. weights = [1, 0, 0] -> x0*x0 = x0^2

Deep network - learns deep non-linear relationships
1. Concat user/item embeddings
2. MLP
"""

import torch
import torch.nn as nn


class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xl):
        return x0 * (torch.sum(xl * self.weight, dim=1, keepdim=True) + self.bias) + xl


class Model(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
        cross_layers=3,
        deep_layers=[512, 256, 128],
        dropout=0.2,
        avg_rating: float = None,
        **kwargs,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        input_dim = embedding_dim * 2

        # Cross Network
        self.cross_layers = nn.ModuleList(
            [CrossLayer(input_dim) for _ in range(cross_layers)]
        )

        # Deep Network
        deep_input_dim = input_dim
        deep_nets = []
        for layer_size in deep_layers:
            deep_nets.extend(
                [nn.Linear(deep_input_dim, layer_size), nn.ReLU(), nn.Dropout(dropout)]
            )
            deep_input_dim = layer_size
        self.deep_net = nn.Sequential(*deep_nets)

        # Output layer
        self.output = nn.Linear(input_dim + deep_layers[-1], 1)

        if avg_rating:
            self.output.bias.data.fill_(avg_rating)

    def forward(self, batch):
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)

        x0 = torch.cat([user_embed, item_embed], dim=1)

        # Cross Network
        xl = x0
        for cross_layer in self.cross_layers:
            xl = cross_layer(x0, xl)

        # Deep Network
        deep_out = self.deep_net(x0)

        # Combine cross and deep
        combined = torch.cat([xl, deep_out], dim=1)
        return self.output(combined)
