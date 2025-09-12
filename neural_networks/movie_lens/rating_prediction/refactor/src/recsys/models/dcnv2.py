"""
DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems - 2020
https://arxiv.org/abs/2008.13535

V1 had the limitation that in cross networks the weights were a vector.
As such we had a scalar weighting multiplied by the input features.
Making this a matrix allows multiple interactions per layer and is more efficient.

The deep network is the same as before.
"""

import torch
import torch.nn as nn


class CrossLayerV2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, xl):
        return x0 * (xl @ self.weight + self.bias) + xl


class DCNV2(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
        cross_layers=3,
        deep_layers=[512, 256, 128],
        dropout=0.2,
        avg_rating: float = None,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        input_dim = embedding_dim * 2

        # Cross Network V2
        self.cross_layers = nn.ModuleList(
            [CrossLayerV2(input_dim) for _ in range(cross_layers)]
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

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)

        x0 = torch.cat([user_embed, item_embed], dim=1)

        # Cross Network V2
        xl = x0
        for cross_layer in self.cross_layers:
            xl = cross_layer(x0, xl)

        # Deep Network
        deep_out = self.deep_net(x0)

        # Combine cross and deep
        combined = torch.cat([xl, deep_out], dim=1)
        return self.output(combined)


Model = DCNV2
