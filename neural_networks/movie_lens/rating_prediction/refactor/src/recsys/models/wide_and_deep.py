"""
Wide & Deep Learning for Recommender Systems - 2016
https://arxiv.org/abs/1606.07792


Wide - for memorizing specific interactions
1. The input is a concatenation of
    1. one-hot encoding of user/item IDs
    2. cross product of the one-hot encodings
2. Linear weights on top
3. This is implemented here through embeddings, which make a linear sum of embedding values (weights) based on binary features

Deep - learns dense embeddings for generalizing unseen feature combinations/non-linear relationships
1. Convert user/item IDs into embeddings
2. Concatenate
3. MLP
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
        deep_layers=[512, 256, 128],
        dropout=0.2,
        avg_rating: float = None,
        include_bias: bool = True,
    ):
        super().__init__()

        # Deep part - embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Wide part - linear weights for user, item, and cross-product
        self.user_wide = nn.Embedding(num_users, 1)
        self.item_wide = nn.Embedding(num_items, 1)
        # Cross-product: unique combination of user-item pairs
        self.cross_product = nn.Embedding(num_users * num_items, 1)

        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1) if include_bias else None
        self.item_bias = nn.Embedding(num_items, 1) if include_bias else None

        # Deep MLP
        deep_input_dim = embedding_dim * 2
        layers = []
        for layer_size in deep_layers:
            layers.extend(
                [nn.Linear(deep_input_dim, layer_size), nn.ReLU(), nn.Dropout(dropout)]
            )
            deep_input_dim = layer_size

        self.deep_net = nn.Sequential(*layers)
        self.deep_output = nn.Linear(deep_layers[-1], 1)

        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))
        if avg_rating:
            self.bias.data.fill_(avg_rating)

    def forward(self, user_ids, item_ids):
        # Wide component
        user_wide = self.user_wide(user_ids)
        item_wide = self.item_wide(item_ids)

        # Cross-product feature: hash user-item combination
        cross_idx = user_ids * self.item_wide.num_embeddings + item_ids
        cross_wide = self.cross_product(cross_idx)

        wide_out = user_wide + item_wide + cross_wide

        # Deep component
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        deep_input = torch.cat([user_embed, item_embed], dim=1)

        deep_hidden = self.deep_net(deep_input)
        deep_out = self.deep_output(deep_hidden)

        # Combine wide + deep + bias
        output = wide_out + deep_out + self.bias

        # Add bias terms
        if self.user_bias is not None:
            output += self.user_bias(user_ids)
        if self.item_bias is not None:
            output += self.item_bias(item_ids)

        return output
