"""
Matrix Factorization for Collaborative Filtering
Classic approach that decomposes the user-item interaction matrix into 
low-dimensional user and item embeddings.

Predicts ratings as the inner product of user and item embeddings plus bias terms:
rating = user_embedding · item_embedding + user_bias + item_bias

Uses sigmoid activation for binary classification (implicit feedback).
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
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1) if include_bias else None
        self.item_bias = nn.Embedding(num_items, 1) if include_bias else None

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        dot_product = torch.sum(user_embeds * item_embeds, dim=1, keepdim=True)

        # Add bias terms
        if self.user_bias is not None:
            dot_product += self.user_bias(user_ids)
        if self.item_bias is not None:
            dot_product += self.item_bias(item_ids)

        return dot_product
