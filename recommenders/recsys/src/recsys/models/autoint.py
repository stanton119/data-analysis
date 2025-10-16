"""
AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks - 2018
https://arxiv.org/abs/1810.11921

This model uses a multi-head self-attention mechanism to explicitly model the
interactions between features. It can be used for tasks like click-through rate (CTR)
prediction and recommendation.

The core idea is to map both categorical and numerical features into a common
low-dimensional space and then use a self-attention network to learn the
feature interactions.
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_res = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        Q = (
            self.W_q(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.W_k(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.W_v(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )
        return self.W_res(attn_output) + x


class Model(torch.nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
        num_heads=8,
        num_layers=3,
        dropout=0.1,
        avg_rating: float = None,
        include_bias: bool = True,
        categorical_features: dict = None,
        num_user_continuous: int = 0,
        num_item_continuous: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.categorical_embeddings = nn.ModuleDict()
        if categorical_features:
            for feature, cardinality in categorical_features.items():
                self.categorical_embeddings[feature] = nn.Embedding(
                    cardinality, embedding_dim
                )

        if num_user_continuous > 0:
            self.user_continuous_proj = nn.Linear(num_user_continuous, embedding_dim)
        if num_item_continuous > 0:
            self.item_continuous_proj = nn.Linear(num_item_continuous, embedding_dim)

        # Multi-head attention layers
        self.attention_layers = nn.ModuleList(
            [MultiHeadAttention(embedding_dim, num_heads) for _ in range(num_layers)]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(embedding_dim) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        # Calculate the total number of features
        num_features = 2  # user_id, item_id
        if categorical_features:
            num_features += len(categorical_features)
        if num_user_continuous > 0:
            num_features += 1
        if num_item_continuous > 0:
            num_features += 1

        self.output = nn.Linear(embedding_dim * num_features, 1)

        if avg_rating:
            self.output.bias.data.fill_(avg_rating)

    def forward(self, batch):
        embeddings = []

        # User and item IDs
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        embeddings.append(self.user_embedding(user_ids).unsqueeze(1))
        embeddings.append(self.item_embedding(item_ids).unsqueeze(1))

        # Categorical features
        if "user_features" in batch and "categorical" in batch["user_features"]:
            for feature, embed_layer in self.categorical_embeddings.items():
                if feature in batch["user_features"]["categorical"]:
                    embeddings.append(
                        embed_layer(
                            batch["user_features"]["categorical"][feature]
                        ).unsqueeze(1)
                    )
        if "item_features" in batch and "categorical" in batch["item_features"]:
            for feature, embed_layer in self.categorical_embeddings.items():
                if feature in batch["item_features"]["categorical"]:
                    embeddings.append(
                        embed_layer(
                            batch["item_features"]["categorical"][feature]
                        ).unsqueeze(1)
                    )

        # Continuous features
        if hasattr(self, "user_continuous_proj") and "user_features" in batch and "continuous" in batch["user_features"]:
            user_cont = self.user_continuous_proj(
                batch["user_features"]["continuous"]
            ).unsqueeze(1)
            embeddings.append(user_cont)
        if hasattr(self, "item_continuous_proj") and "item_features" in batch and "continuous" in batch["item_features"]:
            item_cont = self.item_continuous_proj(
                batch["item_features"]["continuous"]
            ).unsqueeze(1)
            embeddings.append(item_cont)

        # Stack embeddings
        x = torch.cat(embeddings, dim=1)

        # Apply attention layers
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms):
            x = layer_norm(attention(x))
            x = self.dropout(x)

        # Flatten and output
        x = x.view(x.size(0), -1)
        return self.output(x)
