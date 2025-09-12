from typing import List
import pytorch_lightning as pyl
import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=10,
        layer_sizes: List[int] = [32, 16],
        avg_rating: float = None,
        include_bias: bool = True,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)
        self.include_bias = include_bias

        layer_sizes = [embedding_dim * 2] + layer_sizes
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        self.output = nn.Sequential(*layers, nn.Linear(layer_sizes[-1], 1))

        if avg_rating:
            self.output[-1].bias.data.fill_(avg_rating)

        self.max_rating = 5.0
        self.min_rating = 0.5

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        embeds_concat = torch.concat([user_embeds, item_embeds], dim=1)

        if self.include_bias:
            user_bias = self.user_biases(user_ids)
            item_bias = self.item_biases(item_ids)
            prediction = self.output(embeds_concat) + user_bias + item_bias
        else:
            prediction = self.output(embeds_concat)

        prediction = torch.clamp(prediction, min=self.min_rating, max=self.max_rating)
        return prediction
