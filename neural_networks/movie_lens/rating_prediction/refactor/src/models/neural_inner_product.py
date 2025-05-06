import pytorch_lightning as pyl
import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(
        self,
        n_users,
        n_movies,
        embedding_dim=10,
        avg_rating: float = None,
        # learning_rate: float = 5e-3,
        include_bias: bool = True,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.user_biases = nn.Embedding(n_users, 1)
        self.movie_biases = nn.Embedding(n_movies, 1)
        self.include_bias = include_bias

        self.max_rating = 5.0
        self.min_rating = 0.5
        # self.learning_rate = learning_rate
        # self.save_hyperparameters()

    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)

        dot_product = torch.sum(user_embeds * movie_embeds, dim=1)

        if self.include_bias:
            user_bias = self.user_biases(user_ids).squeeze()
            movie_bias = self.movie_biases(movie_ids).squeeze()
            prediction = dot_product + user_bias + movie_bias
        else:
            prediction = dot_product

        prediction = torch.clamp(prediction, min=self.min_rating, max=self.max_rating)
        return prediction
