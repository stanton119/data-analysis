import pytorch_lightning as pyl
import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(
        self,
        n_users,
        n_movies,
        embedding_dim=10,  # , learning_rate: float = 5e-3
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.output = nn.Linear(embedding_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.max_rating = 5.0
        self.min_rating = 0.5
        self.output_bias = torch.logit(
            torch.tensor((3.9 - self.min_rating) / (self.max_rating - self.min_rating))
        )
        self.output.bias.data.fill_(self.output_bias)
        # self.learning_rate = learning_rate
        # self.save_hyperparameters()

    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        embeds_concat = torch.concat([user_embeds, movie_embeds], dim=1)
        output = self.output(embeds_concat)
        rating = (
            self.sigmoid(output) * (self.max_rating - self.min_rating) + self.min_rating
        )
        return rating.flatten()
