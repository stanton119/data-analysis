"""
Neural Collaborative Filtering - 2017
https://arxiv.org/abs/1708.05031

Combines traditional matrix factorization with neural networks to learn
non-linear user-item interactions. Uses embeddings for users and items
that are processed through a multi-layer perceptron to predict ratings.

The NCF framework includes three instantiations:
1. Generalized Matrix Factorization (GMF) - Element-wise product of user and item embeddings (linear interactions)
2. Multi-Layer Perceptron (MLP) - Concatenated embeddings passed through deep layers (non-linear interactions)
3. Neural Matrix Factorization (NeuMF) - which combines GMF and MLP

"""
import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        mf_dim=8,
        mlp_dim=32,
        layers=[64, 32, 16],
        avg_rating: float = None,
        include_bias: bool = True,
    ):
        super().__init__()
        # GMF part
        self.user_mf_embedding = nn.Embedding(num_users, mf_dim)
        self.item_mf_embedding = nn.Embedding(num_items, mf_dim)
        
        # MLP part  
        self.user_mlp_embedding = nn.Embedding(num_users, mlp_dim)
        self.item_mlp_embedding = nn.Embedding(num_items, mlp_dim)
        
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1) if include_bias else None
        self.item_bias = nn.Embedding(num_items, 1) if include_bias else None
        
        # MLP layers
        mlp_layers = []
        input_size = mlp_dim * 2
        for layer_size in layers:
            mlp_layers.extend([nn.Linear(input_size, layer_size), nn.ReLU()])
            input_size = layer_size
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Output layer
        self.output = nn.Linear(mf_dim + layers[-1], 1)
        
        if avg_rating:
            self.output.bias.data.fill_(avg_rating)
        
    def forward(self, user_ids, item_ids):
        # GMF path
        user_mf = self.user_mf_embedding(user_ids)
        item_mf = self.item_mf_embedding(item_ids)
        gmf_output = user_mf * item_mf
        
        # MLP path
        user_mlp = self.user_mlp_embedding(user_ids)
        item_mlp = self.item_mlp_embedding(item_ids)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=1)
        mlp_output = self.mlp(mlp_input)
        
        # Combine and output
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        output = self.output(combined)
        
        # Add bias terms
        if self.user_bias is not None:
            output += self.user_bias(user_ids)
        if self.item_bias is not None:
            output += self.item_bias(item_ids)
            
        return output
