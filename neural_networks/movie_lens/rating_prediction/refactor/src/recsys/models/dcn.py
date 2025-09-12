import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
        cross_layers=3,
        deep_layers=[512, 256, 128],
        dropout=0.2,
        avg_rating: float = None,
        include_bias: bool = True,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        input_dim = embedding_dim * 2
        
        # Cross Network
        self.cross_layers = nn.ModuleList()
        for _ in range(cross_layers):
            self.cross_layers.append(nn.Linear(input_dim, input_dim))
        
        # Deep Network
        deep_input_dim = input_dim
        deep_nets = []
        for layer_size in deep_layers:
            deep_nets.extend([
                nn.Linear(deep_input_dim, layer_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            deep_input_dim = layer_size
        self.deep_net = nn.Sequential(*deep_nets)
        
        # Output layer
        self.output = nn.Linear(input_dim + deep_layers[-1], 1)
        
        if avg_rating:
            self.output.bias.data.fill_(avg_rating)
    
    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        
        # Input features
        x0 = torch.cat([user_embed, item_embed], dim=1)
        
        # Cross Network
        xl = x0
        for cross_layer in self.cross_layers:
            xl_w = cross_layer(xl)
            xl = x0 * xl_w + xl
        
        # Deep Network
        deep_out = self.deep_net(x0)
        
        # Combine cross and deep
        combined = torch.cat([xl, deep_out], dim=1)
        return self.output(combined)
