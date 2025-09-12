import torch
import torch.nn as nn


class Model(torch.nn.Module):
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
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Wide part - linear on user and item IDs
        self.user_wide = nn.Embedding(num_users, 1)
        self.item_wide = nn.Embedding(num_items, 1)
        
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1) if include_bias else None
        self.item_bias = nn.Embedding(num_items, 1) if include_bias else None
        
        # Deep part - MLP
        deep_input_dim = embedding_dim * 2
        deep_nets = []
        for layer_size in deep_layers:
            deep_nets.extend([
                nn.Linear(deep_input_dim, layer_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            deep_input_dim = layer_size
        self.deep = nn.Sequential(*deep_nets)
        
        # Final output
        self.output = nn.Linear(deep_layers[-1], 1)
        
        if avg_rating:
            self.output.bias.data.fill_(avg_rating)
    
    def forward(self, user_ids, item_ids):
        # Wide part - linear combination
        wide_out = self.user_wide(user_ids) + self.item_wide(item_ids)
        
        # Deep part - embeddings through MLP
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        deep_input = torch.cat([user_embed, item_embed], dim=1)
        deep_out = self.deep(deep_input)
        deep_out = self.output(deep_out)
        
        # Combine wide and deep
        output = wide_out + deep_out
        
        # Add bias terms
        if self.user_bias is not None:
            output += self.user_bias(user_ids)
        if self.item_bias is not None:
            output += self.item_bias(item_ids)
            
        return torch.sigmoid(output)
