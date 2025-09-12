import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class LightGCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
    
    def forward(self, x, edge_index):
        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class Model(torch.nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
        n_layers=3,
        avg_rating: float = None,
        include_bias: bool = True,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Graph convolution
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(n_layers)])
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
    
    def forward(self, user_ids, item_ids, edge_index=None):
        # If no edge_index provided, use simple embedding lookup
        if edge_index is None:
            user_embed = self.user_embedding(user_ids)
            item_embed = self.item_embedding(item_ids)
            return torch.sigmoid(torch.sum(user_embed * item_embed, dim=1, keepdim=True))
        
        # Full graph convolution
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embeddings_list = [all_embeddings]
        
        for conv in self.convs:
            all_embeddings = conv(all_embeddings, edge_index)
            embeddings_list.append(all_embeddings)
        
        # Average all layers
        final_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        
        user_embeddings = final_embeddings[:self.num_users]
        item_embeddings = final_embeddings[self.num_users:]
        
        user_embed = user_embeddings[user_ids]
        item_embed = item_embeddings[item_ids]
        
        return torch.sigmoid(torch.sum(user_embed * item_embed, dim=1, keepdim=True))
