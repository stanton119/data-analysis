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
        **kwargs,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Multi-head attention layers
        self.attention_layers = nn.ModuleList(
            [MultiHeadAttention(embedding_dim, num_heads) for _ in range(num_layers)]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(embedding_dim) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embedding_dim * 2, 1)

        if avg_rating:
            self.output.bias.data.fill_(avg_rating)

    def forward(self, batch):
        user_ids = batch["user_id"]
        item_ids = batch["item_id"]
        user_embed = self.user_embedding(user_ids).unsqueeze(1)  # [batch, 1, embed_dim]
        item_embed = self.item_embedding(item_ids).unsqueeze(1)  # [batch, 1, embed_dim]

        # Stack embeddings
        x = torch.cat([user_embed, item_embed], dim=1)  # [batch, 2, embed_dim]

        # Apply attention layers
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms):
            x = layer_norm(attention(x))
            x = self.dropout(x)

        # Flatten and output
        x = x.view(x.size(0), -1)
        return self.output(x)
