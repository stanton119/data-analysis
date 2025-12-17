"""
Self-Attentive Sequential Recommendation - 2018
https://arxiv.org/abs/1808.09781
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        num_items,
        embedding_dim=64,
        max_sequence_length=50,
        num_blocks=2,
        num_heads=2,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.positional_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_blocks,
        )

        self.output_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, batch):
        input_sequence = batch["input_sequence"]
        attention_mask = input_sequence == 0

        item_embeds = self.item_embedding(input_sequence)

        positions = torch.arange(
            0, input_sequence.size(1), device=input_sequence.device
        ).unsqueeze(0)
        pos_embeds = self.positional_embedding(positions)

        seq_embedding = self.dropout(item_embeds + pos_embeds)

        transformer_out = self.transformer_encoder(
            seq_embedding, src_key_padding_mask=attention_mask
        )

        seq_lengths = (input_sequence != 0).sum(dim=1)
        last_item_indices = seq_lengths - 1

        batch_indices = torch.arange(
            input_sequence.size(0), device=input_sequence.device
        )
        last_item_representation = transformer_out[batch_indices, last_item_indices, :]

        last_item_representation = self.output_layer_norm(last_item_representation)

        # During training, we calculate scores for target and negative items
        if "target_item" in batch:
            target_item = batch["target_item"]
            negative_samples = batch["negative_samples"]

            target_item_embed = self.item_embedding(target_item)
            negative_samples_embed = self.item_embedding(negative_samples)

            pos_scores = torch.sum(
                last_item_representation * target_item_embed, dim=1
            ).unsqueeze(1)
            neg_scores = torch.sum(
                last_item_representation.unsqueeze(1) * negative_samples_embed, dim=2
            )

            return torch.cat([pos_scores, neg_scores], dim=1)

        # During inference, we calculate scores for all items
        else:
            all_item_embeds = self.item_embedding.weight
            scores = torch.matmul(last_item_representation, all_item_embeds.t())
            return scores
