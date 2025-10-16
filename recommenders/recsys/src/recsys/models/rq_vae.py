"""
Residual Quantized Variational Autoencoder for Collaborative Filtering
https://arxiv.org/pdf/2306.08121
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)


class Model(nn.Module):
    def __init__(
        self,
        interaction_matrix: torch.Tensor,
        num_users,
        num_items,
        hidden_dim=256,
        latent_dim=64,
        num_embeddings=512,
        num_quantizers=4,  # Number of residual quantizers
        num_residual_layers=2,
        commitment_cost=0.25,
        dropout=0.5,
        avg_rating: float = None,
        include_bias: bool = True,
    ):
        super().__init__()
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(dropout)
        self.num_quantizers = num_quantizers

        # Store the full user-item interaction matrix
        self.register_buffer("interaction_matrix", interaction_matrix)

        # Encoder
        self.encoder_conv = nn.Conv1d(1, hidden_dim, 4, stride=2, padding=1)
        self.encoder_residual = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_residual_layers)]
        )
        self.encoder_final = nn.Conv1d(hidden_dim, latent_dim, 3, padding=1)

        # Vector Quantizers
        self.vqs = nn.ModuleList(
            [
                VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
                for _ in range(num_quantizers)
            ]
        )

        # Decoder
        self.decoder_conv = nn.ConvTranspose1d(
            latent_dim, hidden_dim, 4, stride=2, padding=1
        )
        self.decoder_residual = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_residual_layers)]
        )
        self.decoder_final = nn.ConvTranspose1d(hidden_dim, 1, 3, padding=1)

        # Output projection
        self.output_proj = nn.Linear(num_items, num_items)

    def encode(self, x):
        # x shape: (batch_size, num_items)
        x = x.unsqueeze(1)  # (batch_size, 1, num_items)
        x = F.relu(self.encoder_conv(x))

        for layer in self.encoder_residual:
            x = layer(x)

        x = self.encoder_final(x)
        return x

    def decode(self, z):
        z = F.relu(self.decoder_conv(z))

        for layer in self.decoder_residual:
            z = layer(z)

        z = self.decoder_final(z)
        z = z.squeeze(1)  # (batch_size, num_items)
        return self.output_proj(z)

    def forward(self, user_ids, item_ids=None):
        # Get the user-item interaction vectors for the current batch of users
        x = self.interaction_matrix[user_ids]
        # Encode
        z_e = self.encode(self.dropout(x))

        # Residual Quantization
        quantized_vectors = []
        losses = []
        perplexities = []
        residual = z_e

        for vq in self.vqs:
            quantized, loss, perplexity = vq(residual)
            residual = residual - quantized
            quantized_vectors.append(quantized)
            losses.append(loss)
            perplexities.append(perplexity)

        # Sum the quantized vectors to get the final quantized representation
        z_q = torch.sum(torch.stack(quantized_vectors), dim=0)
        vq_loss = torch.sum(torch.stack(losses))
        # Average perplexity across quantizers
        perplexity = torch.mean(torch.stack(perplexities))

        # Decode
        recon = self.decode(z_q)

        if item_ids is not None:
            # For ranking/prediction, gather scores for specific items
            return recon.gather(1, item_ids.unsqueeze(1)).squeeze()

        return recon, vq_loss, perplexity
