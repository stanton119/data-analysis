import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        hidden_dim=600,
        latent_dim=200,
        dropout=0.5,
        avg_rating: float = None,
        include_bias: bool = True,
    ):
        super().__init__()
        self.num_items = num_items
        self.dropout = nn.Dropout(dropout)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.Tanh()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_items)
        )
        
    def encode(self, x):
        h = self.encoder(self.dropout(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, user_ids, item_ids=None):
        # For VAE, we need user interaction vectors
        # This is a simplified version - in practice you'd need to construct user vectors
        batch_size = user_ids.size(0)
        x = torch.zeros(batch_size, self.num_items, device=user_ids.device)
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        if item_ids is not None:
            # Return specific item scores for evaluation
            return recon.gather(1, item_ids.unsqueeze(1)).squeeze()
        
        return recon, mu, logvar
