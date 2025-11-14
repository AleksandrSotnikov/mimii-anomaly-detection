"""Autoencoder models for anomaly detection."""

import torch
import torch.nn as nn
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class SoundAnomalyAutoencoder(nn.Module):
    """Autoencoder for detecting anomalies in sound data.
    
    Architecture:
    - Encoder: Input -> Conv2D layers -> Latent (bottleneck)
    - Decoder: Latent -> ConvTranspose2D layers -> Output
    """
    
    def __init__(self, input_channels: int = 1, latent_dim: int = 128):
        """Initialize Autoencoder.
        
        Args:
            input_channels: Number of input channels (1 for mono audio)
            latent_dim: Dimension of latent space
        """
        super(SoundAnomalyAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        
        # Encoder: Input (1, 128, 157) for mel-spectrogram
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # (32, 64, 79)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 32, 40)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 16, 20)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (256, 8, 10)
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # Latent representation
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(256 * 8 * 10, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 8 * 10)
        self.unflatten = nn.Unflatten(1, (256, 8, 10))
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128, 16, 20)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, 32, 40)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),  # (32, 64, 79)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # (1, 128, 158)
            nn.Tanh()
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Latent representation (B, latent_dim)
        """
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.fc_encode(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to output.
        
        Args:
            z: Latent tensor (B, latent_dim)
            
        Returns:
            Reconstructed output (B, C, H, W)
        """
        x = self.fc_decode(z)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Tuple of (reconstructed_output, latent_representation)
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        
        # Adjust output size to match input
        if reconstructed.shape != x.shape:
            reconstructed = reconstructed[:, :, :x.shape[2], :x.shape[3]]
        
        return reconstructed, latent
    
    def reconstruction_error(self, x: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
        """Calculate reconstruction error (MSE).
        
        Args:
            x: Input tensor
            reduction: 'none', 'mean', or 'sum'
            
        Returns:
            MSE reconstruction error
        """
        reconstructed, _ = self.forward(x)
        error = ((x - reconstructed) ** 2)
        
        if reduction == 'mean':
            return error.mean()
        elif reduction == 'sum':
            return error.sum()
        else:
            # Return per-sample error
            return error.view(error.size(0), -1).mean(dim=1)


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for anomaly detection.
    
    Uses reparameterization trick for stochastic encoding.
    """
    
    def __init__(self, input_channels: int = 1, latent_dim: int = 128):
        """Initialize VAE.
        
        Args:
            input_channels: Number of input channels
            latent_dim: Dimension of latent space
        """
        super(VariationalAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 8 * 10, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 10, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 8 * 10)
        self.unflatten = nn.Unflatten(1, (256, 8, 10))
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent space parameters.
        
        Returns:
            Tuple of (mu, logvar)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        h = self.encoder(x)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space.
        
        Args:
            z: Latent vector
            
        Returns:
            Reconstructed output
        """
        x = self.fc_decode(z)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        # Adjust output size
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if reconstruction.shape != x.shape:
            reconstruction = reconstruction[:, :, :x.shape[2], :x.shape[3]]
        
        return reconstruction, mu, logvar
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor,
                     beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VAE loss function.
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            beta: Weight for KL divergence term
            
        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_loss)
        """
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
