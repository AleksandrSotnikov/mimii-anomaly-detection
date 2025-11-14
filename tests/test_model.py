"""Tests for model architectures."""

import unittest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SoundAnomalyAutoencoder, VariationalAutoencoder


class TestModels(unittest.TestCase):
    """Test cases for model architectures."""
    
    def test_autoencoder_forward(self):
        """Test Autoencoder forward pass."""
        model = SoundAnomalyAutoencoder()
        # Input: (batch, channels, height, width)
        x = torch.randn(4, 1, 128, 157)
        recon, latent = model(x)
        
        self.assertEqual(recon.shape[0], 4)  # Batch size
        self.assertEqual(latent.shape, (4, 128))  # Latent dimension
    
    def test_vae_forward(self):
        """Test VAE forward pass."""
        model = VariationalAutoencoder()
        x = torch.randn(4, 1, 128, 157)
        recon, mu, logvar = model(x)
        
        self.assertEqual(recon.shape[0], 4)
        self.assertEqual(mu.shape, (4, 128))
        self.assertEqual(logvar.shape, (4, 128))
    
    def test_reconstruction_error(self):
        """Test reconstruction error calculation."""
        model = SoundAnomalyAutoencoder()
        x = torch.randn(4, 1, 128, 157)
        error = model.reconstruction_error(x)
        
        self.assertEqual(error.shape[0], 4)  # One error per sample
        self.assertTrue(torch.all(error >= 0))  # Errors should be non-negative


if __name__ == '__main__':
    unittest.main()
