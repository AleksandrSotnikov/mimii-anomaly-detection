"""Tests for feature extraction module."""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_extraction import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor(sr=16000)
        # Create dummy audio (1 second at 16kHz)
        self.audio = np.random.randn(16000)
    
    def test_mfcc_extraction(self):
        """Test MFCC extraction."""
        mfcc = self.extractor.extract_mfcc(self.audio)
        self.assertIsNotNone(mfcc)
        self.assertEqual(mfcc.shape[0], 13)  # Default n_mfcc
        self.assertGreater(mfcc.shape[1], 0)  # Time steps
    
    def test_mel_spectrogram_extraction(self):
        """Test Mel-spectrogram extraction."""
        mel_spec = self.extractor.extract_mel_spectrogram(self.audio)
        self.assertIsNotNone(mel_spec)
        self.assertEqual(mel_spec.shape[0], 128)  # Default n_mels
        self.assertGreater(mel_spec.shape[1], 0)
    
    def test_zcr_extraction(self):
        """Test Zero Crossing Rate extraction."""
        zcr = self.extractor.extract_zcr(self.audio)
        self.assertIsNotNone(zcr)
        self.assertEqual(zcr.shape[0], 1)
    
    def test_all_features(self):
        """Test extraction of all features."""
        features = self.extractor.extract_all_features(self.audio)
        self.assertIsInstance(features, dict)
        self.assertIn('mfcc', features)
        self.assertIn('mel_spectrogram', features)


if __name__ == '__main__':
    unittest.main()
