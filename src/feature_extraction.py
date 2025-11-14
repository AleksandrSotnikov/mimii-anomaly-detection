"""Audio feature extraction for MIMII dataset."""

import numpy as np
import librosa
import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Extract audio features from waveforms."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 40
    ):
        """
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            n_mfcc: Number of MFCC coefficients
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram.
        
        Args:
            audio: Audio waveform (numpy array)
        
        Returns:
            Mel spectrogram (n_mels x time_frames)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features.
        
        Args:
            audio: Audio waveform (numpy array)
        
        Returns:
            MFCC features (n_mfcc x time_frames)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mfcc=self.n_mfcc
        )
        
        return mfcc
    
    def extract_features(self, audio: np.ndarray, feature_type: str = "mel") -> np.ndarray:
        """Extract features from audio.
        
        Args:
            audio: Audio waveform
            feature_type: Type of features ('mel' or 'mfcc')
        
        Returns:
            Extracted features
        """
        if feature_type == "mel":
            return self.extract_mel_spectrogram(audio)
        elif feature_type == "mfcc":
            return self.extract_mfcc(audio)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean and unit variance.
        
        Args:
            features: Feature matrix
        
        Returns:
            Normalized features
        """
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        normalized = (features - mean) / std
        
        return normalized


def collate_fn_with_features(
    batch,
    feature_extractor: AudioFeatureExtractor,
    feature_type: str = "mel"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function to extract features on-the-fly.
    
    Args:
        batch: List of (audio, label) tuples
        feature_extractor: Feature extractor instance
        feature_type: Type of features to extract
    
    Returns:
        features_batch: Batch of features as torch.Tensor
        labels_batch: Batch of labels as torch.Tensor
    """
    audios, labels = zip(*batch)
    
    features_list = []
    for audio in audios:
        # Extract features
        features = feature_extractor.extract_features(audio, feature_type)
        features = feature_extractor.normalize_features(features)
        
        # Flatten to 1D vector
        features_flat = features.flatten()
        features_list.append(features_flat)
    
    # Stack into batch
    features_batch = torch.FloatTensor(np.stack(features_list))
    labels_batch = torch.LongTensor(labels)
    
    return features_batch, labels_batch
