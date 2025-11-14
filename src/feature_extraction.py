"""Feature extraction from audio signals."""

import numpy as np
import librosa
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract audio features for anomaly detection.
    
    Supports MFCC, Mel-spectrogram, and other acoustic features.
    """
    
    def __init__(self, sr: int = 16000, n_mfcc: int = 13, 
                 n_fft: int = 2048, hop_length: int = 512,
                 n_mels: int = 128):
        """Initialize feature extractor.
        
        Args:
            sr: Sampling rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of Mel bins
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def extract_mfcc(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract MFCC features.
        
        Args:
            audio: Audio signal (1D array)
            
        Returns:
            MFCC features (n_mfcc x time_steps)
        """
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            # Normalize
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
            return mfcc
        except Exception as e:
            logger.error(f"Error extracting MFCC: {e}")
            return None
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract Mel-spectrogram.
        
        Args:
            audio: Audio signal (1D array)
            
        Returns:
            Mel-spectrogram (n_mels x time_steps)
        """
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            # Normalize
            mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)
            return mel_spec_db
        except Exception as e:
            logger.error(f"Error extracting Mel-spectrogram: {e}")
            return None
    
    def extract_zcr(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract Zero Crossing Rate.
        
        Args:
            audio: Audio signal (1D array)
            
        Returns:
            ZCR values (1 x time_steps)
        """
        try:
            zcr = librosa.feature.zero_crossing_rate(
                audio,
                hop_length=self.hop_length
            )
            return zcr
        except Exception as e:
            logger.error(f"Error extracting ZCR: {e}")
            return None
    
    def extract_spectral_centroid(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract Spectral Centroid.
        
        Args:
            audio: Audio signal (1D array)
            
        Returns:
            Spectral centroid values (1 x time_steps)
        """
        try:
            spec_cent = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            return spec_cent
        except Exception as e:
            logger.error(f"Error extracting spectral centroid: {e}")
            return None
    
    def extract_spectral_rolloff(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract Spectral Rolloff.
        
        Args:
            audio: Audio signal (1D array)
            
        Returns:
            Spectral rolloff values (1 x time_steps)
        """
        try:
            rolloff = librosa.feature.spectral_rolloff(
                y=audio,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            return rolloff
        except Exception as e:
            logger.error(f"Error extracting spectral rolloff: {e}")
            return None
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all available features.
        
        Args:
            audio: Audio signal (1D array)
            
        Returns:
            Dictionary with all extracted features
        """
        features = {
            'mfcc': self.extract_mfcc(audio),
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'zcr': self.extract_zcr(audio),
            'spectral_centroid': self.extract_spectral_centroid(audio),
            'spectral_rolloff': self.extract_spectral_rolloff(audio)
        }
        # Remove None values
        features = {k: v for k, v in features.items() if v is not None}
        return features
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Make the feature extractor callable.
        
        Args:
            audio: Audio signal
            
        Returns:
            Mel-spectrogram by default
        """
        return self.extract_mel_spectrogram(audio)
