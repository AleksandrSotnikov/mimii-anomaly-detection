"""Data loader for MIMII Dataset."""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging
from torch.utils.data import Dataset, DataLoader
import torch

logger = logging.getLogger(__name__)


class MIMIIAudioDataset(Dataset):
    """PyTorch Dataset for MIMII audio files."""
    
    def __init__(self, file_paths: List[str], labels: List[int], 
                 sr: int = 16000, feature_extractor=None, segment_length: int = 5):
        """Initialize dataset.
        
        Args:
            file_paths: List of audio file paths
            labels: List of labels (0=normal, 1=anomaly)
            sr: Sampling rate
            feature_extractor: FeatureExtractor object
            segment_length: Length of audio segments in seconds
        """
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.feature_extractor = feature_extractor
        self.segment_length = segment_length
        self.segment_samples = segment_length * sr
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index.
        
        Returns:
            Tuple of (features, label)
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            audio, _ = librosa.load(file_path, sr=self.sr, mono=True)
            
            # Segment audio
            if len(audio) > self.segment_samples:
                # Random crop for training
                start = np.random.randint(0, len(audio) - self.segment_samples)
                audio = audio[start:start + self.segment_samples]
            else:
                # Pad if too short
                audio = np.pad(audio, (0, max(0, self.segment_samples - len(audio))), 
                             mode='constant')
            
            if self.feature_extractor is not None:
                features = self.feature_extractor.extract_mel_spectrogram(audio)
                # Flatten or use as is depending on model input
                if features is not None:
                    features = torch.FloatTensor(features)
                else:
                    features = torch.zeros(128, 157)  # Default shape
            else:
                features = torch.FloatTensor(audio)
            
            return features, label
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return torch.zeros(128, 157), 0


class MIMIIDataLoader:
    """Load and preprocess MIMII Dataset.
    
    Attributes:
        data_dir: Path to MIMII dataset directory
        machine_type: Type of machine ('fan', 'pump', 'valve', 'slider')
        model_id: Specific model ID to load
        sr: Sampling rate (default: 16000)
    """
    
    MACHINE_TYPES = ['fan', 'pump', 'valve', 'slider']
    
    def __init__(self, data_dir: str, machine_type: str = 'fan', 
                 model_id: Optional[str] = None, sr: int = 16000,
                 db_level: int = 6):
        """Initialize data loader.
        
        Args:
            data_dir: Path to MIMII dataset
            machine_type: Type of machine to load
            model_id: Specific model ID (e.g., '00', '02', '04', '06')
            sr: Sampling rate
            db_level: dB level (6, 0, -6)
        """
        self.data_dir = Path(data_dir)
        self.machine_type = machine_type
        self.model_id = model_id
        self.sr = sr
        self.db_level = db_level
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
        
        if machine_type not in self.MACHINE_TYPES:
            raise ValueError(f"Machine type must be one of {self.MACHINE_TYPES}")
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file using librosa.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sampling_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sr, mono=True)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, None
    
    def get_file_paths(self, machine_dir: Path, label: str = 'normal') -> List[str]:
        """Get list of audio files for specific label.
        
        Args:
            machine_dir: Directory with machine data
            label: 'normal' or 'anomaly'
            
        Returns:
            List of file paths
        """
        files = []
        pattern = f"*{label}*.wav"
        
        if machine_dir.exists():
            for file_path in machine_dir.glob(pattern):
                files.append(str(file_path))
        
        return sorted(files)
    
    def prepare_datasets(self, test_size: float = 0.2, 
                        feature_extractor=None) -> Tuple[Dataset, Dataset, Dataset]:
        """Prepare train, validation, and test datasets.
        
        Args:
            test_size: Fraction of data for testing
            feature_extractor: FeatureExtractor instance
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Construct path to machine data
        # MIMII structure: data/{db_level}dB_{machine_type}/id_{model_id}
        db_str = f"{self.db_level}dB" if self.db_level >= 0 else f"min{abs(self.db_level)}dB"
        machine_dir = self.data_dir / f"{db_str}_{self.machine_type}"
        
        if self.model_id:
            machine_dir = machine_dir / f"id_{self.model_id}"
        
        logger.info(f"Looking for data in: {machine_dir}")
        
        # Get normal and anomaly files
        normal_files = self.get_file_paths(machine_dir, 'normal')
        anomaly_files = self.get_file_paths(machine_dir, 'anomaly')
        
        logger.info(f"Found {len(normal_files)} normal files, {len(anomaly_files)} anomaly files")
        
        # Create labels
        normal_labels = [0] * len(normal_files)
        anomaly_labels = [1] * len(anomaly_files)
        
        # Split normal data for training (80%) and validation (20%)
        split_idx = int(len(normal_files) * (1 - test_size))
        train_files = normal_files[:split_idx]
        val_files = normal_files[split_idx:]
        
        # Test set includes some normal + all anomaly
        test_files = val_files + anomaly_files
        test_labels = [0] * len(val_files) + anomaly_labels
        
        train_labels = [0] * len(train_files)
        val_labels = [0] * len(val_files)
        
        # Create datasets
        train_dataset = MIMIIAudioDataset(train_files, train_labels, self.sr, feature_extractor)
        val_dataset = MIMIIAudioDataset(val_files, val_labels, self.sr, feature_extractor)
        test_dataset = MIMIIAudioDataset(test_files, test_labels, self.sr, feature_extractor)
        
        return train_dataset, val_dataset, test_dataset
    
    def get_dataloaders(self, batch_size: int = 32, num_workers: int = 4,
                       feature_extractor=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get DataLoaders for train, validation, and test.
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            feature_extractor: FeatureExtractor instance
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_ds, val_ds, test_ds = self.prepare_datasets(feature_extractor=feature_extractor)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
