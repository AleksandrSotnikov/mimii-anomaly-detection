"""Data loader for MIMII dataset."""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class MIMIIDataset(Dataset):
    """PyTorch Dataset for MIMII sound data."""
    
    def __init__(
        self,
        data_path: str,
        machine_type: str,
        mode: str = "train",
        sample_rate: int = 16000,
        duration: float = 10.0,
        is_anomaly: bool = False
    ):
        """
        Args:
            data_path: Path to MIMII dataset root directory
            machine_type: Type of machine ('valve', 'pump', 'fan', 'slider')
            mode: 'train' or 'test'
            sample_rate: Audio sample rate
            duration: Audio clip duration in seconds
            is_anomaly: If True, load anomalous sounds; else normal sounds
        """
        self.data_path = Path(data_path)
        self.machine_type = machine_type
        self.mode = mode
        self.sample_rate = sample_rate
        self.duration = duration
        self.is_anomaly = is_anomaly
        
        self.samples_dir = self._get_samples_directory()
        self.audio_files = self._load_file_list()
        
        logger.info(
            f"Loaded {len(self.audio_files)} {machine_type} "
            f"{'anomaly' if is_anomaly else 'normal'} files for {mode}"
        )
    
    def _get_samples_directory(self) -> Path:
        """Get the directory containing audio samples."""
        # MIMII structure: data_path/machine_type/id_XX/normal or anomaly
        status = "anomaly" if self.is_anomaly else "normal"
        
        # Find all model directories (id_00, id_02, id_04, id_06)
        machine_path = self.data_path / self.machine_type
        if not machine_path.exists():
            raise ValueError(f"Machine type directory not found: {machine_path}")
        
        return machine_path
    
    def _load_file_list(self) -> List[Path]:
        """Load list of audio files."""
        audio_files = []
        status = "anomaly" if self.is_anomaly else "normal"
        
        # Iterate through model IDs
        for model_dir in self.samples_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            status_dir = model_dir / status
            if not status_dir.exists():
                continue
            
            # Get all .wav files
            wav_files = list(status_dir.glob("*.wav"))
            audio_files.extend(wav_files)
        
        if len(audio_files) == 0:
            logger.warning(f"No audio files found in {self.samples_dir}")
        
        return sorted(audio_files)
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Load and return audio sample.
        
        Returns:
            audio: Audio waveform as numpy array
            label: 0 for normal, 1 for anomaly
        """
        audio_path = self.audio_files[idx]
        
        # Load audio
        audio, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            duration=self.duration
        )
        
        # Pad if necessary
        target_length = int(self.sample_rate * self.duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        label = 1 if self.is_anomaly else 0
        
        return audio, label


def get_dataloaders(
    data_path: str,
    machine_type: str,
    batch_size: int = 32,
    validation_split: float = 0.2,
    num_workers: int = 4,
    sample_rate: int = 16000,
    duration: float = 10.0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.
    
    Args:
        data_path: Path to MIMII dataset
        machine_type: Type of machine
        batch_size: Batch size for dataloaders
        validation_split: Fraction of training data for validation
        num_workers: Number of worker processes
        sample_rate: Audio sample rate
        duration: Audio duration in seconds
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Training data (normal sounds only for autoencoder)
    train_dataset = MIMIIDataset(
        data_path=data_path,
        machine_type=machine_type,
        mode="train",
        sample_rate=sample_rate,
        duration=duration,
        is_anomaly=False
    )
    
    # Split training data for validation
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    from torch.utils.data import random_split
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test data (both normal and anomalous)
    test_normal = MIMIIDataset(
        data_path=data_path,
        machine_type=machine_type,
        mode="test",
        sample_rate=sample_rate,
        duration=duration,
        is_anomaly=False
    )
    
    test_anomaly = MIMIIDataset(
        data_path=data_path,
        machine_type=machine_type,
        mode="test",
        sample_rate=sample_rate,
        duration=duration,
        is_anomaly=True
    )
    
    from torch.utils.data import ConcatDataset
    test_dataset = ConcatDataset([test_normal, test_anomaly])
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
