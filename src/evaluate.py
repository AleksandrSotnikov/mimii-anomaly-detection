"""Evaluation module for anomaly detection."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score, 
    confusion_matrix, roc_curve, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for anomaly detection models."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """Initialize evaluator.
        
        Args:
            model: Trained model
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def get_reconstruction_errors(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get reconstruction errors for all samples.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Tuple of (errors, labels)
        """
        errors = []
        labels = []
        
        with torch.no_grad():
            for batch, batch_labels in data_loader:
                batch = batch.to(self.device)
                error = self.model.reconstruction_error(batch)
                errors.extend(error.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
        
        return np.array(errors), np.array(labels)
    
    @staticmethod
    def calculate_threshold(scores: np.ndarray, percentile: float = 95) -> float:
        """Calculate anomaly threshold based on percentile.
        
        Args:
            scores: Anomaly scores (reconstruction errors)
            percentile: Percentile for threshold (default: 95)
            
        Returns:
            Threshold value
        """
        return np.percentile(scores, percentile)
    
    def evaluate(self, test_loader: DataLoader, 
                threshold: Optional[float] = None,
                percentile: float = 95) -> Dict:
        """Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            threshold: Anomaly threshold (if None, calculated from data)
            percentile: Percentile for automatic threshold calculation
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Starting evaluation...")
        
        # Get reconstruction errors
        errors, labels = self.get_reconstruction_errors(test_loader)
        
        # Calculate threshold if not provided
        if threshold is None:
            # Use only normal samples (label=0) for threshold calculation
            normal_errors = errors[labels == 0]
            if len(normal_errors) > 0:
                threshold = self.calculate_threshold(normal_errors, percentile)
            else:
                threshold = self.calculate_threshold(errors, percentile)
        
        logger.info(f"Using threshold: {threshold:.6f}")
        
        # Predictions
        predictions = (errors > threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'threshold': threshold,
            'auc_roc': roc_auc_score(labels, errors) if len(np.unique(labels)) > 1 else 0.0,
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1_score': f1_score(labels, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, predictions),
            'errors': errors,
            'labels': labels,
            'predictions': predictions
        }
        
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def plot_roc_curve(self, labels: np.ndarray, scores: np.ndarray, 
                       save_path: Optional[Path] = None) -> None:
        """Plot ROC curve.
        
        Args:
            labels: True labels
            scores: Anomaly scores
            save_path: Path to save plot
        """
        if len(np.unique(labels)) < 2:
            logger.warning("Cannot plot ROC curve with only one class")
            return
        
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                             save_path: Optional[Path] = None) -> None:
        """Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_error_distribution(self, errors: np.ndarray, labels: np.ndarray,
                               threshold: float, save_path: Optional[Path] = None) -> None:
        """Plot distribution of reconstruction errors.
        
        Args:
            errors: Reconstruction errors
            labels: True labels
            threshold: Anomaly threshold
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        
        normal_errors = errors[labels == 0]
        anomaly_errors = errors[labels == 1]
        
        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue')
        if len(anomaly_errors) > 0:
            plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color='red')
        
        plt.axvline(threshold, color='green', linestyle='--', linewidth=2, 
                   label=f'Threshold = {threshold:.4f}')
        
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Error distribution plot saved to {save_path}")
        plt.close()
    
    def generate_report(self, metrics: Dict, output_dir: Path) -> None:
        """Generate evaluation report with plots.
        
        Args:
            metrics: Evaluation metrics dictionary
            output_dir: Directory to save report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot ROC curve
        self.plot_roc_curve(metrics['labels'], metrics['errors'], 
                           output_dir / 'roc_curve.png')
        
        # Plot confusion matrix
        self.plot_confusion_matrix(metrics['confusion_matrix'],
                                   output_dir / 'confusion_matrix.png')
        
        # Plot error distribution
        self.plot_error_distribution(metrics['errors'], metrics['labels'],
                                    metrics['threshold'],
                                    output_dir / 'error_distribution.png')
        
        # Save metrics to text file
        with open(output_dir / 'metrics.txt', 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("Anomaly Detection Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Threshold: {metrics['threshold']:.6f}\n")
            f.write(f"AUC-ROC: {metrics['auc_roc']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1_score']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(metrics['confusion_matrix']) + "\n")
        
        logger.info(f"Evaluation report saved to {output_dir}")
