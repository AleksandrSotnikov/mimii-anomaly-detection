#!/usr/bin/env python3
"""Evaluation script for trained models."""

import argparse
import sys
from pathlib import Path

from src import (
    MIMIIDataLoader,
    FeatureExtractor,
    SoundAnomalyAutoencoder,
    Evaluator
)
from src.utils import setup_logging, get_device, set_seed
import logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate MIMII anomaly detection model'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/',
        help='Path to MIMII dataset'
    )
    
    parser.add_argument(
        '--machine-type',
        type=str,
        default='fan',
        choices=['fan', 'pump', 'valve', 'slider'],
        help='Type of machine'
    )
    
    parser.add_argument(
        '--model-id',
        type=str,
        default='00',
        help='Model ID'
    )
    
    parser.add_argument(
        '--db-level',
        type=int,
        default=6,
        help='dB level'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    setup_logging()
    set_seed(42)
    device = get_device(use_cuda=not args.no_cuda)
    
    logger.info("="*60)
    logger.info("MIMII Anomaly Detection Evaluation")
    logger.info("="*60)
    
    # Load data
    feature_extractor = FeatureExtractor()
    data_loader = MIMIIDataLoader(
        data_dir=args.data_dir,
        machine_type=args.machine_type,
        model_id=args.model_id,
        db_level=args.db_level
    )
    
    _, _, test_loader = data_loader.get_dataloaders(
        batch_size=args.batch_size,
        feature_extractor=feature_extractor
    )
    
    # Load model
    model = SoundAnomalyAutoencoder()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    logger.info(f"Model loaded from {args.checkpoint}")
    
    # Evaluate
    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(test_loader)
    
    # Generate report
    output_dir = Path(args.output_dir) / f"{args.machine_type}_id{args.model_id}"
    evaluator.generate_report(metrics, output_dir)
    
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    import torch
    main()
