#!/usr/bin/env python3
"""Main training script for MIMII anomaly detection."""

import argparse
import sys
from pathlib import Path

import torch

from src import (
    MIMIIDataLoader,
    FeatureExtractor,
    SoundAnomalyAutoencoder,
    VariationalAutoencoder,
    Trainer,
    Evaluator
)
from src.utils import (
    setup_logging,
    load_config,
    set_seed,
    get_device,
    count_parameters
)
import logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MIMII anomaly detection model'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
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
        help='Model ID (00, 02, 04, 06)'
    )
    
    parser.add_argument(
        '--db-level',
        type=int,
        default=6,
        choices=[-6, 0, 6],
        help='dB level of dataset'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='autoencoder',
        choices=['autoencoder', 'vae'],
        help='Type of model'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for logs'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA'
    )
    
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        help='Only evaluate (requires trained model)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume/evaluate'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    setup_logging(args.log_dir)
    set_seed(args.seed)
    device = get_device(use_cuda=not args.no_cuda)
    
    logger.info("="*60)
    logger.info("MIMII Anomaly Detection Training")
    logger.info("="*60)
    logger.info(f"Machine type: {args.machine_type}")
    logger.info(f"Model ID: {args.model_id}")
    logger.info(f"dB level: {args.db_level}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")
    
    # Load configuration if exists
    if Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = {}
        logger.warning(f"Config file not found: {args.config}")
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    logger.info("Feature extractor initialized")
    
    # Load data
    logger.info("Loading MIMII dataset...")
    data_loader = MIMIIDataLoader(
        data_dir=args.data_dir,
        machine_type=args.machine_type,
        model_id=args.model_id,
        db_level=args.db_level
    )
    
    try:
        train_loader, val_loader, test_loader = data_loader.get_dataloaders(
            batch_size=args.batch_size,
            num_workers=4,
            feature_extractor=feature_extractor
        )
        logger.info(f"Data loaded successfully")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.error("Make sure the MIMII dataset is downloaded and extracted to the data directory")
        sys.exit(1)
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    if args.model_type == 'autoencoder':
        model = SoundAnomalyAutoencoder()
    else:
        model = VariationalAutoencoder()
    
    model = model.to(device)
    num_params = count_parameters(model)
    logger.info(f"Model created with {num_params:,} trainable parameters")
    
    # Training or evaluation
    if args.evaluate_only:
        if args.checkpoint is None:
            logger.error("--checkpoint required for evaluation")
            sys.exit(1)
        
        logger.info("Evaluation mode")
        trainer = Trainer(model, device=device)
        trainer.load_model(Path(args.checkpoint))
        
        evaluator = Evaluator(model, device=device)
        metrics = evaluator.evaluate(test_loader)
        
        # Generate report
        output_dir = Path('results') / f"{args.machine_type}_id{args.model_id}"
        evaluator.generate_report(metrics, output_dir)
        logger.info(f"Evaluation complete. Results saved to {output_dir}")
    else:
        # Training
        logger.info("Starting training...")
        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=args.lr
        )
        
        if args.checkpoint:
            logger.info(f"Resuming from checkpoint: {args.checkpoint}")
            trainer.load_model(Path(args.checkpoint))
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            patience=20,
            checkpoint_dir=args.checkpoint_dir
        )
        
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {history['best_val_loss']:.6f}")
        logger.info(f"Training time: {history['training_time']:.2f} seconds")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        trainer.load_model(Path(args.checkpoint_dir) / 'best_model.pth')
        evaluator = Evaluator(model, device=device)
        metrics = evaluator.evaluate(test_loader)
        
        # Generate report
        output_dir = Path('results') / f"{args.machine_type}_id{args.model_id}"
        evaluator.generate_report(metrics, output_dir)
        logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
