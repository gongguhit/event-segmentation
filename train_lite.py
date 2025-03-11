#!/usr/bin/env python3
# Lightweight training script for event-based segmentation

import os
import argparse
import yaml
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from data.mvsec_dataset import get_mvsec_dataloaders
from models.lite_model import get_lite_model
from utils.metrics import compute_metrics
from utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train the event segmentation model (lite version)')
    parser.add_argument('--config', type=str, default='config/lite.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset (overrides config)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if MPS/CUDA is available')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()

def train_epoch(model, train_loader, optimizer, device, epoch, config, logger):
    """Run one training epoch"""
    model.train()
    running_loss = 0.0
    
    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}") as pbar:
        for i, batch in pbar:
            # Move data to device
            events = batch['events'].to(device)
            targets = batch['segmentation'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(events)
            
            # Compute loss directly
            logits = outputs['logits']
            loss = F.cross_entropy(logits, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': running_loss / (i + 1)})
    
    # Calculate average loss
    avg_loss = running_loss / len(train_loader)
    logger.info(f"Epoch {epoch} - Training Loss: {avg_loss:.4f}")
    
    return avg_loss

def validate(model, val_loader, device, epoch, logger):
    """Run validation"""
    model.eval()
    val_loss = 0.0
    metrics = {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    num_samples = 0
    
    with torch.no_grad():
        with tqdm(val_loader, desc=f"Validation", leave=False) as pbar:
            for i, batch in enumerate(pbar):
                # Move data to device
                events = batch['events'].to(device)
                targets = batch['segmentation'].to(device)
                
                # Forward pass
                outputs = model(events)
                
                # Compute loss
                logits = outputs['logits']
                loss = F.cross_entropy(logits, targets)
                
                # Update validation loss
                val_loss += loss.item()
                
                # Compute predictions
                preds = torch.argmax(logits, dim=1)
                
                # Update metrics
                batch_metrics = compute_metrics(preds, targets)
                for k in metrics.keys():
                    metrics[k] += batch_metrics[k] * events.size(0)
                
                num_samples += events.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'val_loss': val_loss / (i + 1),
                    'val_iou': metrics['iou'] / num_samples
                })
    
    # Calculate average values
    val_loss /= len(val_loader)
    for k in metrics:
        metrics[k] /= num_samples
    
    # Log metrics
    logger.info(f"Epoch {epoch} - Validation Loss: {val_loss:.4f}, "
                f"IoU: {metrics['iou']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}")
    
    return val_loss, metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override data path if specified
    if args.data_path:
        config['data']['data_path'] = args.data_path
    
    # Setup logging
    log_dir = Path(config['logging']['save_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('train_lite', log_dir / 'train_lite.log')
    
    # Log configuration
    logger.info(f"Configuration: {config}")
    
    # Set device
    if args.cpu:
        device = torch.device('cpu')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using device: Apple Metal (MPS)")
        logger.info("Applied MPS-specific optimizations for Apple Silicon")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        logger.info("Using device: CPU")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = get_mvsec_dataloaders(config)
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Create model - use the lite model
    logger.info("Creating lightweight model...")
    model = get_lite_model(config)
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create learning rate scheduler
    num_epochs = config['training']['epochs']
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Start training
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        # Train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, config, logger)
        
        # Validate
        val_loss, metrics = validate(model, val_loader, device, epoch, logger)
        
        # Save checkpoint
        checkpoint_dir = Path(config['logging']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'val_loss': val_loss,
            'metrics': metrics
        }
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save regular checkpoint
        if (epoch + 1) % config['logging']['save_interval'] == 0:
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
            logger.info(f"Saved checkpoint at epoch {epoch}")
    
    logger.info(f"Training completed after {num_epochs} epochs")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main() 