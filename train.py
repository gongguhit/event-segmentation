import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time
from pathlib import Path

from data.mvsec_dataset import get_mvsec_dataloaders
from models.event_segmentation_model import get_model
from utils.logger import setup_logger
from utils.metrics import compute_metrics
from utils.device_utils import get_device, to_device, mps_fix_for_training

# Optional GPT-4 integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train event segmentation model')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset (overrides config)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if MPS/CUDA is available')
    return parser.parse_args()

def get_lr_scheduler(optimizer, config, num_epochs):
    """
    Create learning rate scheduler based on configuration.
    """
    if config['training']['lr_scheduler'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs - config['training']['warmup_epochs']
        )
    elif config['training']['lr_scheduler'] == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    else:
        return None

def save_checkpoint(model, optimizer, epoch, save_path, config, loss):
    """
    Save model checkpoint.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': loss
    }, save_path)

def train_epoch(model, train_loader, optimizer, device, epoch, config, logger, writer):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_transition_loss = 0.0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    
    for i, batch in pbar:
        # Move data to device
        batch = to_device(batch, device)
        events = batch['events']
        images = batch['image']
        targets = batch['segmentation']
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(events)
        
        # Compute loss
        loss_dict = model.compute_loss(outputs, targets, config)
        loss = loss_dict['total_loss']
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update running losses
        running_loss += loss.item()
        running_seg_loss += loss_dict['seg_loss'].item()
        running_transition_loss += loss_dict['transition_loss'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (i + 1),
            'seg_loss': running_seg_loss / (i + 1),
            'trans_loss': running_transition_loss / (i + 1)
        })
    
    # Calculate epoch losses
    epoch_loss = running_loss / len(train_loader)
    epoch_seg_loss = running_seg_loss / len(train_loader)
    epoch_transition_loss = running_transition_loss / len(train_loader)
    
    # Log metrics
    logger.info(f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, "
                f"Seg Loss: {epoch_seg_loss:.4f}, "
                f"Transition Loss: {epoch_transition_loss:.4f}")
    
    # Write to tensorboard
    writer.add_scalar('train/total_loss', epoch_loss, epoch)
    writer.add_scalar('train/seg_loss', epoch_seg_loss, epoch)
    writer.add_scalar('train/transition_loss', epoch_transition_loss, epoch)
    
    return epoch_loss

def validate(model, val_loader, device, epoch, config, logger, writer):
    """
    Validate the model.
    """
    model.eval()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_transition_loss = 0.0
    
    # Initialize metrics
    metrics = {'iou': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            batch = to_device(batch, device)
            events = batch['events']
            images = batch['image']
            targets = batch['segmentation']
            
            # Forward pass
            outputs = model(events)
            
            # Compute loss
            loss_dict = model.compute_loss(outputs, targets, config)
            
            # Update running losses
            running_loss += loss_dict['total_loss'].item()
            running_seg_loss += loss_dict['seg_loss'].item()
            running_transition_loss += loss_dict['transition_loss'].item()
            
            # Compute metrics
            batch_metrics = compute_metrics(outputs['logits'], targets)
            for k in metrics:
                metrics[k] += batch_metrics[k]
    
    # Calculate validation metrics
    for k in metrics:
        metrics[k] /= len(val_loader)
    
    # Calculate validation losses
    val_loss = running_loss / len(val_loader)
    val_seg_loss = running_seg_loss / len(val_loader)
    val_transition_loss = running_transition_loss / len(val_loader)
    
    # Log metrics
    logger.info(f"Epoch {epoch} - Validation Loss: {val_loss:.4f}, "
                f"Seg Loss: {val_seg_loss:.4f}, "
                f"Transition Loss: {val_transition_loss:.4f}")
    logger.info(f"Metrics - IoU: {metrics['iou']:.4f}, "
                f"Accuracy: {metrics['accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}")
    
    # Write to tensorboard
    writer.add_scalar('val/total_loss', val_loss, epoch)
    writer.add_scalar('val/seg_loss', val_seg_loss, epoch)
    writer.add_scalar('val/transition_loss', val_transition_loss, epoch)
    writer.add_scalar('val/iou', metrics['iou'], epoch)
    writer.add_scalar('val/accuracy', metrics['accuracy'], epoch)
    
    return val_loss, metrics

def gpt4_analysis(model, config, val_loss, metrics, epoch):
    """
    Use Azure OpenAI GPT-4o to analyze model performance and suggest improvements.
    """
    if not OPENAI_AVAILABLE or not config['training']['use_gpt4o']:
        return None
    
    # Setup Azure OpenAI client
    client = openai.AzureOpenAI(
        api_key=config['training']['azure_openai']['api_key'],
        api_version="2023-05-15",
        azure_endpoint=config['training']['azure_openai']['endpoint']
    )
    
    # Prepare prompt
    prompt = f"""
    Analyze the performance of an event-based segmentation model with the following metrics at epoch {epoch}:
    
    Validation Loss: {val_loss:.4f}
    IoU: {metrics['iou']:.4f}
    Accuracy: {metrics['accuracy']:.4f}
    Precision: {metrics['precision']:.4f}
    Recall: {metrics['recall']:.4f}
    
    Model configuration:
    - Embedding dimensions: {config['model']['embedding_dims']}
    - Number of attention heads: {config['model']['num_heads']}
    - Alpha values: {config['model']['alpha_values']}
    - Layer weights: {config['model']['layer_weights']}
    
    Please analyze the performance and suggest specific improvements that could be made to:
    1. The model architecture
    2. The training process
    3. The token transition mechanism
    
    Focus on practical suggestions that could improve segmentation quality.
    """
    
    try:
        # Call Azure OpenAI
        response = client.chat.completions.create(
            model=config['training']['azure_openai']['deployment_name'],
            messages=[
                {"role": "system", "content": "You are an AI assistant that specializes in computer vision and deep learning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error using Azure OpenAI: {e}")
        return None

def main():
    """
    Main training function.
    """
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override data path if provided
    if args.data_path:
        config['data']['data_path'] = args.data_path
    
    # Create output directory
    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger('train', save_dir / 'train.log')
    logger.info(f"Configuration: {config}")
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=str(save_dir / 'tensorboard'))
    
    # Set device
    if args.cpu:
        device = torch.device('cpu')
        logger.info("Forcing CPU usage as requested")
    else:
        device = get_device(logger)
    
    # Apply MPS-specific fixes if needed
    if device.type == 'mps':
        mps_fix_for_training()
        logger.info("Applied MPS-specific optimizations for Apple Silicon")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = get_mvsec_dataloaders(config)
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = get_model(config)
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create learning rate scheduler
    lr_scheduler = get_lr_scheduler(optimizer, config, config['training']['epochs'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # Load checkpoint to CPU first to prevent GPU OOM errors
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        # Move model to the selected device
        model = model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer states to the selected device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # Train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, config, logger, writer)
        
        # Validate
        if epoch % config['logging']['eval_interval'] == 0:
            val_loss, metrics = validate(model, val_loader, device, epoch, config, logger, writer)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch,
                    save_dir / 'best_model.pth',
                    config, val_loss
                )
                logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
            
            # Use GPT-4o for analysis (if configured)
            if (epoch + 1) % 10 == 0 and config['training']['use_gpt4o']:
                analysis = gpt4_analysis(model, config, val_loss, metrics, epoch)
                if analysis:
                    logger.info(f"GPT-4o Analysis:\n{analysis}")
        
        # Save regular checkpoint
        if epoch % config['logging']['save_interval'] == 0:
            save_checkpoint(
                model, optimizer, epoch,
                save_dir / f'checkpoint_epoch_{epoch}.pth',
                config, train_loss
            )
        
        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
    
    # Save final model
    save_checkpoint(
        model, optimizer, config['training']['epochs'] - 1,
        save_dir / 'final_model.pth',
        config, train_loss
    )
    
    logger.info("Training complete!")
    writer.close()

if __name__ == '__main__':
    main() 