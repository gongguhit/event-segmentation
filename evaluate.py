import os
import argparse
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path

from data.mvsec_dataset import get_mvsec_dataloaders
from models.event_segmentation_model import get_model
from utils.logger import setup_logger
from utils.metrics import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate event segmentation model')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset (overrides config)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save evaluation results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    return parser.parse_args()

def visualize_prediction(events, pred, target, idx, output_dir):
    """
    Visualize model prediction and save to file.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Event representation (take mean across time bins if voxel grid)
    if events.shape[0] > 3:  # More than 3 channels (probably voxel grid)
        event_vis = events.mean(dim=0).cpu().numpy()
    else:
        # Combine positive and negative events for visualization
        event_vis = events[0].cpu().numpy() - events[1].cpu().numpy() if events.shape[0] > 1 else events[0].cpu().numpy()
    
    # Normalize for visualization
    event_vis = (event_vis - event_vis.min()) / (event_vis.max() - event_vis.min() + 1e-6)
    
    pred = pred.argmax(dim=0).cpu().numpy()
    target = target.cpu().numpy()
    
    # Plot
    axes[0].imshow(event_vis, cmap='gray')
    axes[0].set_title('Event Representation')
    axes[0].axis('off')
    
    axes[1].imshow(pred, cmap='viridis')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    axes[2].imshow(target, cmap='viridis')
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    # Save figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'sample_{idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def evaluate(model, val_loader, device, output_dir, visualize=False, num_samples=10):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    metrics_dict = {'iou': [], 'accuracy': [], 'precision': [], 'recall': []}
    
    # Create directory for visualizations
    vis_dir = os.path.join(output_dir, 'visualizations')
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
    
    # Sample indices for visualization
    num_batches = len(val_loader)
    vis_batch_indices = np.random.choice(num_batches, min(num_samples, num_batches), replace=False)
    vis_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            # Move data to device
            events = batch['events'].to(device)
            images = batch['image'].to(device)
            targets = batch['segmentation'].to(device)
            
            # Forward pass
            outputs = model(events)
            
            # Compute metrics
            batch_metrics = compute_metrics(outputs['logits'], targets)
            for k, v in batch_metrics.items():
                metrics_dict[k].append(v)
            
            # Visualize predictions
            if visualize and batch_idx in vis_batch_indices and vis_count < num_samples:
                for i in range(min(events.shape[0], 1)):  # Visualize first sample in batch
                    visualize_prediction(
                        events[i], outputs['logits'][i], targets[i], 
                        f"{batch_idx}_{i}", vis_dir
                    )
                    vis_count += 1
    
    # Compute average metrics
    results = {}
    for k, v in metrics_dict.items():
        results[k] = np.mean(v)
    
    return results

def main():
    """
    Main evaluation function.
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger('evaluate', output_dir / 'evaluation.log')
    logger.info(f"Configuration: {config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders (only validation needed)
    logger.info("Creating data loader...")
    _, val_loader = get_mvsec_dataloaders(config)
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Create and load model
    logger.info("Loading model...")
    model = get_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Evaluate model
    logger.info("Starting evaluation...")
    results = evaluate(
        model, val_loader, device, output_dir,
        visualize=args.visualize, num_samples=args.num_samples
    )
    
    # Log results
    logger.info("Evaluation results:")
    for k, v in results.items():
        logger.info(f"{k}: {v:.4f}")
    
    # Save results to file
    with open(output_dir / 'results.txt', 'w') as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    
    logger.info(f"Results saved to {output_dir}")

if __name__ == '__main__':
    main() 