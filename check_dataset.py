import os
import argparse
import yaml
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from data.mvsec_dataset import MVSECDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Check MVSEC dataset loading')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset (overrides config)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./dataset_check',
                        help='Directory to save visualizations')
    return parser.parse_args()

def explore_h5_structure(file_path):
    """Print the HDF5 file structure for debugging"""
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    with h5py.File(file_path, 'r') as f:
        print(f"\nExploring structure of {file_path}:")
        f.visititems(print_structure)

def visualize_sample(sample, idx, output_dir):
    """Visualize a sample from the dataset"""
    events = sample['events']
    image = sample['image']
    segmentation = sample['segmentation']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Event representation (take mean across time bins if voxel grid)
    if events.shape[0] > 3:  # More than 3 channels (probably voxel grid)
        event_vis = events.mean(dim=0).cpu().numpy()
    else:
        # Combine positive and negative events for visualization
        event_vis = events[0].cpu().numpy() - events[1].cpu().numpy() if events.shape[0] > 1 else events[0].cpu().numpy()
    
    # Normalize for visualization
    event_vis = (event_vis - event_vis.min()) / (event_vis.max() - event_vis.min() + 1e-6)
    
    # Convert image and segmentation for visualization
    image_vis = image.squeeze(0).cpu().numpy() if image.shape[0] == 1 else image.permute(1, 2, 0).cpu().numpy()
    seg_vis = segmentation.cpu().numpy()
    
    # Plot
    axes[0].imshow(event_vis, cmap='gray')
    axes[0].set_title('Event Representation')
    axes[0].axis('off')
    
    axes[1].imshow(image_vis, cmap='gray' if image_vis.ndim == 2 else None)
    axes[1].set_title('Image')
    axes[1].axis('off')
    
    axes[2].imshow(seg_vis, cmap='viridis')
    axes[2].set_title('Segmentation (Ground Truth/Dummy)')
    axes[2].axis('off')
    
    # Save figure
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'sample_{idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved visualization for sample {idx} to {save_path}")
    print(f"Event tensor shape: {events.shape}")
    print(f"Image tensor shape: {image.shape}")
    print(f"Segmentation tensor shape: {segmentation.shape}")
    print("-" * 40)

def main():
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
    
    # Print config for debugging
    print("Configuration:")
    print(f"Dataset path: {config['data']['data_path']}")
    print(f"Event representation: {config['data']['event_representation']}")
    print(f"Time bins: {config['data']['time_bins']}")
    
    # First, explore the structure of one HDF5 file
    data_path = config['data']['data_path']
    for seq in os.listdir(data_path):
        seq_path = os.path.join(data_path, seq)
        if os.path.isdir(seq_path) and 'calib' not in seq:
            for f in os.listdir(seq_path):
                if f.endswith('_data.hdf5'):
                    data_file = os.path.join(seq_path, f)
                    explore_h5_structure(data_file)
                    # Just explore one file for brevity
                    break
            # Just explore one sequence for brevity
            break
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = MVSECDataset(
        root_dir=config['data']['data_path'],
        split='train',
        time_bins=config['data']['time_bins'],
        event_representation=config['data']['event_representation']
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check a few samples
    num_samples = min(args.num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    print(f"\nChecking {num_samples} random samples:")
    for i, idx in enumerate(indices):
        try:
            sample = dataset[idx]
            print(f"Sample {i+1}/{num_samples} (index {idx}):")
            visualize_sample(sample, idx, output_dir)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
    
    print(f"\nCheck complete. Visualizations saved to {output_dir}")

if __name__ == '__main__':
    main() 