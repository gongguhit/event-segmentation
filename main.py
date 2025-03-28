import argparse
import subprocess
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Event-based Segmentation with Token Transition')
    parser.add_argument('action', type=str, choices=['train', 'evaluate', 'download'],
                        help='Action to perform: train, evaluate, or download dataset')
    
    # Common arguments
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset (overrides config)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if MPS/CUDA is available')
    parser.add_argument('--lite', action='store_true',
                        help='Use lightweight training with reduced dataset and model size')
    
    # Train-specific arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (for training)')
    
    # Evaluate-specific arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (for evaluation)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save evaluation results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions (for evaluation)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize (for evaluation)')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum number of batches to evaluate (for quick testing)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size from config')
    
    return parser.parse_args()

def train(args):
    """Run the training script with the specified arguments."""
    # Use the lite training script if --lite flag is specified
    if args.lite:
        script = 'train_lite.py'
        if args.config == 'config/default.yaml':
            args.config = 'config/lite.yaml'
    else:
        script = 'train.py'
    
    cmd = [sys.executable, script,
           '--config', args.config]
    
    if args.data_path:
        cmd.extend(['--data_path', args.data_path])
    
    if args.resume:
        cmd.extend(['--resume', args.resume])
        
    if args.cpu:
        cmd.append('--cpu')
    
    subprocess.run(cmd)

def evaluate(args):
    """Run the evaluation script with the specified arguments."""
    if not args.checkpoint:
        print("Error: --checkpoint is required for evaluation")
        sys.exit(1)
    
    cmd = [sys.executable, 'evaluate.py',
           '--config', args.config,
           '--checkpoint', args.checkpoint,
           '--output_dir', args.output_dir]
    
    if args.data_path:
        cmd.extend(['--data_path', args.data_path])
    
    if args.visualize:
        cmd.append('--visualize')
    
    cmd.extend(['--num_samples', str(args.num_samples)])
    
    if args.max_batches:
        cmd.extend(['--max_batches', str(args.max_batches)])
    
    if args.batch_size:
        cmd.extend(['--batch_size', str(args.batch_size)])
    
    if args.cpu:
        cmd.append('--cpu')
    
    subprocess.run(cmd)

def download_dataset():
    """Print instructions for downloading the MVSEC dataset."""
    print("=== MVSEC Dataset Download Instructions ===")
    print("Please follow the instructions in data/README.md to download and set up the MVSEC dataset.")
    print("You can also visit the official MVSEC website: https://daniilidis-group.github.io/mvsec/")
    print("\nAfter downloading, place the data in the correct directory structure and update the config file.")

def main():
    args = parse_args()
    
    # Execute the requested action
    if args.action == 'train':
        train(args)
    elif args.action == 'evaluate':
        evaluate(args)
    elif args.action == 'download':
        download_dataset()

if __name__ == '__main__':
    main() 