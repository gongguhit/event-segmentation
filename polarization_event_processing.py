#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Polarization-Enhanced Event Data Processing

This module implements the complete processing pipeline for polarization-encoded
event data captured through a rotating polarizer setup with a DVXplorer-mini camera.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Try to import DV library for AEDAT4 file parsing
try:
    from dv import AedatFile
    DV_AVAILABLE = True
except ImportError:
    print("Warning: DV library not available. Install with: pip install dv")
    DV_AVAILABLE = False


class PolarizationEventProcessor:
    """Process polarization-encoded event data from a rotating polarizer setup."""
    
    def __init__(self, 
                 rotation_rpm=1000, 
                 angular_bins=36, 
                 spatial_downsample=4,
                 decay_factor=30e3,
                 sensor_size=(640, 480)):
        """
        Initialize the processor with configuration parameters.
        
        Args:
            rotation_rpm: Rotation speed of the polarizer in RPM
            angular_bins: Number of angular bins for rotation phase quantization
            spatial_downsample: Spatial downsampling factor
            decay_factor: Time surface decay factor in microseconds
            sensor_size: Original sensor resolution (width, height)
        """
        self.rotation_rpm = rotation_rpm
        self.angular_bins = angular_bins
        self.spatial_downsample = spatial_downsample
        self.decay_factor = decay_factor
        self.sensor_size = sensor_size
        self.downsampled_size = (sensor_size[0] // spatial_downsample, 
                                sensor_size[1] // spatial_downsample)
        
        # Calculate rotation period in seconds
        self.rotation_period = 60.0 / rotation_rpm
        
        print(f"Initialized PolarizationEventProcessor with:")
        print(f"  - Rotation speed: {rotation_rpm} RPM")
        print(f"  - Angular resolution: {angular_bins} bins")
        print(f"  - Spatial downsampling: {spatial_downsample}x")
        print(f"  - Sensor size: {sensor_size}")
        print(f"  - Downsampled size: {self.downsampled_size}")

    def read_aedat4_events(self, filepath, max_events=None, time_limit=None):
        """
        Read events from AEDAT4 file.
        
        Args:
            filepath: Path to the AEDAT4 file
            max_events: Optional maximum number of events to read
            time_limit: Optional time limit in seconds to read
            
        Returns:
            Structured array of events
        """
        if not DV_AVAILABLE:
            raise ImportError("DV library is required to read AEDAT4 files")
            
        print(f"Reading events from {filepath}...")
        
        with AedatFile(filepath) as f:
            # Get event packets
            packet_iterator = f['events'].numpy()
            
            # Read initial packet to get timestamp reference
            first_packet = next(packet_iterator)
            initial_time = first_packet['timestamp'][0]
            packets = [first_packet]
            
            # Read remaining packets, respecting limits
            total_events = len(first_packet)
            for packet in tqdm(packet_iterator, desc="Reading event packets"):
                if max_events is not None and total_events >= max_events:
                    break
                    
                if time_limit is not None:
                    # Check if we've exceeded the time limit
                    if (packet['timestamp'][0] - initial_time) > time_limit * 1e6:  # Convert to μs
                        break
                        
                packets.append(packet)
                total_events += len(packet)
            
            # Concatenate all packets
            events = np.hstack(packets)
            
            # Apply max_events limit if needed
            if max_events is not None and len(events) > max_events:
                events = events[:max_events]
                
        print(f"Read {len(events)} events spanning {(events['timestamp'][-1] - events['timestamp'][0])/1e6:.2f} seconds")
        return events
    
    def group_events_by_rotation(self, events):
        """
        Group events by polarizer rotation cycle.
        
        Args:
            events: Array of events
            
        Returns:
            Events with additional fields 'rotation_phase' and 'rotation_bin'
        """
        print("Assigning rotation phase to events...")
        
        # Create copies of the arrays to add the new fields
        rotation_phase = np.zeros(len(events), dtype=np.float32)
        rotation_bin = np.zeros(len(events), dtype=np.int32)
        
        # Calculate rotation phase for each event
        # Convert timestamps to seconds by dividing by 1e6 (timestamps are in μs)
        # Then take modulo of rotation period and normalize to [0,1]
        timestamps_sec = events['timestamp'] / 1e6
        rotation_phase = (timestamps_sec % self.rotation_period) / self.rotation_period
        
        # Assign to rotation bins
        rotation_bin = np.floor(rotation_phase * self.angular_bins).astype(np.int32)
        
        # Create a new structured array that includes the original fields plus our new ones
        new_dtype = np.dtype(events.dtype.descr + [('rotation_phase', np.float32), 
                                                  ('rotation_bin', np.int32)])
        result = np.zeros(len(events), dtype=new_dtype)
        
        # Copy original fields
        for name in events.dtype.names:
            result[name] = events[name]
            
        # Set new fields
        result['rotation_phase'] = rotation_phase
        result['rotation_bin'] = rotation_bin
        
        print(f"Events grouped into {self.angular_bins} rotation bins")
        return result
    
    def spatiotemporal_downsample(self, events):
        """
        Downsample events spatially while preserving angular distribution.
        
        Args:
            events: Events with rotation information
            
        Returns:
            Downsampled events, keeping earliest event per voxel
        """
        print("Performing spatiotemporal downsampling...")
        
        # Create spatial bin coordinates
        x_bin = events['x'] // self.spatial_downsample
        y_bin = events['y'] // self.spatial_downsample
        
        # Calculate maximum bin indices
        x_max = np.max(x_bin)
        y_max = np.max(y_bin)
        
        # Create unique voxel IDs
        voxel_ids = (events['rotation_bin'] * (x_max+1) * (y_max+1) + 
                     y_bin * (x_max+1) + x_bin)
        
        # Keep earliest event per voxel
        unique_voxels = {}
        print("Selecting earliest events per voxel...")
        
        # Process in chunks for better progress visibility
        chunk_size = 100000
        for i in tqdm(range(0, len(events), chunk_size)):
            chunk_end = min(i + chunk_size, len(events))
            chunk = events[i:chunk_end]
            chunk_voxel_ids = voxel_ids[i:chunk_end]
            
            for j, event in enumerate(chunk):
                voxel_id = chunk_voxel_ids[j]
                if voxel_id not in unique_voxels or event['timestamp'] < unique_voxels[voxel_id]['timestamp']:
                    unique_voxels[voxel_id] = event
        
        # Convert dictionary to array
        downsampled_events = np.array(list(unique_voxels.values()))
        
        reduction_ratio = (1.0 - len(downsampled_events) / len(events)) * 100
        print(f"Downsampled from {len(events)} to {len(downsampled_events)} events ({reduction_ratio:.1f}% reduction)")
        
        return downsampled_events
    
    def compute_polarization_time_surface(self, events, t_ref=None):
        """
        Create time surface enhanced with polarization information.
        
        Args:
            events: Events for a specific rotation bin
            t_ref: Reference timestamp (if None, use latest event timestamp)
            
        Returns:
            3-channel feature representation (time surface, pol ratio, event density)
        """
        # Initialize time surface and polarity counters
        width, height = self.downsampled_size
        time_surface = np.zeros((height, width), dtype=np.float32)
        pol_ratio_surface = np.zeros((height, width), dtype=np.float32)
        counter_pos = np.zeros((height, width), dtype=np.int32)
        counter_neg = np.zeros((height, width), dtype=np.int32)
        
        if len(events) == 0:
            # Return empty surface if no events
            feature_volume = np.stack([
                time_surface,
                pol_ratio_surface,
                np.zeros((height, width), dtype=np.float32)  # Log event density
            ], axis=0)
            return feature_volume
        
        # Reference time (latest event)
        if t_ref is None:
            t_ref = events['timestamp'].max()
        
        # Process events to build time surface
        for event in events:
            # Downsampled coordinates
            x = event['x'] // self.spatial_downsample
            y = event['y'] // self.spatial_downsample
            
            # Ensure coordinates are within bounds
            if x >= width or y >= height or x < 0 or y < 0:
                continue
                
            t = event['timestamp']
            p = event['polarity']
            
            # Update time surface with exponential decay
            time_surface[y, x] = np.exp(-(t_ref - t) / self.decay_factor)
            
            # Track polarity statistics for polarization properties
            if p > 0:
                counter_pos[y, x] += 1
            else:
                counter_neg[y, x] += 1
        
        # Compute polarity ratio as polarization feature
        total = counter_pos + counter_neg
        mask = total > 0
        pol_ratio_surface[mask] = (counter_pos[mask] - counter_neg[mask]) / total[mask].astype(np.float32)
        
        # Combine time surface with polarization information
        feature_volume = np.stack([
            time_surface,
            pol_ratio_surface,
            np.log1p(counter_pos + counter_neg)  # Log event density
        ], axis=0)
        
        return feature_volume
    
    def parallel_process_rotation_bins(self, events, num_workers=8):
        """
        Process rotation bins in parallel.
        
        Args:
            events: Events with rotation information
            num_workers: Number of parallel workers
            
        Returns:
            Dictionary mapping bin indices to PETS features
        """
        print(f"Processing {self.angular_bins} rotation bins in parallel...")
        
        # Get unique rotation bins
        unique_bins = np.unique(events['rotation_bin'])
        
        # Function to process a single bin
        def process_bin(bin_idx):
            # Filter events for this bin
            bin_events = events[events['rotation_bin'] == bin_idx]
            
            # Compute features
            features = self.compute_polarization_time_surface(bin_events)
            
            return bin_idx, features
        
        # Process bins in parallel
        features_by_bin = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_bin, bin_idx) for bin_idx in unique_bins]
            
            for future in tqdm(futures, desc="Processing bins"):
                bin_idx, features = future.result()
                features_by_bin[bin_idx] = features
        
        # Ensure all bins are represented (fill empty ones)
        for bin_idx in range(self.angular_bins):
            if bin_idx not in features_by_bin:
                features_by_bin[bin_idx] = self.compute_polarization_time_surface([])
        
        return features_by_bin
    
    def prepare_for_segmentation(self, features_by_bin):
        """
        Prepare features for segmentation model.
        
        Args:
            features_by_bin: Dictionary mapping bin indices to PETS features
            
        Returns:
            Feature tensor ready for input to segmentation model
        """
        print("Preparing features for segmentation model...")
        
        # Stack rotation bins as channels
        bins_to_use = sorted(features_by_bin.keys())
        stacked_features = np.concatenate([features_by_bin[bin] for bin in bins_to_use], axis=0)
        
        # Reshape to model input format (batch, channels, height, width)
        model_input = stacked_features.reshape(1, -1, stacked_features.shape[-2], stacked_features.shape[-1])
        
        print(f"Feature tensor shape: {model_input.shape}")
        return model_input
    
    def visualize_features(self, features_by_bin, output_dir):
        """
        Visualize feature representations.
        
        Args:
            features_by_bin: Dictionary mapping bin indices to PETS features
            output_dir: Directory to save visualizations
        """
        print(f"Generating visualizations in {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize each bin's features
        for bin_idx, features in features_by_bin.items():
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Time surface
            im0 = axes[0].imshow(features[0], cmap='hot')
            axes[0].set_title('Time Surface')
            plt.colorbar(im0, ax=axes[0])
            
            # Polarization ratio
            im1 = axes[1].imshow(features[1], cmap='coolwarm', vmin=-1, vmax=1)
            axes[1].set_title('Polarization Ratio')
            plt.colorbar(im1, ax=axes[1])
            
            # Event density
            im2 = axes[2].imshow(features[2], cmap='viridis')
            axes[2].set_title('Event Density (log scale)')
            plt.colorbar(im2, ax=axes[2])
            
            plt.suptitle(f'Rotation Bin {bin_idx} / {self.angular_bins}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'features_bin_{bin_idx:02d}.png'), dpi=150)
            plt.close(fig)
        
        # Create composite visualization
        # Sample a few bins for the composite visualization
        bins_to_sample = sorted(features_by_bin.keys())[::len(features_by_bin)//6][:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, bin_idx in enumerate(bins_to_sample):
            if i < len(axes):
                # Create RGB composite
                rgb_composite = np.zeros((features_by_bin[bin_idx].shape[1], 
                                          features_by_bin[bin_idx].shape[2], 3))
                
                # Normalize each channel for visualization
                time_surface = features_by_bin[bin_idx][0]
                pol_ratio = features_by_bin[bin_idx][1]
                event_density = features_by_bin[bin_idx][2]
                
                # Scale to [0,1]
                if np.max(time_surface) > 0:
                    time_surface = time_surface / np.max(time_surface)
                
                pol_ratio = (pol_ratio + 1) / 2  # Scale from [-1,1] to [0,1]
                
                if np.max(event_density) > 0:
                    event_density = event_density / np.max(event_density)
                
                # Assign to RGB channels
                rgb_composite[..., 0] = time_surface
                rgb_composite[..., 1] = pol_ratio
                rgb_composite[..., 2] = event_density
                
                axes[i].imshow(rgb_composite)
                axes[i].set_title(f'Bin {bin_idx}')
                axes[i].axis('off')
        
        plt.suptitle('Polarization-Enhanced Time Surface Features (RGB: Time, Polarity Ratio, Density)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'composite_visualization.png'), dpi=200)
        plt.close(fig)
        
        print(f"Saved visualizations to {output_dir}")
    
    def process_file(self, filepath, output_dir=None, visualize=True, max_events=None, time_limit=None):
        """
        Process a complete AEDAT4 file through the entire pipeline.
        
        Args:
            filepath: Path to AEDAT4 file
            output_dir: Directory to save results and visualizations
            visualize: Whether to generate visualizations
            max_events: Maximum number of events to process
            time_limit: Time limit in seconds to process
            
        Returns:
            Feature tensor ready for segmentation
        """
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(filepath), 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        # Read events
        events = self.read_aedat4_events(filepath, max_events=max_events, time_limit=time_limit)
        
        # Group by rotation
        events_with_rotation = self.group_events_by_rotation(events)
        
        # Visualize rotation phase distribution
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.hist(events_with_rotation['rotation_phase'], bins=36)
            plt.title('Event Distribution by Rotation Phase')
            plt.xlabel('Rotation Phase')
            plt.ylabel('Event Count')
            plt.savefig(os.path.join(output_dir, 'rotation_histogram.png'))
            plt.close()
        
        # Downsample
        downsampled_events = self.spatiotemporal_downsample(events_with_rotation)
        
        # Process rotation bins
        features_by_bin = self.parallel_process_rotation_bins(downsampled_events)
        
        # Prepare for segmentation
        model_input = self.prepare_for_segmentation(features_by_bin)
        
        # Generate visualizations
        if visualize:
            self.visualize_features(features_by_bin, output_dir)
        
        # Save processed data
        np.save(os.path.join(output_dir, 'feature_tensor.npy'), model_input)
        
        print(f"Processing complete. Results saved to {output_dir}")
        return model_input


def segment_polarization_data(feature_tensor, model, device):
    """
    Process features through segmentation model.
    
    Args:
        feature_tensor: Prepared feature tensor from PolarizationEventProcessor
        model: PyTorch segmentation model
        device: Computation device (CPU or CUDA)
        
    Returns:
        Segmentation prediction
    """
    # Convert to PyTorch tensor
    input_tensor = torch.from_numpy(feature_tensor).float().to(device)
    
    # Forward pass through model
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Extract segmentation prediction
    logits = outputs['logits']
    prediction = torch.argmax(logits, dim=1)
    
    return prediction.cpu().numpy()


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process polarization-encoded event data')
    parser.add_argument('--input', '-i', required=True, help='Path to AEDAT4 file')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--rpm', type=float, default=1000, help='Rotation speed in RPM')
    parser.add_argument('--bins', type=int, default=36, help='Number of angular bins')
    parser.add_argument('--downsample', type=int, default=4, help='Spatial downsampling factor')
    parser.add_argument('--decay', type=float, default=30e3, help='Time surface decay factor in microseconds')
    parser.add_argument('--max-events', type=int, help='Maximum number of events to process')
    parser.add_argument('--time-limit', type=float, help='Time limit in seconds to process')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualizations')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PolarizationEventProcessor(
        rotation_rpm=args.rpm,
        angular_bins=args.bins,
        spatial_downsample=args.downsample,
        decay_factor=args.decay
    )
    
    # Process file
    processor.process_file(
        filepath=args.input,
        output_dir=args.output,
        visualize=not args.no_vis,
        max_events=args.max_events,
        time_limit=args.time_limit
    )


if __name__ == "__main__":
    main() 