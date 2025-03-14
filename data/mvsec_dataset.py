import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import yaml

class MVSECDataset(Dataset):
    """
    Dataset loader for the MVSEC (Multi-Vehicle Stereo Event Camera) dataset.
    This dataset contains stereo event camera data for automotive applications.
    """
    
    def __init__(self, root_dir, split='train', time_bins=10, 
                 event_representation='voxel_grid', transform=None, sequences=None,
                 downsample_factor=1):
        """
        Args:
            root_dir (str): Directory with the MVSEC data.
            split (str): 'train' or 'val'
            time_bins (int): Number of time bins for event representation
            event_representation (str): Type of event representation ('voxel_grid', 'event_frame', etc.)
            transform (callable, optional): Optional transform to be applied on a sample.
            sequences (list, optional): List of sequence names to use. If None, all sequences are used.
            downsample_factor (int, optional): Factor by which to downsample images. Default is 1 (no downsampling).
        """
        self.root_dir = root_dir
        self.split = split
        self.time_bins = time_bins
        self.event_representation = event_representation
        self.transform = transform
        self.filter_sequences = sequences
        self.downsample_factor = downsample_factor
        
        # Load dataset metadata
        self.sequences = self._get_sequences()
        self.samples = self._prepare_samples()
        
    def _get_sequences(self):
        """Get all valid sequences from the dataset directory."""
        sequences = []
        for seq in os.listdir(self.root_dir):
            seq_path = os.path.join(self.root_dir, seq)
            # Accept both outdoor and indoor sequences but skip the calibration directories
            if os.path.isdir(seq_path) and 'calib' not in seq:
                # Filter sequences if specified
                if self.filter_sequences is None or seq in self.filter_sequences:
                    sequences.append(seq)
        return sequences
    
    def _prepare_samples(self):
        """Prepare a list of all samples in the dataset."""
        samples = []
        for seq in self.sequences:
            seq_path = os.path.join(self.root_dir, seq)
            
            # Get data and ground truth HDF5 files
            data_file = None
            gt_file = None
            
            for f in os.listdir(seq_path):
                if f.endswith('_data.hdf5'):
                    data_file = os.path.join(seq_path, f)
                elif f.endswith('_gt.hdf5'):
                    gt_file = os.path.join(seq_path, f)
            
            # Skip if either file is missing
            if not data_file or not gt_file:
                print(f"Warning: Missing data or ground truth file for {seq}, skipping.")
                continue
            
            # Open HDF5 files to check structure and prepare samples
            try:
                with h5py.File(data_file, 'r') as data_f, h5py.File(gt_file, 'r') as gt_f:
                    # Based on the actual structure in output, MVSEC has:
                    # - davis/left/events
                    # - davis/left/image_raw
                    # - davis/left/image_raw_ts
                    
                    if 'davis/left/events' in data_f and 'davis/left/image_raw' in data_f:
                        events_data = data_f['davis/left/events']
                        images_data = data_f['davis/left/image_raw']
                        
                        # Check image timestamps
                        if 'davis/left/image_raw_ts' in data_f:
                            image_timestamps = data_f['davis/left/image_raw_ts'][:]
                        else:
                            print(f"Warning: Image timestamps not found in {data_file}, skipping.")
                            continue
                        
                        # Create samples based on image timestamps
                        for i in range(1, len(image_timestamps)):
                            samples.append({
                                'data_file': data_file,
                                'gt_file': gt_file,
                                'img_idx': i,
                                'img_timestamp': image_timestamps[i],
                                'prev_img_timestamp': image_timestamps[i-1],
                                'sequence': seq
                            })
                    else:
                        print(f"Warning: Expected data structure not found in {data_file}, skipping.")
            except Exception as e:
                print(f"Error processing {seq}: {e}")
                continue
        
        print(f"Found {len(samples)} valid samples across {len(self.sequences)} sequences")
        
        # Split train/val
        if self.split == 'train':
            return samples[:int(len(samples) * 0.8)]
        else:
            return samples[int(len(samples) * 0.8):]
    
    def __len__(self):
        return len(self.samples)
    
    def _events_to_voxel_grid(self, events, num_bins, width, height):
        """
        Convert events to a voxel grid representation.
        
        Args:
            events: Nx4 tensor (x, y, t, p) with positions, timestamps, polarities
            num_bins: Number of time bins
            width: Image width
            height: Image height
            
        Returns:
            Voxel grid tensor of shape (num_bins, height, width)
        """
        # Normalize timestamps to [0, 1]
        if events.shape[0] == 0:
            return torch.zeros((num_bins, height, width), dtype=torch.float32)
        
        timestamps = events[:, 2]
        min_timestamp = timestamps.min()
        max_timestamp = timestamps.max()
        
        # Handle edge case when all events have same timestamp
        if max_timestamp == min_timestamp:
            voxel = torch.zeros((num_bins, height, width), dtype=torch.float32)
            # Place all events in the middle bin
            middle_bin = num_bins // 2
            xs, ys, ps = events[:, 0].long(), events[:, 1].long(), events[:, 3]
            for x, y, p in zip(xs, ys, ps):
                if 0 <= x < width and 0 <= y < height:
                    voxel[middle_bin, y, x] += p
            return voxel
        
        norm_timestamps = (timestamps - min_timestamp) / (max_timestamp - min_timestamp)
        
        # Assign events to bins
        ts_bin = (norm_timestamps * (num_bins - 1)).long()
        
        # Create voxel grid
        voxel = torch.zeros((num_bins, height, width), dtype=torch.float32)
        
        # Add events to voxel grid
        xs, ys, ps = events[:, 0].long(), events[:, 1].long(), events[:, 3]
        for ts, x, y, p in zip(ts_bin, xs, ys, ps):
            if 0 <= x < width and 0 <= y < height:
                voxel[ts, y, x] += p
                
        return voxel
    
    def _create_dummy_segmentation(self, image):
        """
        Create a dummy segmentation mask since real ground truth may not be in the format we expect.
        
        Args:
            image: Input image
            
        Returns:
            Segmentation mask
        """
        height, width = image.shape[:2]
        
        # Create a simple edge-based segmentation for demonstration
        edges = cv2.Canny(image, 100, 200)
        # Dilate the edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        # Convert to binary mask
        segmentation = (edges > 0).astype(np.int64)
                
        return torch.from_numpy(segmentation)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to get
            
        Returns:
            Dictionary containing events, image, and segmentation
        """
        sample_info = self.samples[idx]
        data_file = sample_info['data_file']
        img_idx = sample_info['img_idx']
        prev_img_idx = max(0, img_idx - 1)
        
        with h5py.File(data_file, 'r') as f:
            # Get events between previous and current image
            events_data = f['davis/left/events']
            
            # Extract events that occurred between the two image timestamps
            # In MVSEC, events are stored as a Nx4 dataset with columns: x, y, timestamp, polarity
            all_events = events_data[:]  # This reads the entire events array
            
            # Get timestamps for the current and previous images
            image_timestamps = f['davis/left/image_raw_ts'][:]
            current_ts = image_timestamps[img_idx]
            prev_ts = image_timestamps[prev_img_idx]
            
            # Filter events by timestamp
            event_indices = np.where((all_events[:, 2] >= prev_ts) & 
                                    (all_events[:, 2] <= current_ts))[0]
            
            # Extract events
            events = all_events[event_indices]
            
            # Convert polarity (0/1) to (-1/1) for better voxel grid representation
            events[:, 3] = 2 * events[:, 3] - 1
            
            # Get corresponding image
            image_data = f['davis/left/image_raw']
            image = image_data[img_idx]
            
            # Apply downsampling if needed
            if self.downsample_factor > 1:
                # Downsample image
                h, w = image.shape
                new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Downsample events by scaling coordinates
                events[:, 0] = events[:, 0] / self.downsample_factor
                events[:, 1] = events[:, 1] / self.downsample_factor
                events[:, 0] = np.clip(events[:, 0], 0, new_w - 1)
                events[:, 1] = np.clip(events[:, 1], 0, new_h - 1)
            
            # Create dummy segmentation (for now)
            segmentation = self._create_dummy_segmentation(image)
            
            # Convert events to voxel grid or other representation
            if self.event_representation == 'voxel_grid':
                # Convert events to torch tensor
                events_tensor = torch.from_numpy(events).float()
                # Create voxel grid
                height, width = image.shape
                voxel_grid = self._events_to_voxel_grid(events_tensor, self.time_bins, width, height)
            else:
                # Other event representations can be added here
                raise NotImplementedError(f"Event representation '{self.event_representation}' not implemented")
            
            # Convert to torch tensors
            image_tensor = torch.from_numpy(image).float() / 255.0
            
            # Check if segmentation is already a tensor
            if isinstance(segmentation, torch.Tensor):
                segmentation_tensor = segmentation.long()
            else:
                segmentation_tensor = torch.from_numpy(segmentation).long()
            
            # Add channel dimension for image
            image_tensor = image_tensor.unsqueeze(0)
            
            # Apply any additional transforms
            if self.transform:
                image_tensor = self.transform(image_tensor)
            
            sample = {
                'events': voxel_grid,
                'image': image_tensor,
                'segmentation': segmentation_tensor
            }
            
            return sample

def get_mvsec_dataloaders(config):
    """
    Create dataloaders for MVSEC dataset.
    
    Args:
        config: Configuration dictionary with dataset parameters
        
    Returns:
        train_loader, val_loader
    """
    root_dir = config['data']['data_path']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    time_bins = config['data']['time_bins']
    event_representation = config['data']['event_representation']
    
    # Get sequences filter if specified
    sequences = config['data'].get('sequences', None)
    
    # Get downsample factor if specified
    downsample_factor = config['data'].get('downsample_factor', 1)
    
    # Create datasets
    train_dataset = MVSECDataset(
        root_dir=root_dir,
        split='train',
        time_bins=time_bins,
        event_representation=event_representation,
        sequences=sequences,
        downsample_factor=downsample_factor
    )
    
    val_dataset = MVSECDataset(
        root_dir=root_dir,
        split='val',
        time_bins=time_bins,
        event_representation=event_representation,
        sequences=sequences,
        downsample_factor=downsample_factor
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 