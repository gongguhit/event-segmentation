#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Event Stream Processor for Segmentation

This script implements event-based segmentation with token transitions based
on the architecture described in the paper.

Key features:
- Cross-modal distillation with mixed inputs
- Correlation-aware weighted token distillation
- Token transition through self-attention layers
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import h5py
from tqdm import tqdm
import time
import math

class TokenTransition(nn.Module):
    """
    Token Transition module that implements the token transition process
    as described in the paper, formulated as:
    
    H^(s) = ∏(i=s to n) [α_i P^(i) + (1 - α_i)I]
    
    Where:
    - H^(s) is the comprehensive information transition matrix
    - P^(i) is the attention matrix from layer i
    - α_i is the scaling influence
    - I is the identity matrix
    """
    
    def __init__(self, alpha_values=None):
        """
        Initialize the token transition module.
        
        Args:
            alpha_values: List of alpha values for each layer
        """
        super(TokenTransition, self).__init__()
        self.alpha_values = alpha_values if alpha_values is not None else [0.5, 0.5, 0.5, 0.5]
    
    def forward(self, attention_matrices, source_embeddings):
        """
        Apply token transition using attention matrices.
        
        Args:
            attention_matrices: List of attention matrices from each layer [P^(i)]
            source_embeddings: Source token embeddings X^(s)
            
        Returns:
            Transformed embeddings X^(n+1)
        """
        batch_size, num_tokens, embed_dim = source_embeddings.shape
        device = source_embeddings.device
        
        # Initialize identity matrix for residual connections
        identity = torch.eye(num_tokens, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Initialize transition matrix as identity
        transition_matrix = identity.clone()
        
        # Calculate the transition matrix as per equation (2)
        for i, (attn_matrix, alpha) in enumerate(zip(attention_matrices, self.alpha_values)):
            # For each layer: [α_i P^(i) + (1 - α_i)I]
            layer_transition = alpha * attn_matrix + (1 - alpha) * identity
            # Multiply with previous transitions: H^(s) = ∏(i=s to n) [α_i P^(i) + (1 - α_i)I]
            transition_matrix = torch.bmm(transition_matrix, layer_transition)
        
        # Apply the transition to the embeddings
        transformed_embeddings = torch.bmm(transition_matrix, source_embeddings)
        
        return transformed_embeddings, transition_matrix

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module for correlation-aware weighted token distillation.
    """
    
    def __init__(self, embed_dim, num_heads):
        """
        Initialize multi-head self-attention.
        
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """
        Forward pass through self-attention.
        
        Args:
            x: Input embeddings of shape (batch_size, num_tokens, embed_dim)
            
        Returns:
            Updated embeddings and attention matrix
        """
        batch_size, num_tokens, _ = x.shape
        
        # Project input to queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, num_tokens, self.embed_dim)
        output = self.out_proj(context)
        
        # Average attention weights across heads for the transition matrix
        attn_matrix = attn_weights.mean(dim=1)
        
        return output, attn_matrix

class SelfAttentionLayer(nn.Module):
    """
    Self-attention layer with residual connection and layer normalization.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Initialize self-attention layer.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(SelfAttentionLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Forward pass through the self-attention layer.
        
        Args:
            x: Input embeddings
            
        Returns:
            Updated embeddings and attention matrix
        """
        # Self-attention
        attn_output, attn_matrix = self.attention(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward
        ff_output = self.ff(x)
        x = x + ff_output
        x = self.norm2(x)
            
        return x, attn_matrix

class TokenEmbedding(nn.Module):
    """
    Token embedding module for initial embeddings from event representations.
    """
    
    def __init__(self, input_channels, embed_dim, patch_size=16):
        """
        Initialize token embedding module.
        
        Args:
            input_channels: Number of input channels in event representation
            embed_dim: Dimension of token embeddings
            patch_size: Size of patches to extract as tokens
        """
        super(TokenEmbedding, self).__init__()
        
        self.patch_size = patch_size
        
        # Convolutional embedding
        self.embedding = nn.Conv2d(
            input_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
    def forward(self, x):
        """
        Forward pass through token embedding.
        
        Args:
            x: Input event representation tensor (batch_size, channels, height, width)
            
        Returns:
            Token embeddings of shape (batch_size, num_tokens, embed_dim)
        """
        # Get batch size and input dimensions
        batch_size, _, height, width = x.shape
        
        # Apply patch embedding
        x = self.embedding(x)  # (batch_size, embed_dim, height/patch_size, width/patch_size)
        
        # Reshape to (batch_size, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose to (batch_size, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        return x

class EventSegmentationModel(nn.Module):
    """
    Event segmentation model with token transition through self-attention layers.
    """
    
    def __init__(self, 
                 input_channels=10, 
                 embed_dim=256, 
                 num_layers=4, 
                 num_heads=8, 
                 patch_size=16,
                 img_size=(256, 256),
                 alpha_values=None,
                 num_classes=2,
                 dropout=0.1):
        """
        Initialize the event segmentation model.
        
        Args:
            input_channels: Number of input channels in event representation
            embed_dim: Dimension of token embeddings
            num_layers: Number of self-attention layers
            num_heads: Number of attention heads
            patch_size: Size of patches to extract as tokens
            img_size: Input image size (height, width)
            alpha_values: List of alpha values for token transition
            num_classes: Number of output classes for segmentation
            dropout: Dropout rate
        """
        super(EventSegmentationModel, self).__init__()
        
        # Initialize parameters
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Calculate number of patches
        self.h_patches = img_size[0] // patch_size
        self.w_patches = img_size[1] // patch_size
        self.num_patches = self.h_patches * self.w_patches
        self.num_tokens = self.num_patches + 1  # +1 for CLS token
        
        print(f"Model initialized with {self.num_patches} patches ({self.h_patches}x{self.w_patches})")
        
        # Token embedding
        self.token_embedding = TokenEmbedding(
            input_channels=input_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Self-attention layers
        self.layers = nn.ModuleList([
            SelfAttentionLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Token transition
        if alpha_values is None:
            alpha_values = [0.5] * num_layers
        self.token_transition = TokenTransition(alpha_values=alpha_values)
        
        # MLP head for segmentation
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Segmentation head to reshape tokens back to image
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(
                embed_dim,
                embed_dim // 2,
                kernel_size=patch_size // 2,
                stride=patch_size // 2
            ),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(
                embed_dim // 2,
                embed_dim // 4,
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(
                embed_dim // 4,
                num_classes,
                kernel_size=1
            )
        )
        
        # Initialize weights
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initialize model weights."""
        # Initialize embedding weights
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.token_embedding.cls_token, std=0.02)
        
        # Initialize other weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights for different layer types."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input event representation tensor (batch_size, channels, height, width)
            
        Returns:
            Segmentation logits and token embeddings
        """
        # Get batch size
        batch_size = x.shape[0]
        
        # Tokenize input
        tokens = self.token_embedding(x)
        
        # Add positional embedding
        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)
        
        # Store attention matrices for token transition
        attention_matrices = []
        layer_embeddings = []
        
        # Pass through self-attention layers
        for layer in self.layers:
            tokens, attn_matrix = layer(tokens)
            attention_matrices.append(attn_matrix)
            layer_embeddings.append(tokens)
        
        # Apply token transition
        transition_embeddings, transition_matrix = self.token_transition(
            attention_matrices, 
            tokens
        )
        
        # Extract patch tokens (excluding CLS token)
        patch_tokens = transition_embeddings[:, 1:, :]
        
        # Reshape patch tokens to 2D feature map
        # Note: Be careful with the reshaping here!
        patch_tokens = patch_tokens.transpose(1, 2).contiguous()
        patch_tokens = patch_tokens.view(batch_size, self.embed_dim, self.h_patches, self.w_patches)
        
        # Apply segmentation head
        logits = self.seg_head(patch_tokens)
        
        # Resize logits to match original image size
        if logits.shape[2:] != self.img_size:
            logits = F.interpolate(
                logits, 
                size=self.img_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        return {
            'logits': logits,
            'layer_embeddings': layer_embeddings,
            'transition_embeddings': transition_embeddings,
            'transition_matrix': transition_matrix
        }

class EventProcessor:
    """
    Process event data for segmentation with knowledge distillation and token transitions.
    """
    
    def __init__(self, 
                 input_channels=10, 
                 embed_dim=256,
                 img_size=(256, 256),
                 patch_size=16,
                 device=None):
        """
        Initialize the event processor.
        
        Args:
            input_channels: Number of input channels in event representation
            embed_dim: Embedding dimension
            img_size: Input image size
            patch_size: Patch size for tokenization
            device: Device to run the model on
        """
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = EventSegmentationModel(
            input_channels=input_channels,
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size
        ).to(self.device)
        
        # For the real application, we would load pre-trained weights here
        # self.model.load_state_dict(torch.load('path/to/pretrained/weights.pth'))
        
        # Set to evaluation mode
        self.model.eval()
        
    def create_voxel_grid(self, events, num_bins):
        """
        Create voxel grid representation from events.
        
        Args:
            events: Event data as numpy structured array with x, y, timestamp, polarity
            num_bins: Number of time bins
            
        Returns:
            Voxel grid tensor of shape (num_bins, height, width)
        """
        # Extract dimensions based on img_size
        height, width = self.img_size
        
        # Create empty voxel grid
        voxel_grid = torch.zeros((num_bins, height, width), dtype=torch.float32)
        
        # Check if events is empty
        if len(events) == 0:
            return voxel_grid
        
        # Normalize timestamps to [0, 1]
        timestamps = events['timestamp']
        min_timestamp = timestamps.min()
        max_timestamp = timestamps.max()
        
        # Handle edge case when all events have same timestamp
        if max_timestamp == min_timestamp:
            norm_timestamps = np.zeros_like(timestamps, dtype=np.float32)
        else:
            norm_timestamps = (timestamps - min_timestamp) / (max_timestamp - min_timestamp)
        
        # Assign events to bins
        ts_bin = (norm_timestamps * (num_bins - 1)).astype(np.int64)
        
        # Add events to voxel grid
        xs, ys, ps = events['x'], events['y'], events['polarity']
        
        # Scale x, y if needed to match img_size
        if np.max(xs) >= width or np.max(ys) >= height:
            scale_x = width / (np.max(xs) + 1)
            scale_y = height / (np.max(ys) + 1)
            xs = (xs * scale_x).astype(np.int64)
            ys = (ys * scale_y).astype(np.int64)
        
        # Add events to voxel grid
        for ts, x, y, p in zip(ts_bin, xs, ys, ps):
            if 0 <= x < width and 0 <= y < height:
                voxel_grid[ts, y, x] += 1 if p else -1
                
        return voxel_grid
        
    def process_events(self, events, visualize=False, output_dir=None):
        """
        Process events for segmentation.
        
        Args:
            events: Event data as numpy structured array with x, y, timestamp, polarity
            visualize: Whether to visualize the results
            output_dir: Directory to save visualizations
            
        Returns:
            Segmentation result and attention visualizations
        """
        # Create voxel grid
        voxel_grid = self.create_voxel_grid(events, self.input_channels)
        
        # Add batch dimension
        voxel_grid = voxel_grid.unsqueeze(0).to(self.device)
        
        # Process through model
        with torch.no_grad():
            outputs = self.model(voxel_grid)
        
        # Get segmentation logits
        logits = outputs['logits']
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get segmentation mask
        seg_mask = torch.argmax(probs, dim=1)
        
        # Create result dictionary
        result = {
            'segmentation': seg_mask.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'transition_matrix': outputs['transition_matrix'].cpu().numpy()
        }
        
        # Visualize if requested
        if visualize:
            self.visualize_results(
                voxel_grid.cpu().numpy(),
                result,
                output_dir=output_dir
            )
        
        return result
    
    def visualize_results(self, voxel_grid, result, output_dir=None):
        """
        Visualize segmentation results and attention maps.
        
        Args:
            voxel_grid: Input voxel grid
            result: Result dictionary from process_events
            output_dir: Directory to save visualizations
        """
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract data from result
        seg_mask = result['segmentation'][0]  # Remove batch dimension
        probs = result['probabilities'][0]  # Remove batch dimension
        transition_matrix = result['transition_matrix'][0]  # Remove batch dimension
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot input voxel grid (sum across time bins)
        voxel_sum = np.sum(voxel_grid[0], axis=0)
        voxel_norm = (voxel_sum - np.min(voxel_sum)) / (np.max(voxel_sum) - np.min(voxel_sum) + 1e-6)
        axes[0, 0].imshow(voxel_norm, cmap='gray')
        axes[0, 0].set_title('Input Event Representation')
        axes[0, 0].axis('off')
        
        # Plot segmentation mask
        axes[0, 1].imshow(seg_mask, cmap='viridis')
        axes[0, 1].set_title('Segmentation Mask')
        axes[0, 1].axis('off')
        
        # Plot segmentation probability for first class
        axes[0, 2].imshow(probs[0], cmap='jet')
        axes[0, 2].set_title('Class 0 Probability')
        axes[0, 2].axis('off')
        
        # Plot segmentation probability for second class (if available)
        if probs.shape[0] > 1:
            axes[1, 0].imshow(probs[1], cmap='jet')
            axes[1, 0].set_title('Class 1 Probability')
            axes[1, 0].axis('off')
        
        # Plot transition matrix
        axes[1, 1].imshow(transition_matrix, cmap='viridis')
        axes[1, 1].set_title('Token Transition Matrix')
        axes[1, 1].axis('off')
        
        # Plot combined visualization
        combined = np.zeros((*seg_mask.shape, 3))
        combined[..., 0] = voxel_norm  # Red channel: input
        combined[..., 1] = seg_mask / (np.max(seg_mask) + 1e-6)  # Green channel: segmentation
        combined[..., 2] = probs[0]  # Blue channel: probability
        axes[1, 2].imshow(combined)
        axes[1, 2].set_title('Combined Visualization')
        axes[1, 2].axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if output directory is provided
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, 'segmentation_result.png'), dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {os.path.join(output_dir, 'segmentation_result.png')}")
        
        plt.close()

def process_events_from_h5(h5_path, output_dir="./segmentation_results", visualize=True):
    """
    Process events from an HDF5 file for segmentation.
    
    Args:
        h5_path: Path to HDF5 file with events
        output_dir: Directory to save results
        visualize: Whether to visualize results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load events from HDF5
    with h5py.File(h5_path, 'r') as f:
        events = f['events'][:]
        sensor_width = f['events'].attrs.get('sensor_width', 640)
        sensor_height = f['events'].attrs.get('sensor_height', 480)
    
    # Determine model input size based on sensor dimensions
    # Round to multiples of 16 for patch compatibility
    width = (sensor_width // 16) * 16
    height = (sensor_height // 16) * 16
    img_size = (height, width)
    
    print(f"Processing events with sensor dimensions: {sensor_height}x{sensor_width}")
    print(f"Model input size: {img_size[0]}x{img_size[1]}")
    
    # Create event processor
    processor = EventProcessor(
        input_channels=10,  # Number of time bins
        embed_dim=256,
        img_size=img_size,
        patch_size=16
    )
    
    # Process events
    result = processor.process_events(
        events,
        visualize=visualize,
        output_dir=output_dir
    )
    
    print("Event processing complete!")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Process events for segmentation with token transitions')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input HDF5 file with events')
    parser.add_argument('--output_dir', type=str, default='./segmentation_results',
                        help='Directory to save results')
    parser.add_argument('--no_vis', action='store_true',
                        help='Disable visualization')
    
    args = parser.parse_args()
    
    # Process events
    process_events_from_h5(
        args.input,
        output_dir=args.output_dir,
        visualize=not args.no_vis
    )

if __name__ == "__main__":
    main() 