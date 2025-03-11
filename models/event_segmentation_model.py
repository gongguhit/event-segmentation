import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EventSegmentationModel(nn.Module):
    """
    Simplified event-based segmentation model with a convolutional architecture.
    """
    
    def __init__(self, config):
        """
        Initialize the event segmentation model.
        
        Args:
            config: Model configuration
        """
        super(EventSegmentationModel, self).__init__()
        
        # Extract configuration
        self.embedding_dims = config['model']['embedding_dims']
        
        # Number of input channels depends on event representation
        # For voxel grid, it's equal to the number of time bins
        self.input_channels = config['data']['time_bins'] if config['data']['event_representation'] == 'voxel_grid' else 2
        
        # Number of output classes (change as needed for your segmentation task)
        self.num_classes = 2  # Binary segmentation by default
        
        # Encoder
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(self.input_channels, self.embedding_dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.embedding_dims[0]),
            nn.ReLU(inplace=True),
            
            # Layer 2
            nn.Conv2d(self.embedding_dims[0], self.embedding_dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.embedding_dims[1]),
            nn.ReLU(inplace=True),
            
            # Layer 3
            nn.Conv2d(self.embedding_dims[1], self.embedding_dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.embedding_dims[2]),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(self.embedding_dims[2], self.embedding_dims[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.embedding_dims[3]),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(self.embedding_dims[3], self.embedding_dims[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.embedding_dims[2]),
            nn.ReLU(inplace=True),
            
            # Layer 2
            nn.ConvTranspose2d(self.embedding_dims[2], self.embedding_dims[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.embedding_dims[1]),
            nn.ReLU(inplace=True),
            
            # Layer 3
            nn.ConvTranspose2d(self.embedding_dims[1], self.embedding_dims[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.embedding_dims[0]),
            nn.ReLU(inplace=True),
            
            # Output layer
            nn.Conv2d(self.embedding_dims[0], self.num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Event representation tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Segmentation logits of shape (batch_size, num_classes, height, width)
        """
        # Get input dimensions
        _, _, height, width = x.shape
        
        # Encode
        features = self.encoder(x)
        
        # Decode
        logits = self.decoder(features)
        
        # Ensure output size matches input size
        if logits.shape[2] != height or logits.shape[3] != width:
            logits = F.interpolate(logits, size=(height, width), mode='bilinear', align_corners=False)
        
        # Store embeddings for compatibility with the original model
        embeddings = [features]
        
        # Make sure logits are contiguous
        logits = logits.contiguous()
        
        return {
            'logits': logits,
            'embeddings': embeddings,
            'transition_embeddings': []
        }

def get_model(config):
    """
    Create and initialize the event segmentation model.
    
    Args:
        config: Model configuration
    
    Returns:
        Initialized model
    """
    model = EventSegmentationModel(config)
    
    # Initialize model weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    return model 