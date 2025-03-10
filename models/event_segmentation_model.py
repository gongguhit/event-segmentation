import torch
import torch.nn as nn
import torch.nn.functional as F
from .token_transition import TokenEmbedding, SelfAttentionLayer, TokenTransition

class EventSegmentationModel(nn.Module):
    """
    Event-based segmentation model with token transition layers.
    
    The model architecture consists of:
    1. Token embedding from event representation
    2. Multiple self-attention layers with token transition
    3. Decoder head for segmentation
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
        self.num_heads = config['model']['num_heads']
        self.alpha_values = config['model']['alpha_values']
        self.layer_weights = config['model']['layer_weights']
        self.dropout = config['model']['dropout']
        self.use_residual = config['model']['use_residual']
        self.regularized_layers = config['model']['regularized_layers']
        
        # Number of input channels depends on event representation
        # For voxel grid, it's equal to the number of time bins
        self.input_channels = config['data']['time_bins'] if config['data']['event_representation'] == 'voxel_grid' else 2
        
        # Number of output classes (change as needed for your segmentation task)
        self.num_classes = 2  # Binary segmentation by default
        
        # Token embedding
        self.token_embedding = TokenEmbedding(
            input_channels=self.input_channels,
            embed_dim=self.embedding_dims[0]
        )
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList()
        for i in range(len(self.embedding_dims)):
            # Create current layer
            self.attention_layers.append(
                SelfAttentionLayer(
                    embed_dim=self.embedding_dims[i],
                    num_heads=self.num_heads[i],
                    dropout=self.dropout,
                    use_residual=self.use_residual
                )
            )
            
            # Add projection layer if not the last layer
            if i < len(self.embedding_dims) - 1:
                self.attention_layers.append(
                    nn.Linear(self.embedding_dims[i], self.embedding_dims[i+1])
                )
        
        # Token transition module
        self.token_transition = TokenTransition(alpha_values=self.alpha_values)
        
        # Decoder head for segmentation
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dims[-1], self.embedding_dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dims[-1] // 2, self.num_classes)
        )
        
        # Upsampling convolution to restore spatial dimensions
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(self.embedding_dims[-1], self.embedding_dims[-1] // 2, 
                              kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(self.embedding_dims[-1] // 2, self.embedding_dims[-1] // 4, 
                              kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(self.embedding_dims[-1] // 4, self.num_classes, 
                              kernel_size=4, stride=2, padding=1)
        )
        
    def get_patch_embeddings(self, x):
        """
        Extract patch embeddings from event representation.
        
        Args:
            x: Event representation tensor
            
        Returns:
            Token embeddings
        """
        return self.token_embedding(x)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Event representation tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Segmentation logits of shape (batch_size, num_classes, height, width)
        """
        batch_size, _, height, width = x.shape
        
        # Extract patch embeddings
        x = self.get_patch_embeddings(x)  # (batch_size, num_tokens, embed_dim)
        
        # Store embeddings for token transition and regularization
        embeddings = [x]
        attention_matrices = []
        
        # Apply self-attention layers
        for i, layer in enumerate(self.attention_layers):
            if isinstance(layer, SelfAttentionLayer):
                # Apply attention layer
                x, attn_matrix = layer(x)
                attention_matrices.append(attn_matrix)
                embeddings.append(x)
            else:
                # Apply projection layer
                x = layer(x)
        
        # Apply token transition from each regularized layer to the final layer
        transition_outputs = []
        for s in self.regularized_layers:
            if s < len(embeddings):
                trans_embed, _ = self.token_transition(
                    attention_matrices[s:], embeddings[s]
                )
                transition_outputs.append(trans_embed)
        
        # Use the final layer embedding for decoding
        x = embeddings[-1]
        
        # Apply decoder
        tokens_decoded = self.decoder(x)  # (batch_size, num_tokens, num_classes)
        
        # Reshape tokens back to spatial dimensions for segmentation
        tokens_h = int((height // 8))  # Due to 3 conv layers with stride 2
        tokens_w = int((width // 8))
        
        # If the input is not perfectly divisible by the downsampling factor,
        # adjust the output shape
        if tokens_h * tokens_w != tokens_decoded.shape[1]:
            tokens_h = int(tokens_decoded.shape[1] ** 0.5)
            tokens_w = tokens_h
        
        # Reshape to spatial dimensions
        x = tokens_decoded.transpose(1, 2).reshape(batch_size, self.num_classes, tokens_h, tokens_w)
        
        # Upsample to original dimensions
        logits = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
        
        return {
            'logits': logits,
            'embeddings': embeddings,
            'transition_embeddings': transition_outputs
        }
    
    def compute_loss(self, outputs, targets, config):
        """
        Compute the training loss as described in the paper.
        
        L = ∑(s∈S) γs Ls,  Ls = ||H^(s) × (X_M^(s) - X_E^(s))||₁
        
        Args:
            outputs: Model outputs dictionary
            targets: Target segmentation maps
            config: Training configuration
            
        Returns:
            Total loss
        """
        # Main segmentation loss (cross-entropy)
        logits = outputs['logits']
        seg_loss = F.cross_entropy(logits, targets)
        
        # Token transition regularization loss
        transition_loss = 0.0
        embeddings = outputs['embeddings']
        transition_embeddings = outputs['transition_embeddings']
        
        # Calculate weighted sum of token transition losses
        for i, (s, gamma_s) in enumerate(zip(self.regularized_layers, self.layer_weights)):
            if s < len(embeddings) and i < len(transition_embeddings):
                # L1 loss between transition embeddings and teacher model embeddings
                # (in this implementation, we compare with the last layer as a proxy)
                l1_loss = F.l1_loss(transition_embeddings[i], embeddings[-1])
                transition_loss += gamma_s * l1_loss
        
        # Total loss
        total_loss = seg_loss + transition_loss
        
        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'transition_loss': transition_loss
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
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    return model 