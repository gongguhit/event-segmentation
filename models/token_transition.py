import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.alpha_values = alpha_values if alpha_values is not None else [0.5]
    
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
            transition_matrix = torch.bmm(transition_matrix.contiguous(), layer_transition.contiguous())
        
        # Apply the transition to the embeddings
        transformed_embeddings = torch.bmm(transition_matrix.contiguous(), source_embeddings.contiguous())
        
        return transformed_embeddings, transition_matrix

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module for token embedding attention.
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
        
        # Simple self-attention implementation without reshaping
        # Project input to queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.embed_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.bmm(attn_weights, v)
        output = self.out_proj(output)
        
        return output, attn_weights

class SelfAttentionLayer(nn.Module):
    """
    Self-attention layer with residual connection and layer normalization.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_residual=True):
        """
        Initialize self-attention layer.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_residual: Whether to use residual connections
        """
        super(SelfAttentionLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        
    def forward(self, x):
        """
        Forward pass through the self-attention layer.
        
        Args:
            x: Input embeddings
            
        Returns:
            Updated embeddings and attention matrix
        """
        residual = x
        
        # Apply attention
        attn_out, attn_matrix = self.attention(x)
        attn_out = self.dropout(attn_out)
        
        # Apply residual connection and normalization
        if self.use_residual:
            output = self.norm(residual + attn_out)
        else:
            output = self.norm(attn_out)
            
        return output, attn_matrix

class TokenEmbedding(nn.Module):
    """
    Token embedding module for initial embeddings from event representations.
    """
    
    def __init__(self, input_channels, embed_dim):
        """
        Initialize token embedding module.
        
        Args:
            input_channels: Number of input channels in event representation
            embed_dim: Dimension of token embeddings
        """
        super(TokenEmbedding, self).__init__()
        
        # Convolutional layers for patch embedding
        self.conv1 = nn.Conv2d(input_channels, embed_dim // 4, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        """
        Forward pass through token embedding.
        
        Args:
            x: Input event representation tensor (batch_size, channels, height, width)
            
        Returns:
            Token embeddings of shape (batch_size, num_tokens, embed_dim)
        """
        # Apply convolutional layers
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        
        # Reshape to (batch_size, num_tokens, embed_dim) without using reshape
        batch_size, channels, height, width = x.shape
        x = x.flatten(2)  # (batch_size, channels, height*width)
        x = x.transpose(1, 2)  # (batch_size, height*width, channels)
        
        # Apply layer normalization
        x = self.norm(x)
        
        return x 