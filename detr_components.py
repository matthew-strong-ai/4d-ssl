"""
DETR-style detection components for Pi3 model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEmbedding2D(nn.Module):
    """
    2D Positional embedding for DETR-style attention.
    """
    def __init__(self, hidden_dim: int = 256, temperature: float = 10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: [B, C, H, W] feature tensor
        Returns:
            pos_embed: [B, C, H, W] positional embedding
        """
        B, C, H, W = tensor.shape
        
        # Create coordinate grids
        y_embed = torch.arange(H, dtype=torch.float32, device=tensor.device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=tensor.device).unsqueeze(0).repeat(H, 1)
        
        # Normalize coordinates to [0, 1]
        y_embed = y_embed / H
        x_embed = x_embed / W
        
        dim_t = torch.arange(self.hidden_dim // 2, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.hidden_dim // 2))
        
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        pos_embed = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)  # [hidden_dim, H, W]
        pos_embed = pos_embed.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, hidden_dim, H, W]
        
        return pos_embed


class DETRDecoderLayer(nn.Module):
    """
    Single DETR decoder layer with self-attention and cross-attention.
    """
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Self-attention for object queries
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Cross-attention between queries and image features
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, queries: torch.Tensor, key_value: torch.Tensor, 
                pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            queries: [B, num_queries, hidden_dim] object queries
            key_value: [B, H*W, hidden_dim] flattened image features
            pos_embed: [B, H*W, hidden_dim] positional embeddings for image features
        Returns:
            queries: [B, num_queries, hidden_dim] updated queries
        """
        # Self-attention among object queries
        queries2, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + self.dropout1(queries2))
        
        # Cross-attention between queries and image features
        if pos_embed is not None:
            key_value_with_pos = key_value + pos_embed
        else:
            key_value_with_pos = key_value
            
        queries2, _ = self.cross_attn(queries, key_value_with_pos, key_value)
        queries = self.norm2(queries + self.dropout2(queries2))
        
        # Feed-forward network
        queries2 = self.ffn(queries)
        queries = self.norm3(queries + queries2)
        
        return queries


class DETRDecoder(nn.Module):
    """
    Multi-layer DETR decoder.
    """
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            DETRDecoderLayer(hidden_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        self.pos_embed = PositionalEmbedding2D(hidden_dim)

    def forward(self, queries: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [B, num_queries, hidden_dim] object queries
            image_features: [B, hidden_dim, H, W] image features from encoder
        Returns:
            output_queries: [B, num_queries, hidden_dim] final query representations
        """
        B, C, H, W = image_features.shape
        
        # Generate positional embeddings
        pos_embed = self.pos_embed(image_features)  # [B, hidden_dim, H, W]
        
        # Flatten spatial dimensions
        image_features_flat = image_features.flatten(2).transpose(1, 2)  # [B, H*W, hidden_dim]
        pos_embed_flat = pos_embed.flatten(2).transpose(1, 2)  # [B, H*W, hidden_dim]
        
        # Pass through decoder layers
        output = queries
        for layer in self.layers:
            output = layer(output, image_features_flat, pos_embed_flat)
            
        return output


class DETRHead(nn.Module):
    """
    DETR detection head for classification and bounding box regression.
    """
    def __init__(self, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        
        # Classification head (includes "no object" class)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        
        # Bounding box regression head
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # [x, y, w, h]
        )

    def forward(self, query_features: torch.Tensor) -> tuple:
        """
        Args:
            query_features: [B, num_queries, hidden_dim] decoded query features
        Returns:
            class_logits: [B, num_queries, num_classes + 1] classification logits
            bbox_preds: [B, num_queries, 4] bbox predictions (normalized coordinates)
        """
        class_logits = self.class_embed(query_features)
        bbox_preds = self.bbox_embed(query_features).sigmoid()  # Normalize to [0, 1]
        
        return class_logits, bbox_preds


class DETRDetectionModule(nn.Module):
    """
    Complete DETR detection module combining decoder and head.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_queries: int = 100, 
                 num_classes: int = 2, num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Input projection to match hidden dimension
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        
        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # DETR decoder
        self.decoder = DETRDecoder(hidden_dim, num_heads, num_layers, dropout)
        
        # Detection head
        self.head = DETRHead(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: [B, input_dim, H, W] features from backbone
        Returns:
            dict with:
                'class_logits': [B, num_queries, num_classes + 1]
                'bbox_preds': [B, num_queries, 4] 
        """
        B = features.shape[0]
        
        # Project input features to hidden dimension
        features = self.input_proj(features)  # [B, hidden_dim, H, W]
        
        # Get object queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, hidden_dim]
        
        # Decode queries with cross-attention
        decoded_queries = self.decoder(query_embed, features)  # [B, num_queries, hidden_dim]
        
        # Generate predictions
        class_logits, bbox_preds = self.head(decoded_queries)
        
        return {
            'class_logits': class_logits,
            'bbox_preds': bbox_preds
        }