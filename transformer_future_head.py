"""
Transformer-based future frame prediction for Pi3.
Uses causal self-attention to model temporal dependencies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for temporal sequences."""
    
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, d_model))
        
    def forward(self, x):
        seq_len = x.size(-2)
        return x + self.encoding[:seq_len].unsqueeze(0)


class CausalSelfAttention(nn.Module):
    """Causal self-attention for autoregressive future prediction."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer("causal_mask", torch.tril(torch.ones(1000, 1000)))
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, T, head_dim]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask
        causal_mask = self.causal_mask[:T, :T]
        attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with causal attention."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


class TransformerFuturePts3d(nn.Module):
    """
    Transformer-based future point prediction.
    Uses causal self-attention to model temporal dependencies.
    """
    
    def __init__(self, patch_size, dec_embed_dim, output_dim=3, extra_tokens=3,
                 n_layers=4, n_heads=8, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.extra_tokens = extra_tokens
        self.output_dim = output_dim
        
        # Temporal transformer
        self.pos_encoding = PositionalEncoding(dec_embed_dim)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(dec_embed_dim, n_heads, dec_embed_dim * 4, dropout)
            for _ in range(n_layers)
        ])
        
        # Future frame conditioning
        self.future_query_proj = nn.Linear(dec_embed_dim, dec_embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(dec_embed_dim, output_dim * patch_size**2)
        
        # Temporal embeddings for future queries
        self.future_embeddings = nn.Parameter(torch.randn(extra_tokens, dec_embed_dim))
        
    def forward(self, decout, img_shape, batch_size, num_current_frames):
        H, W = img_shape
        tokens = decout[-1]  # [B*N, S, D]
        BN, S, D = tokens.shape
        B, N = batch_size, num_current_frames
        M = self.extra_tokens
        
        # Reshape to proper batch format [B, N, S, D]
        tokens = tokens.view(B, N, S, D)
        
        # Process each spatial location independently
        all_predictions = []
        
        for s in range(S):  # For each spatial location
            # Extract temporal sequence for this spatial location
            spatial_tokens = tokens[:, :, s, :]  # [B, N, D]
            
            # Add future query tokens
            future_queries = self.future_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]
            future_queries = self.future_query_proj(future_queries)
            
            # Concatenate current and future tokens
            temporal_seq = torch.cat([spatial_tokens, future_queries], dim=1)  # [B, N+M, D]
            
            # Apply positional encoding
            temporal_seq = self.pos_encoding(temporal_seq)
            
            # Pass through transformer layers
            for layer in self.transformer_layers:
                temporal_seq = layer(temporal_seq)
            
            # Extract predictions for all frames (current + future)
            predictions = self.output_proj(temporal_seq)  # [B, N+M, output_dim*patch_size^2]
            all_predictions.append(predictions)
        
        # Stack spatial predictions: [S, B, N+M, output_dim*patch_size^2] -> [B, N+M, S, output_dim*patch_size^2]
        all_predictions = torch.stack(all_predictions, dim=2)  # [B, N+M, S, output_dim*patch_size^2]
        
        # Reshape to final format
        BNM, S, feat_dim = all_predictions.shape
        all_predictions = all_predictions.view(B * (N + M), S, feat_dim)
        
        # Convert to spatial format
        all_predictions = all_predictions.transpose(-1, -2).view(B * (N + M), -1, H//self.patch_size, W//self.patch_size)
        all_predictions = F.pixel_shuffle(all_predictions, self.patch_size)
        all_predictions = all_predictions.permute(0, 2, 3, 1)  # [B*(N+M), H, W, output_dim]
        
        return all_predictions