"""
Distilled Vision Transformer for feature distillation from AutoregressivePi3.

This module implements a lightweight ViT that learns to distill features from the 
AutoregressivePi3 model's point predictions, camera poses, and scene representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from functools import partial
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import numpy as np
from copy import deepcopy

from Pi3.pi3.models.dinov2.models.vision_transformer import DinoVisionTransformer


class DistilledViT(nn.Module):
    """
    Distilled Vision Transformer that learns from AutoregressivePi3 features.
    
    Uses DinoVisionTransformer as backbone with distillation adapter heads.
    """
    
    def __init__(
        self,
        teacher_embed_dim=2048,  # AutoregressivePi3 encoder dim
        distill_tokens=['point_features', 'camera_features', 'autonomy_features'],
        point_feature_dim=1024,  # Point decoder hidden dim
        camera_feature_dim=512,  # Camera decoder hidden dim
        autonomy_feature_dim=2048,  # Will be set to teacher_embed_dim
    ):
        super().__init__()
        
        # Distillation configuration
        self.distill_tokens = distill_tokens
        self.teacher_embed_dim = teacher_embed_dim
        self.point_feature_dim = point_feature_dim
        self.camera_feature_dim = camera_feature_dim
        self.autonomy_feature_dim = autonomy_feature_dim or teacher_embed_dim
        
        # Use pretrained DinoV2 backbone directly (like Pi3 does)
        from Pi3.pi3.models.dinov2.hub.backbones import dinov2_vitl14_reg
        self.backbone = dinov2_vitl14_reg(pretrained=True)
        # Remove mask_token if it exists (like Pi3 does)
        if hasattr(self.backbone, 'mask_token'):
            del self.backbone.mask_token
        
        # Get embedding dimension from backbone
        self.embed_dim = self.backbone.embed_dim
        
        # Feature projection heads for distillation targets
        self.feature_projectors = nn.ModuleDict()
        
        if 'point_features' in distill_tokens:
            # Project to point feature space
            self.feature_projectors['point_features'] = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, point_feature_dim)
            )
            
        if 'camera_features' in distill_tokens:
            # Project to camera feature space
            self.feature_projectors['camera_features'] = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, camera_feature_dim)
            )
            
        if 'autonomy_features' in distill_tokens:
            # Project to autonomy feature space
            self.feature_projectors['autonomy_features'] = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.autonomy_feature_dim)
            )
            
        # Teacher feature alignment projectors
        self.teacher_projectors = nn.ModuleDict()
        
        if 'point_features' in distill_tokens:
            self.teacher_projectors['point_features'] = nn.Sequential(
                nn.LayerNorm(teacher_embed_dim),
                nn.Linear(teacher_embed_dim, point_feature_dim)
            )
            
        if 'camera_features' in distill_tokens:
            self.teacher_projectors['camera_features'] = nn.Sequential(
                nn.LayerNorm(teacher_embed_dim), 
                nn.Linear(teacher_embed_dim, camera_feature_dim)
            )
            
        if 'autonomy_features' in distill_tokens:
            self.teacher_projectors['autonomy_features'] = nn.Sequential(
                nn.LayerNorm(teacher_embed_dim),
                nn.Linear(teacher_embed_dim, self.autonomy_feature_dim)
            )
            
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights for adapter heads only (backbone already initialized)."""
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        # Only initialize the adapter heads
        self.feature_projectors.apply(_init_weights)
        self.teacher_projectors.apply(_init_weights)
        
    def forward(self, x, masks=None, return_features=False):
        """
        Forward pass with optional feature extraction for distillation.
        
        Args:
            x: Input images (B, C, H, W)
            masks: Optional attention masks
            return_features: If True, return intermediate features for distillation
            
        Returns:
            Dict with distillation targets and optionally intermediate features
        """
        # Use DinoVisionTransformer backbone
        backbone_out = self.backbone.forward_features(x, masks)
        
        # Extract features for distillation
        cls_token = backbone_out["x_norm_clstoken"]  # B, embed_dim
        patch_tokens = backbone_out["x_norm_patchtokens"]  # B, N, embed_dim
        
        # Apply feature projectors to get distillation targets
        distill_features = {}
        
        if 'point_features' in self.distill_tokens:
            # Use patch tokens for point feature prediction
            distill_features['point_features'] = self.feature_projectors['point_features'](patch_tokens)
            
        if 'camera_features' in self.distill_tokens:
            # Use patch tokens for camera feature prediction (spatial features)
            distill_features['camera_features'] = self.feature_projectors['camera_features'](patch_tokens)
            
        if 'autonomy_features' in self.distill_tokens:
            # Use patch tokens for autonomy feature representation (spatial features)
            distill_features['autonomy_features'] = self.feature_projectors['autonomy_features'](patch_tokens)
            
        if return_features:
            return {
                'distill_features': distill_features,
                'intermediate_features': backbone_out,
                'cls_token': cls_token,
                'patch_tokens': patch_tokens
            }
        else:
            return distill_features


class DistillationLoss(nn.Module):
    """
    Loss function for distilling features from AutoregressivePi3 to DistilledViT.
    """
    
    def __init__(
        self,
        distill_tokens=['point_features', 'camera_features', 'autonomy_features'],
        loss_weights=None,
        temperature=4.0,
        use_cosine_similarity=True
    ):
        super().__init__()
        
        self.distill_tokens = distill_tokens
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        
        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                'point_features': 1.0,
                'camera_features': 1.0, 
                'autonomy_features': 1.0
            }
        self.loss_weights = loss_weights
        
    def forward(self, student_features, teacher_features):
        """
        Compute distillation loss.
        
        Args:
            student_features: Dict with student model outputs
            teacher_features: Dict with teacher model features/predictions
            
        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        total_loss = 0.0
        
        for token_type in self.distill_tokens:
            if token_type not in student_features or token_type not in teacher_features:
                continue
                
            student_feat = student_features[token_type]
            teacher_feat = teacher_features[token_type]
            
            # Handle spatial feature alignment and compute cosine similarity loss
            if token_type in ['point_features', 'camera_features', 'autonomy_features']:
                # For all features, use spatial cosine similarity (camera now also spatial)
                # Student: [B, N_student_patches, D], Teacher: [B, N_teacher_patches, D]
                
                # Handle potential patch count mismatch by interpolation or subsampling
                if student_feat.shape[1] != teacher_feat.shape[1]:
                    # Resize student features to match teacher spatial resolution
                    if student_feat.shape[1] > teacher_feat.shape[1]:
                        # Subsample student features
                        indices = torch.linspace(0, student_feat.shape[1]-1, teacher_feat.shape[1], dtype=torch.long, device=student_feat.device)
                        student_feat = student_feat[:, indices]
                    else:
                        # Subsample teacher features  
                        indices = torch.linspace(0, teacher_feat.shape[1]-1, student_feat.shape[1], dtype=torch.long, device=teacher_feat.device)
                        teacher_feat = teacher_feat[:, indices]
                
                # Compute spatial cosine similarity
                student_norm = F.normalize(student_feat, dim=-1)  # [B, N, D]
                teacher_norm = F.normalize(teacher_feat, dim=-1)   # [B, N, D]

                # Spatial cosine similarity: dot product along feature dimension
                spatial_similarity = torch.sum(student_norm * teacher_norm, dim=-1)  # [B, N]
                cosine_loss = 1.0 - spatial_similarity.mean()
                
                # Add smooth L1 loss between spatial features (low weight)
                smooth_l1_loss = F.smooth_l1_loss(student_feat, teacher_feat)
                
                # Combine losses (cosine is primary, smooth L1 is auxiliary)
                loss = cosine_loss + 0.1 * smooth_l1_loss
            else:
                # Fallback to MSE loss for unknown token types
                loss = F.mse_loss(student_feat, teacher_feat)
                
            losses[f'{token_type}_loss'] = loss
            total_loss += self.loss_weights.get(token_type, 1.0) * loss
        losses['total_distillation_loss'] = total_loss
        return losses


def create_distilled_vit(
    teacher_model_name='dinov2',
    **kwargs
):
    """
    Factory function to create a DistilledViT model.
    
    Args:
        teacher_model_name: Name of teacher encoder ('dinov2' or 'dinov3')
        **kwargs: Additional arguments for DistilledViT
        
    Returns:
        DistilledViT model instance
    """
    
    # Set teacher embedding dimension based on model
    if teacher_model_name == 'dinov2':
        teacher_embed_dim = 1024  # DinoV2 ViT-L
    elif teacher_model_name == 'dinov3': 
        teacher_embed_dim = 1024  # DinoV3 ViT-L
    else:
        raise ValueError(f"Unsupported teacher model: {teacher_model_name}")
        
    # Create model with pretrained backbone (backbone loading handled in __init__)
    model = DistilledViT(
        teacher_embed_dim=teacher_embed_dim,
    )
    
    print(f"✅ Created DistilledViT with pretrained {teacher_model_name} backbone")
    return model


def visualize_distilled_pca(student_features, 
                           input_image, 
                           step,
                           save_dir = "./distilled_vit_viz"):
    """
    Simple PCA visualization of distilled ViT features, saved to disk.
    
    Args:
        student_features: Dict of student features {feature_type: [B, N, D]}
        input_image: Input image tensor [B, C, H, W] 
        step: Current training step
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert input image for display
    def tensor_to_image(tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first batch
        # Denormalize (assumes ImageNet normalization) 
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        if tensor.device.type == 'cuda':
            mean = mean.cuda()
            std = std.cuda()
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.cpu().numpy().transpose(1, 2, 0)
    
    input_img_np = tensor_to_image(input_image)
    
    # Create PCA visualization
    n_features = len(student_features)
    fig, axes = plt.subplots(1, n_features + 1, figsize=(4 * (n_features + 1), 4))
    
    if n_features == 1:
        axes = [axes] if not isinstance(axes, list) else axes
    
    # Show input image
    axes[0].imshow(input_img_np)
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    # PCA for each feature type
    for i, (feature_type, features) in enumerate(student_features.items()):
        student_feat = features[0]  # [N, D] take first batch
        
        # Apply PCA to reduce to 3 components for RGB visualization
        if student_feat.shape[1] > 3:
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(student_feat.detach().cpu().numpy())  # [N, 3]
            
            # Normalize to [0, 1] for RGB
            pca_min = pca_features.min(axis=0)
            pca_max = pca_features.max(axis=0)
            pca_normalized = (pca_features - pca_min) / (pca_max - pca_min + 1e-8)
        else:
            pca_normalized = student_feat.detach().cpu().numpy()
            if pca_normalized.shape[1] < 3:
                # Pad with zeros if less than 3 channels
                pad_shape = (pca_normalized.shape[0], 3 - pca_normalized.shape[1])
                pca_normalized = np.concatenate([pca_normalized, np.zeros(pad_shape)], axis=1)
        
        # Try to reshape to spatial grid
        n_tokens = pca_normalized.shape[0]
        if n_tokens == 196:  # 14x14
            spatial_rgb = pca_normalized.reshape(14, 14, 3)
        elif n_tokens == 256:  # 16x16
            spatial_rgb = pca_normalized.reshape(16, 16, 3)
        else:
            # Find best square arrangement
            sqrt_n = int(np.sqrt(n_tokens))
            if sqrt_n * sqrt_n == n_tokens:
                spatial_rgb = pca_normalized.reshape(sqrt_n, sqrt_n, 3)
            else:
                sqrt_n = int(np.sqrt(min(n_tokens, 256)))
                spatial_rgb = pca_normalized[:sqrt_n*sqrt_n].reshape(sqrt_n, sqrt_n, 3)
        
        # Plot PCA RGB visualization
        axes[i + 1].imshow(spatial_rgb)
        axes[i + 1].set_title(f'{feature_type.replace("_", " ").title()}\nPCA RGB')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    filename = f"distilled_vit_pca_step_{step:06d}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"🎨 DistilledViT PCA features saved: {filepath}")