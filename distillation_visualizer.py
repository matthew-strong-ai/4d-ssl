"""
Feature visualization module for knowledge distillation.

This module provides tools to visualize and compare teacher and student features
during distillation training, helping to understand the effectiveness of the
distillation process.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
from typing import Dict, Optional, Tuple, List
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import wandb


class DistillationVisualizer:
    """
    Visualizer for teacher-student feature distillation analysis.
    """
    
    def __init__(self, save_dir: str = "./distillation_visualizations", use_wandb: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save visualizations
            use_wandb: Whether to log to Weights & Biases
        """
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up matplotlib style
        try:
            # Try different seaborn style names based on version
            available_styles = plt.style.available
            if 'seaborn-v0_8' in available_styles:
                plt.style.use('seaborn-v0_8')
            elif 'seaborn' in available_styles:
                plt.style.use('seaborn')
            else:
                # Fallback to default with some nice settings
                plt.rcParams.update({
                    'figure.facecolor': 'white',
                    'axes.facecolor': 'white',
                    'axes.grid': True,
                    'grid.alpha': 0.3
                })
        except Exception as e:
            print(f"⚠️ Failed to set matplotlib style: {e}, using defaults")
            
        try:
            sns.set_palette("husl")
        except Exception as e:
            print(f"⚠️ Failed to set seaborn palette: {e}, using defaults")
        
    def visualize_feature_comparison(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
        input_image: torch.Tensor,
        step: int,
        feature_types: Optional[List[str]] = None
    ):
        """
        Create comprehensive feature comparison visualization.
        
        Args:
            teacher_features: Dict of teacher features {feature_type: [B, N, D] or [B, H*W, D]}
            student_features: Dict of student features {feature_type: [B, N, D] or [B, H*W, D]}
            input_image: Input image tensor [B, C, H, W]
            step: Current training step
            feature_types: Which feature types to visualize (None = all)
        """
        if feature_types is None:
            feature_types = list(teacher_features.keys())
            
        # Create main figure
        n_features = len(feature_types)
        fig = plt.figure(figsize=(20, 5 * n_features))
        
        # Convert input image for display
        input_img_np = self._tensor_to_image(input_image[0])  # Take first batch item
        
        for i, feature_type in enumerate(feature_types):
            if feature_type not in teacher_features or feature_type not in student_features:
                continue
                
            teacher_feat = teacher_features[feature_type][0]  # [N, D] or [H*W, D]
            student_feat = student_features[feature_type][0]  # [N, D] or [H*W, D]
            
            # Create subplot for this feature type
            row_start = i * 5
            
            # 1. Original input image
            ax1 = plt.subplot2grid((n_features * 5, 4), (row_start, 0), rowspan=2)
            ax1.imshow(input_img_np)
            ax1.set_title(f'{feature_type.replace("_", " ").title()} - Input Image')
            ax1.axis('off')
            
            # 2. Feature similarity heatmap
            ax2 = plt.subplot2grid((n_features * 5, 4), (row_start, 1), rowspan=2)
            similarity_map = self._compute_spatial_similarity(teacher_feat, student_feat)
            im2 = ax2.imshow(similarity_map, cmap='RdYlBu_r', vmin=0, vmax=1)
            ax2.set_title(f'{feature_type.replace("_", " ").title()} - Cosine Similarity')
            plt.colorbar(im2, ax=ax2, shrink=0.6)
            
            # 3. Feature magnitude comparison
            ax3 = plt.subplot2grid((n_features * 5, 4), (row_start, 2), rowspan=2)
            self._plot_feature_magnitudes(teacher_feat, student_feat, ax3, feature_type)
            
            # 4. PCA projection
            ax4 = plt.subplot2grid((n_features * 5, 4), (row_start, 3), rowspan=2)
            self._plot_pca_projection(teacher_feat, student_feat, ax4, feature_type)
            
            # 5. Feature statistics
            ax5 = plt.subplot2grid((n_features * 5, 4), (row_start + 2, 0), colspan=4)
            self._plot_feature_statistics(teacher_feat, student_feat, ax5, feature_type)
            
        plt.tight_layout()
        
        # Save visualization
        filename = f"feature_comparison_step_{step:06d}.png"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log({
                f"distillation/feature_comparison_step_{step}": wandb.Image(filepath)
            }, step=step)
            
        plt.close()
        print(f"📊 Feature comparison saved: {filepath}")
        
    def visualize_distillation_progress(
        self,
        loss_history: Dict[str, List[float]],
        similarity_history: Dict[str, List[float]],
        step: int
    ):
        """
        Visualize distillation training progress over time.
        
        Args:
            loss_history: Dict of loss values over time
            similarity_history: Dict of similarity values over time
            step: Current training step
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss curves
        ax1 = axes[0, 0]
        for loss_name, values in loss_history.items():
            steps = range(len(values))
            ax1.plot(steps, values, label=loss_name, linewidth=2)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Distillation Loss Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot similarity curves
        ax2 = axes[0, 1]
        for sim_name, values in similarity_history.items():
            steps = range(len(values))
            ax2.plot(steps, values, label=sim_name, linewidth=2)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title('Teacher-Student Similarity Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot recent loss distribution
        ax3 = axes[1, 0]
        recent_losses = []
        labels = []
        for loss_name, values in loss_history.items():
            if len(values) >= 10:  # Take last 10 values
                recent_losses.append(values[-10:])
                labels.append(loss_name)
        if recent_losses:
            ax3.boxplot(recent_losses, labels=labels)
            ax3.set_ylabel('Loss')
            ax3.set_title('Recent Loss Distribution')
            plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Plot recent similarity distribution
        ax4 = axes[1, 1]
        recent_sims = []
        sim_labels = []
        for sim_name, values in similarity_history.items():
            if len(values) >= 10:  # Take last 10 values
                recent_sims.append(values[-10:])
                sim_labels.append(sim_name)
        if recent_sims:
            ax4.boxplot(recent_sims, labels=sim_labels)
            ax4.set_ylabel('Cosine Similarity')
            ax4.set_title('Recent Similarity Distribution')
            plt.setp(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save progress visualization
        filename = f"distillation_progress_step_{step:06d}.png"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log({
                f"distillation/progress_step_{step}": wandb.Image(filepath)
            }, step=step)
            
        plt.close()
        print(f"📈 Distillation progress saved: {filepath}")
        
    def create_attention_visualization(
        self,
        teacher_features: torch.Tensor,
        student_features: torch.Tensor,
        input_image: torch.Tensor,
        step: int,
        feature_type: str = "attention"
    ):
        """
        Create attention-like visualization of feature activations.
        
        Args:
            teacher_features: Teacher features [B, N, D] or [B, H*W, D]
            student_features: Student features [B, N, D] or [B, H*W, D]
            input_image: Input image [B, C, H, W]
            step: Current training step
            feature_type: Type of features for naming
        """
        # Take first batch item
        teacher_feat = teacher_features[0]  # [N, D] or [H*W, D]
        student_feat = student_features[0]  # [N, D] or [H*W, D]
        input_img = input_image[0]  # [C, H, W]
        
        # Get spatial dimensions from input
        _, H, W = input_img.shape
        
        # Determine patch size from features
        if teacher_feat.shape[0] == H * W:
            # Already spatial tokens
            patch_h, patch_w = H, W
        else:
            # Assume square patch grid
            n_patches = teacher_feat.shape[0]
            patch_size = int(np.sqrt(H * W / n_patches))
            patch_h, patch_w = H // patch_size, W // patch_size
        
        # Compute feature norms as attention weights
        teacher_norms = torch.norm(teacher_feat, dim=-1)  # [N] or [H*W]
        student_norms = torch.norm(student_feat, dim=-1)  # [N] or [H*W]
        
        # Reshape to spatial dimensions if needed
        if teacher_norms.shape[0] != H * W:
            teacher_norms = teacher_norms.view(patch_h, patch_w)
            student_norms = student_norms.view(patch_h, patch_w)
            # Interpolate to full image size
            teacher_norms = torch.nn.functional.interpolate(
                teacher_norms.unsqueeze(0).unsqueeze(0), 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
            student_norms = torch.nn.functional.interpolate(
                student_norms.unsqueeze(0).unsqueeze(0), 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
        else:
            teacher_norms = teacher_norms.view(H, W)
            student_norms = student_norms.view(H, W)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        input_img_np = self._tensor_to_image(input_img.unsqueeze(0))
        axes[0].imshow(input_img_np)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Teacher attention
        teacher_norm_np = teacher_norms.cpu().numpy()
        im1 = axes[1].imshow(teacher_norm_np, cmap='hot', alpha=0.7)
        axes[1].imshow(input_img_np, alpha=0.3)
        axes[1].set_title(f'Teacher {feature_type.replace("_", " ").title()} Attention')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], shrink=0.6)
        
        # Student attention
        student_norm_np = student_norms.cpu().numpy()
        im2 = axes[2].imshow(student_norm_np, cmap='hot', alpha=0.7)
        axes[2].imshow(input_img_np, alpha=0.3)
        axes[2].set_title(f'Student {feature_type.replace("_", " ").title()} Attention')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], shrink=0.6)
        
        plt.tight_layout()
        
        # Save attention visualization
        filename = f"attention_{feature_type}_step_{step:06d}.png"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log({
                f"distillation/attention_{feature_type}_step_{step}": wandb.Image(filepath)
            }, step=step)
            
        plt.close()
        print(f"👁️ Attention visualization saved: {filepath}")
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy image for matplotlib."""
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first batch item
        
        # Denormalize if needed (assumes ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        if tensor.device.type == 'cuda':
            mean = mean.cuda()
            std = std.cuda()
        
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy and transpose
        img_np = tensor.cpu().numpy().transpose(1, 2, 0)
        return img_np
    
    def _compute_spatial_similarity(
        self, 
        teacher_feat: torch.Tensor, 
        student_feat: torch.Tensor
    ) -> np.ndarray:
        """Compute spatial similarity map between teacher and student features."""
        # Normalize features
        teacher_norm = torch.nn.functional.normalize(teacher_feat, dim=-1)
        student_norm = torch.nn.functional.normalize(student_feat, dim=-1)
        
        # Compute cosine similarity for each spatial location
        similarity = torch.sum(teacher_norm * student_norm, dim=-1)  # [N] or [H*W]
        
        # Reshape to reasonable spatial dimensions for visualization
        n_tokens = similarity.shape[0]
        if n_tokens == 196:  # 14x14 patches
            similarity_map = similarity.view(14, 14)
        elif n_tokens == 256:  # 16x16 patches
            similarity_map = similarity.view(16, 16)
        elif n_tokens == 64:  # 8x8 patches
            similarity_map = similarity.view(8, 8)
        else:
            # Try to find best square arrangement
            sqrt_n = int(np.sqrt(n_tokens))
            if sqrt_n * sqrt_n == n_tokens:
                similarity_map = similarity.view(sqrt_n, sqrt_n)
            else:
                # Fallback: just use first sqrt_n^2 elements
                sqrt_n = int(np.sqrt(min(n_tokens, 256)))
                similarity_map = similarity[:sqrt_n*sqrt_n].view(sqrt_n, sqrt_n)
        
        return similarity_map.cpu().numpy()
    
    def _plot_feature_magnitudes(
        self, 
        teacher_feat: torch.Tensor, 
        student_feat: torch.Tensor, 
        ax: plt.Axes, 
        feature_type: str
    ):
        """Plot feature magnitude comparison."""
        teacher_mags = torch.norm(teacher_feat, dim=-1).cpu().numpy()
        student_mags = torch.norm(student_feat, dim=-1).cpu().numpy()
        
        positions = np.arange(len(teacher_mags))
        
        ax.scatter(positions, teacher_mags, alpha=0.6, label='Teacher', color='red', s=20)
        ax.scatter(positions, student_mags, alpha=0.6, label='Student', color='blue', s=20)
        
        ax.set_xlabel('Spatial Position')
        ax.set_ylabel('Feature Magnitude')
        ax.set_title(f'{feature_type.replace("_", " ").title()} - Feature Magnitudes')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_pca_projection(
        self, 
        teacher_feat: torch.Tensor, 
        student_feat: torch.Tensor, 
        ax: plt.Axes, 
        feature_type: str
    ):
        """Plot PCA projection of features."""
        # Combine features for joint PCA
        combined_feat = torch.cat([teacher_feat, student_feat], dim=0)
        combined_np = combined_feat.cpu().numpy()
        
        # Apply PCA
        if combined_np.shape[1] > 2:  # Only if we have more than 2 dimensions
            pca = PCA(n_components=2)
            projected = pca.fit_transform(combined_np)
            
            # Split back
            n_teacher = teacher_feat.shape[0]
            teacher_proj = projected[:n_teacher]
            student_proj = projected[n_teacher:]
            
            ax.scatter(teacher_proj[:, 0], teacher_proj[:, 1], 
                      alpha=0.7, label='Teacher', color='red', s=30)
            ax.scatter(student_proj[:, 0], student_proj[:, 1], 
                      alpha=0.7, label='Student', color='blue', s=30)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
            ax.set_title(f'{feature_type.replace("_", " ").title()} - PCA Projection')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_feature_statistics(
        self, 
        teacher_feat: torch.Tensor, 
        student_feat: torch.Tensor, 
        ax: plt.Axes, 
        feature_type: str
    ):
        """Plot feature statistics comparison."""
        teacher_np = teacher_feat.cpu().numpy()
        student_np = student_feat.cpu().numpy()
        
        stats = {
            'Teacher Mean': [np.mean(teacher_np)],
            'Student Mean': [np.mean(student_np)],
            'Teacher Std': [np.std(teacher_np)],
            'Student Std': [np.std(student_np)],
            'Cosine Similarity': [torch.nn.functional.cosine_similarity(
                teacher_feat.mean(0, keepdim=True), 
                student_feat.mean(0, keepdim=True)
            ).item()]
        }
        
        x_pos = np.arange(len(stats))
        values = [v[0] for v in stats.values()]
        colors = ['red', 'blue', 'darkred', 'darkblue', 'green']
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7)
        ax.set_xlabel('Statistics')
        ax.set_ylabel('Value')
        ax.set_title(f'{feature_type.replace("_", " ").title()} - Feature Statistics')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(list(stats.keys()), rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)


def create_distillation_visualizer(save_dir: str = "./distillation_visualizations", use_wandb: bool = True) -> DistillationVisualizer:
    """
    Factory function to create a distillation visualizer.
    
    Args:
        save_dir: Directory to save visualizations
        use_wandb: Whether to use Weights & Biases logging
        
    Returns:
        DistillationVisualizer instance
    """
    return DistillationVisualizer(save_dir=save_dir, use_wandb=use_wandb)