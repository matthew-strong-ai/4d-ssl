"""
PPGeo MotionNet for Stage 2 training.
Visual encoder for motion estimation.
"""

import torch
import torch.nn as nn
import sys
import os
from typing import Tuple, Dict

# Add DepthAnythingV2 to path
sys.path.append('/home/matthew_strong/Desktop/autonomy-wild/Depth-Anything-V2/metric_depth')

from depth_anything_v2.dinov2 import DINOv2


class MotionNet(nn.Module):
    """
    MotionNet for PPGeo Stage 2 training.
    Uses DINOv2 encoder for visual motion estimation.
    """
    
    def __init__(self, model_size: str = "vitl"):
        super().__init__()
        
        # Use DINOv2 encoder (similar to Stage 1 but for motion)
        self.visual_encoder = DINOv2(model_name=model_size)
        
        # Motion decoder - predicts pose from visual features
        self.motion_decoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pool over patches
            nn.Flatten(),
            nn.Linear(self.visual_encoder.embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)  # 3 for rotation (axis-angle), 3 for translation
        )
        
        # Layer indices for intermediate features
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        self.model_size = model_size
    
    def extract_motion_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract motion features from an image.
        Args:
            image: Input image [B, 3, H, W]
        Returns:
            features: Motion features [B, embed_dim]
        """
        # Get intermediate features and use the last layer
        features = self.visual_encoder.get_intermediate_layers(
            image, 
            self.intermediate_layer_idx[self.model_size], 
            return_class_token=True
        )
        
        # Use the highest resolution patch features (last layer)
        patch_features = features[-1][0]  # [B, N_patches, embed_dim]
        
        # Global average pool across patches
        motion_features = patch_features.mean(dim=1)  # [B, embed_dim]
        
        return motion_features
    
    def predict_pose(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict pose from motion features.
        Args:
            features: Motion features [B, embed_dim]
        Returns:
            axisangle, translation
        """
        pose = self.motion_decoder(features.unsqueeze(-1))  # Add sequence dim for AdaptiveAvgPool1d
        
        axis_angle = pose[:, :3].unsqueeze(1)  # [B, 1, 3]
        translation = pose[:, 3:].unsqueeze(1)  # [B, 1, 3]
        
        return axis_angle, translation
    
    def forward(self, inputs: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for motion estimation.
        Args:
            inputs: Dictionary with PPGeo-style tuple keys
        Returns:
            axisangle1, translation1, axisangle2, translation2
        """
        # Get frames from PPGeo format
        img_prev = inputs[("color_aug", -1, 0)]  # [B, 3, H, W]
        img_curr = inputs[("color_aug", 0, 0)]   # [B, 3, H, W]
        
        # Extract motion features for both frames
        features_prev = self.extract_motion_features(img_prev)
        features_curr = self.extract_motion_features(img_curr)
        
        # Predict poses from individual frame features
        # This differs from the original PPGeo which used concatenated features
        # but is more robust for our DINOv2-based approach
        axisangle1, translation1 = self.predict_pose(features_prev)
        axisangle2, translation2 = self.predict_pose(features_curr)
        
        return axisangle1, translation1, axisangle2, translation2