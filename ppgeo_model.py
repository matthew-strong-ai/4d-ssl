"""
PPGeo model with DinOV3 ViT encoder and DPT-style depth decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple


class DinOV3Encoder(nn.Module):
    """DinOV3 ViT encoder for PPGeo."""
    
    def __init__(self, model_size="vitl16", pretrained=True):
        super().__init__()
        
        # Load DinOV3 model
        if model_size == "vitl16":
            self.dinov3 = torch.hub.load('dinov3', 'dinov3_vitl16', pretrained=pretrained)
            self.embed_dim = 1024
            self.patch_size = 16
        else:
            raise ValueError(f"Unsupported DinOV3 model size: {model_size}")
        
        # Feature extraction at multiple scales for decoder
        self.feature_dims = [64, 128, 256, 512, self.embed_dim]
        
        # Project ViT features to decoder feature dimensions
        self.feature_projectors = nn.ModuleList([
            nn.Conv2d(self.embed_dim, dim, 1) for dim in self.feature_dims
        ])
    
    def forward(self, x):
        """
        Extract multi-scale features from DinOV3.
        Args:
            x: Input images [B, 3, H, W]
        Returns:
            List of features at different scales
        """
        B, C, H, W = x.shape
        
        # DinOV3 forward pass
        features = self.dinov3.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]  # [B, N, D]
        
        # Reshape to spatial format
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        patch_features = patch_tokens.transpose(1, 2).reshape(
            B, self.embed_dim, num_patches_h, num_patches_w
        )  # [B, D, H/16, W/16]
        
        # Create multi-scale features by projecting and resizing
        multi_scale_features = []
        
        for i, projector in enumerate(self.feature_projectors):
            # Project to target dimension
            feat = projector(patch_features)  # [B, target_dim, H/16, W/16]
            
            # Resize to appropriate scale (mimicking ResNet feature pyramid)
            target_scale = 2 ** (i + 1)  # 2, 4, 8, 16, 32
            target_h = H // target_scale
            target_w = W // target_scale
            
            if target_h > 0 and target_w > 0:
                feat_resized = F.interpolate(
                    feat, size=(target_h, target_w), 
                    mode='bilinear', align_corners=False
                )
                multi_scale_features.append(feat_resized)
        
        return multi_scale_features


class DPTDepthHead(nn.Module):
    """DPT-style depth decoder head."""
    
    def __init__(self, feature_dims: List[int], scales: List[int] = [0, 1, 2, 3]):
        super().__init__()
        
        self.scales = scales
        self.feature_dims = feature_dims
        
        # Fusion modules for each scale
        self.fusion_modules = nn.ModuleList()
        self.depth_heads = nn.ModuleList()
        
        for i, scale in enumerate(scales):
            # Feature fusion
            if i < len(feature_dims):
                in_dim = feature_dims[-(i+1)]  # Start from highest resolution features
            else:
                in_dim = feature_dims[0]
            
            fusion = nn.Sequential(
                nn.Conv2d(in_dim, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.fusion_modules.append(fusion)
            
            # Depth prediction head
            depth_head = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 3, padding=1),
                nn.Sigmoid()
            )
            self.depth_heads.append(depth_head)
    
    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Generate depth predictions at multiple scales.
        Args:
            features: Multi-scale features from encoder
        Returns:
            Dictionary of depth predictions
        """
        outputs = {}
        
        for i, (scale, fusion, depth_head) in enumerate(zip(self.scales, self.fusion_modules, self.depth_heads)):
            # Use appropriate feature scale
            if i < len(features):
                feat = features[-(i+1)]  # Start from highest resolution
            else:
                feat = features[0]
            
            # Fuse features
            fused = fusion(feat)
            
            # Predict depth
            depth_pred = depth_head(fused)
            
            outputs[("disp", scale)] = depth_pred
        
        return outputs


class PoseDecoder(nn.Module):
    """Pose decoder for camera motion estimation."""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Global average pooling + MLP for pose prediction
        self.pose_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)  # 3 for rotation (axis-angle), 3 for translation
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict camera pose.
        Args:
            features: Features from encoder [B, C, H, W]
        Returns:
            axis_angle, translation
        """
        pose = self.pose_head(features)  # [B, 6]
        
        axis_angle = pose[:, :3].unsqueeze(1)  # [B, 1, 3]
        translation = pose[:, 3:].unsqueeze(1)  # [B, 1, 3]
        
        return axis_angle, translation


class PPGeoModel(nn.Module):
    """Complete PPGeo model with DinOV3 encoder and DPT decoder."""
    
    def __init__(
        self, 
        encoder_name: str = "dinov3",
        img_size: Tuple[int, int] = (160, 320),
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        scales: List[int] = [0, 1, 2, 3]
    ):
        super().__init__()
        
        self.img_size = img_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scales = scales
        
        # Encoder
        if encoder_name == "dinov3":
            self.encoder = DinOV3Encoder()
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
        
        # Depth decoder
        self.depth_decoder = DPTDepthHead(
            feature_dims=self.encoder.feature_dims,
            scales=scales
        )
        
        # Pose decoder (uses highest resolution features)
        self.pose_decoder = PoseDecoder(self.encoder.feature_dims[-1])
        
        # Camera intrinsics prediction
        self.intrinsics_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.encoder.feature_dims[-1], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),  # fx, fy, cx, cy
            nn.Sigmoid()
        )
    
    def predict_poses(self, img_prev: torch.Tensor, img_curr: torch.Tensor, img_next: torch.Tensor):
        """Predict poses between frame pairs."""
        B = img_curr.shape[0]
        
        # Extract features for pose estimation
        feat_prev = self.encoder(img_prev)[-1]  # Use highest res features
        feat_curr = self.encoder(img_curr)[-1]
        feat_next = self.encoder(img_next)[-1]
        
        # Pose between prev and curr
        feat_pair1 = torch.cat([feat_prev, feat_curr], dim=1)
        # Pool to reduce dimension for pose decoder
        feat_pair1_pooled = F.adaptive_avg_pool2d(feat_pair1, (feat_curr.shape[2], feat_curr.shape[3]))
        axis_angle1, translation1 = self.pose_decoder(feat_pair1_pooled)
        
        # Pose between curr and next  
        feat_pair2 = torch.cat([feat_curr, feat_next], dim=1)
        feat_pair2_pooled = F.adaptive_avg_pool2d(feat_pair2, (feat_curr.shape[2], feat_curr.shape[3]))
        axis_angle2, translation2 = self.pose_decoder(feat_pair2_pooled)
        
        return axis_angle1, translation1, axis_angle2, translation2
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for PPGeo training.
        Args:
            inputs: Dictionary with 'images' key containing frame triplet [B, 3, 3, H, W]
        Returns:
            Dictionary of model outputs
        """
        images = inputs["images"]  # [B, 3, 3, H, W] - (prev, curr, next)
        B, num_frames, C, H, W = images.shape
        
        img_prev = images[:, 0]  # [B, 3, H, W]
        img_curr = images[:, 1]  # [B, 3, H, W] 
        img_next = images[:, 2]  # [B, 3, H, W]
        
        # Extract features for current frame (for depth)
        curr_features = self.encoder(img_curr)
        
        # Predict depth for current frame
        depth_outputs = self.depth_decoder(curr_features)
        
        # Predict poses
        axis_angle1, translation1, axis_angle2, translation2 = self.predict_poses(
            img_prev, img_curr, img_next
        )
        
        # Predict intrinsics from current frame
        intrinsics_pred = self.intrinsics_head(curr_features[-1])
        
        # Build output dictionary
        outputs = {}
        outputs.update(depth_outputs)
        
        outputs[("axisangle", 0, -1)] = axis_angle1
        outputs[("translation", 0, -1)] = translation1
        outputs[("axisangle", 0, 1)] = axis_angle2
        outputs[("translation", 0, 1)] = translation2
        outputs[("intrinsics", 0)] = intrinsics_pred
        
        return outputs