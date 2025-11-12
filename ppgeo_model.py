"""
PPGeo model with DepthAnythingV2 DINOv2 encoder and DPT decoder.
Using production-ready DepthAnythingV2 architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import List, Dict, Tuple

# Add DepthAnythingV2 to path
sys.path.append('/home/matthew_strong/Desktop/autonomy-wild/Depth-Anything-V2/metric_depth')

# Import DepthAnythingV2 components
from depth_anything_v2.dinov2 import DINOv2
from depth_anything_v2.dpt import DPTHead


class PPGeoDepthAnythingEncoder(nn.Module):
    """DepthAnythingV2 DINOv2 encoder for PPGeo."""
    
    def __init__(self, model_size="vitl"):
        super().__init__()
        
        self.model_size = model_size
        
        # Layer indices for different model sizes
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        # Create DINOv2 encoder using DepthAnythingV2 implementation
        self.encoder = DINOv2(model_name=model_size)
        
        # Store embedding dimension
        self.embed_dim = self.encoder.embed_dim
    
    def forward(self, x):
        """
        Extract multi-scale features using DepthAnythingV2 approach.
        Args:
            x: Input images [B, 3, H, W]
        Returns:
            List of (patch_features, cls_token) tuples for each intermediate layer
        """
        # Use DepthAnythingV2's get_intermediate_layers method
        features = self.encoder.get_intermediate_layers(
            x, 
            self.intermediate_layer_idx[self.model_size], 
            return_class_token=True
        )
        
        return features


class PPGeoPoseEncoder(nn.Module):
    """Modified DepthAnythingV2 DINOv2 encoder for 6-channel pose input."""
    
    def __init__(self, model_size="vitl"):
        super().__init__()
        
        self.model_size = model_size
        
        # Layer indices for different model sizes
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        # Create DINOv2 encoder using DepthAnythingV2 implementation
        self.encoder = DINOv2(model_name=model_size)
        
        # Store embedding dimension
        self.embed_dim = self.encoder.embed_dim
        
        # Modify patch embedding to handle 6 channels instead of 3
        # Create a new patch embedding layer
        original_patch_embed = self.encoder.patch_embed
        
        # Create new patch embedding that takes 6 channels
        self.patch_embed_6ch = nn.Conv2d(
            in_channels=6,  # 6 channels for concatenated frames
            out_channels=original_patch_embed.proj.out_channels,
            kernel_size=original_patch_embed.proj.kernel_size,
            stride=original_patch_embed.proj.stride,
            padding=original_patch_embed.proj.padding,
            bias=original_patch_embed.proj.bias is not None
        )
        
        # Initialize with average of original weights across input channels
        with torch.no_grad():
            original_weight = original_patch_embed.proj.weight  # [out_ch, 3, h, w]
            # Repeat the 3-channel weights for 6 channels and scale
            new_weight = original_weight.repeat(1, 2, 1, 1) * 0.5  # [out_ch, 6, h, w]
            self.patch_embed_6ch.weight.copy_(new_weight)
            
            if original_patch_embed.proj.bias is not None:
                self.patch_embed_6ch.bias.copy_(original_patch_embed.proj.bias)
    
    def forward(self, x):
        """
        Extract multi-scale features for 6-channel pose input.
        Args:
            x: Input concatenated frames [B, 6, H, W] 
        Returns:
            List of (patch_features, cls_token) tuples for each intermediate layer
        """
        # Use our custom 6-channel patch embedding
        x = self.patch_embed_6ch(x)  # [B, embed_dim, H/14, W/14]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim] where N = H*W
        
        # Add position embeddings (handle different sizes)
        pos_embed = self.encoder.pos_embed[:, 1:, :]  # Remove class token pos embed
        if pos_embed.shape[1] != x.shape[1]:
            # Interpolate position embeddings to match input size
            pos_embed = F.interpolate(
                pos_embed.transpose(1, 2),  # [B, embed_dim, N_orig]
                size=x.shape[1],           # N_new
                mode='linear',
                align_corners=False
            ).transpose(1, 2)              # [B, N_new, embed_dim]
        
        x = x + pos_embed
        
        # Add class token 
        cls_token = self.encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Forward through transformer blocks and collect intermediate features
        features = []
        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x)
            
            # Collect features at specified layers
            if i in self.intermediate_layer_idx[self.model_size]:
                # Split class token and patch tokens
                cls_token_out = x[:, 0:1, :]  # [B, 1, embed_dim]
                patch_tokens = x[:, 1:, :]    # [B, N, embed_dim]
                features.append((patch_tokens, cls_token_out))
        
        return features


class PPGeoDepthAnythingDPTHead(nn.Module):
    """DepthAnythingV2 DPT head for PPGeo depth estimation."""
    
    def __init__(self, embed_dim: int, scales: List[int] = [0, 1, 2, 3], max_depth: float = 100.0):
        super().__init__()
        
        self.scales = scales
        self.max_depth = max_depth
        
        # Use DepthAnythingV2's DPTHead directly
        self.dpt_head = DPTHead(
            in_channels=embed_dim,  # 1024 for ViT-L
            features=256,
            use_bn=False, 
            out_channels=[256, 512, 1024, 1024], 
            use_clstoken=False  # Don't use class token
        )
    
    def forward(self, features: List[Tuple], patch_h: int, patch_w: int) -> Dict[str, torch.Tensor]:
        """
        Generate depth predictions using DepthAnythingV2 DPT head.
        Args:
            features: List of (patch_features, cls_token) tuples from encoder
            patch_h: Height in patches
            patch_w: Width in patches
        Returns:
            Dictionary of depth predictions at multiple scales
        """
        # Use DepthAnythingV2's DPT head
        depth_sigmoid = self.dpt_head(features, patch_h, patch_w)  # [B, 1, H, W], sigmoid output
        
        # Scale depth to actual range [0, max_depth]
        depth = depth_sigmoid * self.max_depth  # [B, 1, H, W]
        
        # Create multi-scale outputs for PPGeo losses
        outputs = {}
        
        # Original scale (scale 0)
        outputs[("disp", 0)] = depth
        
        # Generate additional scales if requested
        B, C, H, W = depth.shape
        for scale in self.scales[1:]:
            scale_factor = 2 ** scale
            target_h = H // scale_factor
            target_w = W // scale_factor
            
            if target_h > 0 and target_w > 0:
                depth_scaled = F.interpolate(
                    depth, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                outputs[("disp", scale)] = depth_scaled
        
        return outputs


class PoseDecoder(nn.Module):
    """Pose decoder for camera motion estimation (PPGeo style)."""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # MLP for pose prediction from already pooled features
        self.pose_head = nn.Sequential(
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
            features: Pooled features from encoder [B, feature_dim]
        Returns:
            axis_angle, translation
        """
        pose = self.pose_head(features)  # [B, 6]
        
        axis_angle = pose[:, :3].unsqueeze(1)  # [B, 1, 3]
        translation = pose[:, 3:].unsqueeze(1)  # [B, 1, 3]
        
        return axis_angle, translation
    
    def compute_K_matrix(self, fl, offset, height, width):
        """Build camera matrix from focal length and offset like original PPGeo."""
        B = fl.shape[0]
        device = fl.device
        dtype = fl.dtype
        
        # Extract focal lengths and offsets
        fx = fl[:, 0]  # Already in softplus, so positive
        fy = fl[:, 1] 
        cx = offset[:, 0] * width   # Sigmoid * width
        cy = offset[:, 1] * height  # Sigmoid * height
        
        # Scale focal lengths appropriately (like original PPGeo)
        fx = fx * width
        fy = fy * height
        
        # Build 4x4 camera matrix like original PPGeo
        K = torch.zeros(B, 4, 4, device=device, dtype=dtype)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy 
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1
        K[:, 3, 3] = 1
        
        return K
    
    def add_K_to_inputs(self, K, inputs):
        """Add camera matrices to inputs dictionary at all scales."""
        for scale in self.scales:
            K_scale = K.clone()
            # Scale the camera matrix for different resolutions
            scale_factor = 2 ** scale
            K_scale[:, 0] /= scale_factor  # fx
            K_scale[:, 1] /= scale_factor  # fy 
            K_scale[:, 0, 2] /= scale_factor  # cx
            K_scale[:, 1, 2] /= scale_factor  # cy
            
            inv_K_scale = torch.linalg.pinv(K_scale)
            inputs[("K", scale)] = K_scale
            inputs[("inv_K", scale)] = inv_K_scale
        
        return inputs
    
    def load_pretrained_depth_weights(self, checkpoint_path: str):
        """
        Load pre-trained DepthAnythingV2 weights into encoder and decoder.
        Args:
            checkpoint_path: Path to DepthAnythingV2 checkpoint
        """
        if self._pretrained_loaded:
            print("⚠️ Pre-trained weights already loaded, skipping...")
            return
            
        print(f"📦 Loading DepthAnythingV2 weights from {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract state dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Create a DepthAnythingV2 model to extract weights from
            from depth_anything_v2.dpt import DepthAnythingV2
            temp_model = DepthAnythingV2(
                encoder='vitl',
                features=256,
                out_channels=[256, 512, 1024, 1024],
                max_depth=self.max_depth
            )
            
            # Load weights into temporary model
            temp_model.load_state_dict(state_dict, strict=True)
            print("✅ Successfully loaded weights into temporary DepthAnythingV2 model")
            
            # Transfer encoder weights
            print("🔄 Transferring encoder weights...")
            encoder_state_dict = {}
            for name, param in temp_model.pretrained.named_parameters():
                encoder_state_dict[name] = param.data
            
            # Load into our encoder with error handling
            missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
            if missing_keys:
                print(f"⚠️ Encoder missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"⚠️ Encoder unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            print("✅ Encoder weights transferred")
            
            # Transfer decoder weights
            print("🔄 Transferring decoder weights...")
            decoder_state_dict = {}
            for name, param in temp_model.depth_head.named_parameters():
                decoder_state_dict[name] = param.data
            
            # Load into our decoder with error handling
            missing_keys, unexpected_keys = self.depth_decoder.dpt_head.load_state_dict(decoder_state_dict, strict=False)
            if missing_keys:
                print(f"⚠️ Decoder missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"⚠️ Decoder unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            print("✅ Decoder weights transferred")
            
            self._pretrained_loaded = True
            print("🎉 Successfully loaded pre-trained DepthAnythingV2 weights!")
            
            # Clean up
            del temp_model
            del checkpoint
            
        except Exception as e:
            print(f"❌ Error loading pre-trained weights: {e}")
            print("🔄 Continuing with random initialization...")


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
        
        # Encoder (using DepthAnythingV2 DINOv2)
        if encoder_name == "dinov3" or encoder_name == "dinov2":
            # Use ViT-L by default for best results
            self.encoder = PPGeoDepthAnythingEncoder(model_size="vitl")
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
        
        # Depth decoder (using DepthAnythingV2 DPT head)
        self.depth_decoder = PPGeoDepthAnythingDPTHead(
            embed_dim=self.encoder.embed_dim,
            scales=scales,
            max_depth=max_depth
        )

        
        # Pose encoder: separate encoder for pose estimation (takes concatenated frames)
        # Create a modified DINOv2 encoder for 6-channel pose input
        if encoder_name == "dinov3" or encoder_name == "dinov2":
            self.pose_encoder = PPGeoPoseEncoder(model_size="vitl")
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
            
        # Pose decoder (takes concatenated frame features)
        self.pose_decoder = PoseDecoder(self.pose_encoder.embed_dim)
        
        # Camera intrinsics prediction (from pose encoder features, like original PPGeo)
        self.fl_head = nn.Sequential(
            nn.Linear(self.pose_encoder.embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),  # fx, fy
            nn.Softplus()
        )
        
        self.offset_head = nn.Sequential(
            nn.Linear(self.pose_encoder.embed_dim, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),  # cx, cy
            nn.Sigmoid()
        )
        
        # Global average pooling for intrinsics
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Flag to track if we've loaded pretrained weights
        self._pretrained_loaded = False
    
    def predict_poses_and_intrinsics(self, inputs: Dict):
        """Predict poses and intrinsics like original PPGeo."""
        # Get augmented frames for pose estimation (like original PPGeo)
        img_prev = inputs[("color_aug", -1, 0)]  # [B, 3, H, W]
        img_curr = inputs[("color_aug", 0, 0)]   # [B, 3, H, W] 
        img_next = inputs[("color_aug", 1, 0)]   # [B, 3, H, W]
        
        # Resize pose images to be divisible by 14 (like depth images)
        B, C, H, W = img_curr.shape
        patch_h, patch_w = H // 14, W // 14
        new_H, new_W = patch_h * 14, patch_w * 14

        
        if H != new_H or W != new_W:
            img_prev = F.interpolate(img_prev, size=(new_H, new_W), mode='bilinear', align_corners=False)
            img_curr = F.interpolate(img_curr, size=(new_H, new_W), mode='bilinear', align_corners=False)
            img_next = F.interpolate(img_next, size=(new_H, new_W), mode='bilinear', align_corners=False)
        
        # Concatenate frame pairs (like original PPGeo)
        pose_input1 = torch.cat([img_prev, img_curr], dim=1)  # [B, 6, H, W] prev->curr
        pose_input2 = torch.cat([img_curr, img_next], dim=1)  # [B, 6, H, W] curr->next
        

        # Forward through pose encoder
        pose_features1 = self.pose_encoder(pose_input1)  # Features for prev->curr
        pose_features2 = self.pose_encoder(pose_input2)  # Features for curr->next
        
        # Use highest resolution features and global average pool
        pose_feat1 = pose_features1[-1][0].mean(dim=1)  # [B, embed_dim]
        pose_feat2 = pose_features2[-1][0].mean(dim=1)  # [B, embed_dim]
        
        # Predict poses
        axis_angle1, translation1 = self.pose_decoder(pose_feat1)
        axis_angle2, translation2 = self.pose_decoder(pose_feat2)
        
        # Predict intrinsics from both pose features (average like original PPGeo)
        fl1 = self.fl_head(pose_feat1)  # [B, 2] - fx, fy
        offset1 = self.offset_head(pose_feat1)  # [B, 2] - cx, cy
        
        fl2 = self.fl_head(pose_feat2)  # [B, 2] - fx, fy  
        offset2 = self.offset_head(pose_feat2)  # [B, 2] - cx, cy
        
        # Compute K matrices separately and average them (like original PPGeo)
        K1 = self.compute_K_matrix(fl1, offset1, new_H, new_W)
        K2 = self.compute_K_matrix(fl2, offset2, new_H, new_W)
        K = (K1 + K2) / 2
        
        return axis_angle1, translation1, axis_angle2, translation2, K
    
    def forward(self, inputs: Dict, motion: tuple = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for PPGeo training.
        Args:
            inputs: Dictionary with PPGeo-style tuple keys like ("color_aug", frame_id, scale)
            motion: Optional tuple (axis_angle1, translation1, axis_angle2, translation2) for Stage 2
        Returns:
            Dictionary of model outputs
        """

        # Get current frame for depth prediction
        img_curr = inputs[("color_aug", 0, 0)]  # [B, 3, H, W] 
        B, C, H, W = img_curr.shape
        
        # Calculate patch dimensions for DepthAnything
        patch_h, patch_w = H // 14, W // 14  # DepthAnything uses 14x14 patches

        # lets resize the image to be divisible by 14
        new_H = patch_h * 14
        new_W = patch_w * 14
        img_curr = F.interpolate(img_curr, size=(new_H, new_W), mode='bilinear', align_corners=False)
        H, W = new_H, new_W

        
        # Extract features for current frame (for depth)
        curr_features = self.encoder(img_curr)  # List of (patch_features, cls_token) tuples
        
        # Predict depth for current frame
        depth_outputs = self.depth_decoder(curr_features, patch_h, patch_w)

        # Predict poses and intrinsics (Stage 1) or use provided motion (Stage 2)
        if motion is not None:
            # Stage 2: Use motion from MotionNet
            axis_angle1, translation1, axis_angle2, translation2 = motion
            
            # Still need to predict intrinsics for Stage 2
            _, _, _, _, K = self.predict_poses_and_intrinsics(inputs)
        else:
            # Stage 1: Predict poses and intrinsics from model (like original PPGeo)
            axis_angle1, translation1, axis_angle2, translation2, K = self.predict_poses_and_intrinsics(inputs)
        
        
        # Add K matrix to inputs (like original PPGeo)
        inputs = self.add_K_to_inputs(K, inputs)
        
        # Build output dictionary
        outputs = {}
        outputs.update(depth_outputs)
        
        outputs[("axisangle", 0, -1)] = axis_angle1
        outputs[("translation", 0, -1)] = translation1
        outputs[("axisangle", 0, 1)] = axis_angle2
        outputs[("translation", 0, 1)] = translation2
        
        return outputs, inputs
    
    def compute_K_matrix(self, fl, offsets, height, width):
        """Build camera matrix exactly like original PPGeo."""
        B = fl.shape[0]

        fl = torch.diag_embed(fl) # B * 2 * 2

        K = torch.cat([fl, offsets.view(-1, 2, 1)], 2) # B * 2 * 3
        row = torch.tensor([[0, 0, 1], [0, 0, 0]]).view(1, 2, 3).repeat(B, 1, 1).type_as(K)
        K = torch.cat([K, row], 1) # B * 4 * 3
        col = torch.tensor([0, 0, 0, 1]).view(1, 4, 1).repeat(B, 1, 1).type_as(K)
        K = torch.cat([K, col], 2) # B * 4 * 4

        return K
    
    def add_K_to_inputs(self, K, inputs):
        """Add camera matrices to inputs dictionary exactly like original PPGeo."""
        for scale in self.scales:
            K_scale = K.clone()
            K_scale[:, 0] *= self.img_size[1] // (2 ** scale)  # width
            K_scale[:, 1] *= self.img_size[0] // (2 ** scale)  # height
            inv_K_scale = torch.linalg.pinv(K_scale)
            inputs[("K", scale)] = K_scale
            inputs[("inv_K", scale)] = inv_K_scale
        
        return inputs
    
    def load_pretrained_depth_weights(self, checkpoint_path: str):
        """
        Load pre-trained DepthAnythingV2 weights into encoder and decoder.
        Args:
            checkpoint_path: Path to DepthAnythingV2 checkpoint
        """
        if self._pretrained_loaded:
            print("⚠️ Pre-trained weights already loaded, skipping...")
            return
            
        print(f"📦 Loading DepthAnythingV2 weights from {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract state dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Create a DepthAnythingV2 model to extract weights from
            from depth_anything_v2.dpt import DepthAnythingV2
            temp_model = DepthAnythingV2(
                encoder='vitl',
                features=256,
                out_channels=[256, 512, 1024, 1024],
                max_depth=self.max_depth
            )
            
            # Load weights into temporary model
            temp_model.load_state_dict(state_dict, strict=True)
            print("✅ Successfully loaded weights into temporary DepthAnythingV2 model")
            
            # Transfer encoder weights
            print("🔄 Transferring encoder weights...")
            encoder_state_dict = {}
            for name, param in temp_model.pretrained.named_parameters():
                # add encoder. to name  
                encoder_name = f'encoder.{name}'
                encoder_state_dict[encoder_name] = param.data
            
            # Load into both encoders (depth and pose)
            missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
            if missing_keys:
                print(f"⚠️ Depth encoder missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"⚠️ Depth encoder unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            
            # Also load into pose encoder
            # missing_keys, unexpected_keys = self.pose_encoder.load_state_dict(encoder_state_dict, strict=False)
            # if missing_keys:
            #     print(f"⚠️ Pose encoder missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            # if unexpected_keys:
            #     print(f"⚠️ Pose encoder unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            # print("✅ Both depth and pose encoder weights transferred")
            
            # Transfer decoder weights
            print("🔄 Transferring decoder weights...")
            decoder_state_dict = {}
            for name, param in temp_model.depth_head.named_parameters():
                decoder_state_dict[name] = param.data
            
            # Load into our decoder with error handling
            missing_keys, unexpected_keys = self.depth_decoder.dpt_head.load_state_dict(decoder_state_dict, strict=False)
            if missing_keys:
                print(f"⚠️ Decoder missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"⚠️ Decoder unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            print("✅ Decoder weights transferred")
            
            self._pretrained_loaded = True
            print("🎉 Successfully loaded pre-trained DepthAnythingV2 weights!")
            
            # Clean up
            del temp_model
            del checkpoint
            
        except Exception as e:
            print(f"❌ Error loading pre-trained weights: {e}")
            print("🔄 Continuing with random initialization...")