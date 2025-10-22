"""
AutonomyPi3 model with additional detection head for bounding box detection.
Extends the base AutonomyPi3 model with DETR-style object detection capabilities.
"""

import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
import sys
import os

# Add Pi3 to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pi3"))

from Pi3.pi3.models.dinov2.layers import Mlp
from Pi3.pi3.utils.geometry import homogenize_points
from Pi3.pi3.models.layers.pos_embed import RoPE2D, PositionGetter
from Pi3.pi3.models.layers.block import BlockRope
from Pi3.pi3.models.layers.attention import FlashAttentionRope
from Pi3.pi3.models.layers.transformer_head import TransformerDecoder, FutureLinearPts3d
from Pi3.pi3.models.layers.camera_head import FutureCameraHead
from Pi3.pi3.models.dinov2.hub.backbones import dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin


class DetectionHead(nn.Module):
    """
    DETR-style detection head for bounding box detection.
    """
    def __init__(self, dim=512, num_classes=80, num_queries=100):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Query embeddings for object detection
        self.query_embed = nn.Parameter(torch.randn(num_queries, dim))
        nn.init.normal_(self.query_embed, std=0.02)
        
        # Detection decoder layers
        self.detection_decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation='relu',
                batch_first=True
            ) for _ in range(6)
        ])
        
        # Classification head
        self.class_head = nn.Linear(dim, num_classes + 1)  # +1 for background/no-object
        
        # Bounding box regression head  
        self.bbox_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 4)  # [center_x, center_y, width, height]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize detection head weights."""
        # Initialize classification head
        nn.init.normal_(self.class_head.weight, std=0.01)
        nn.init.constant_(self.class_head.bias, 0)
        
        # Initialize bbox head
        for layer in self.bbox_head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, hidden_features, patch_h, patch_w, B, N):
        """
        Forward pass for detection head.
        
        Args:
            hidden_features: [B*N, num_patches, dim] encoded features
            patch_h, patch_w: spatial dimensions of patches
            B: batch size
            N: number of frames
            
        Returns:
            dict containing:
                - pred_logits: [B, N, num_queries, num_classes+1] classification logits
                - pred_boxes: [B, N, num_queries, 4] bounding box predictions (normalized)
        """
        device = hidden_features.device
        BN, num_patches, dim = hidden_features.shape
        
        # Prepare query embeddings for all frames
        queries = self.query_embed.unsqueeze(0).expand(BN, -1, -1)  # [B*N, num_queries, dim]
        
        # Use patch features as memory for decoder
        memory = hidden_features  # [B*N, num_patches, dim]
        
        # Apply detection decoder layers
        detection_features = queries
        for layer in self.detection_decoder:
            detection_features = layer(
                tgt=detection_features,     # [B*N, num_queries, dim]
                memory=memory,              # [B*N, num_patches, dim]
            )
        
        # Classification predictions
        pred_logits = self.class_head(detection_features)  # [B*N, num_queries, num_classes+1]
        
        # Bounding box predictions (normalized coordinates [0,1])
        pred_boxes_raw = self.bbox_head(detection_features)  # [B*N, num_queries, 4]
        pred_boxes = torch.sigmoid(pred_boxes_raw)  # Normalize to [0,1]
        
        # Reshape to separate batch and temporal dimensions
        pred_logits = pred_logits.reshape(B, N, self.num_queries, self.num_classes + 1)
        pred_boxes = pred_boxes.reshape(B, N, self.num_queries, 4)
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }


class AutonomyPi3WithDetection(nn.Module, PyTorchModelHubMixin):
    """
    AutonomyPi3 model with additional object detection capabilities.
    
    This model extends the base AutonomyPi3 with:
    - All original capabilities (depth, confidence, camera poses)
    - Additional DETR-style object detection head
    - Support for multi-frame detection
    """
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            full_N=6,
            extra_tokens=3,
            num_classes=80,  # COCO classes by default
            num_queries=100,  # Number of detection queries
        ):
        super().__init__()
        
        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        self.full_N = full_N
        self.extra_tokens = extra_tokens
        self.num_classes = num_classes
        self.num_queries = num_queries

        # ----------------------
        #  Positional Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):  # eg rope100 
            if RoPE2D is None: 
                raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features  # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
            
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = FutureLinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3, extra_tokens=self.extra_tokens)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = FutureLinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1, extra_tokens=self.extra_tokens)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = FutureCameraHead(dim=512, N=self.full_N - extra_tokens, M=extra_tokens)

        # ----------------------
        #   Detection Decoder  
        # ----------------------
        self.detection_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim,
            dec_embed_dim=512,
            dec_num_heads=8,
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.detection_head = DetectionHead(
            dim=512, 
            num_classes=num_classes, 
            num_queries=num_queries
        )

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    def decode(self, hidden, N, H, W):
        """Decode features through transformer layers."""
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def forward(self, imgs, return_detection=True):
        """
        Forward pass with optional detection output.
        
        Args:
            imgs: [B, N, C, H, W] input images
            return_detection: bool, whether to compute detection outputs
            
        Returns:
            dict containing:
                - points: [B, N+M, H, W, 3] 3D points
                - local_points: [B, N+M, H, W, 3] local 3D points
                - conf: [B, N+M, H, W, 1] confidence maps
                - camera_poses: [B, N+M, 4, 4] camera poses
                - pred_logits: [B, N+M, num_queries, num_classes+1] (if return_detection=True)
                - pred_boxes: [B, N+M, num_queries, 4] (if return_detection=True)
        """
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden, pos = self.decode(hidden, N, H, W)

        # Decode for all heads
        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        
        # Detection decoder (only if requested)
        if return_detection:
            detection_hidden = self.detection_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points - now returns [B*(N+M), H, W, output_dim]
            point_hidden = point_hidden.float()
            local_points_flat = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W), B, N)
            total_frames = N + self.extra_tokens  # N current + M future
            local_points_raw = local_points_flat.reshape(B, total_frames, H, W, -1)
            
            xy, z = local_points_raw.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence - same temporal structure [B*(N+M), H, W, 1]
            conf_hidden = conf_hidden.float()
            conf_flat = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W), B, N)
            conf = conf_flat.reshape(B, total_frames, H, W, -1)

            # camera poses - now returns [B*(N+M), 4, 4]
            camera_hidden = camera_hidden.float()
            camera_poses_flat = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w, B, N)
            camera_poses = camera_poses_flat.reshape(B, total_frames, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

            # Prepare output dictionary
            output = dict(
                points=points,
                local_points=local_points,
                conf=conf,
                camera_poses=camera_poses,
            )

            # Add detection outputs if requested
            if return_detection:
                detection_hidden = detection_hidden.float()
                detection_output = self.detection_head(
                    detection_hidden[:, self.patch_start_idx:], 
                    patch_h, patch_w, B, N
                )
                output.update(detection_output)

        return output

    def get_detection_only(self, imgs):
        """
        Fast inference mode that only computes detection outputs.
        Useful when you only need object detection without 3D reconstruction.
        """
        imgs = (imgs - self.image_mean) / self.image_std
        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # Encode
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        
        # Decode only what's needed for detection
        hidden, pos = self.decode(hidden, N, H, W)
        detection_hidden = self.detection_decoder(hidden, xpos=pos)
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            detection_hidden = detection_hidden.float()
            detection_output = self.detection_head(
                detection_hidden[:, self.patch_start_idx:], 
                patch_h, patch_w, B, N
            )
        
        return detection_output


# Convenience function to create model with common configurations
def create_autonomy_pi3_detection(
    num_classes=80,      # COCO classes
    num_queries=100,     # Detection queries
    decoder_size='large',
    full_N=6,
    extra_tokens=3,
    **kwargs
):
    """
    Create AutonomyPi3WithDetection model with standard configuration.
    
    Args:
        num_classes: Number of object classes (default: 80 for COCO)
        num_queries: Number of detection queries (default: 100)
        decoder_size: Size of transformer decoder ('small', 'base', 'large')
        full_N: Total number of frames (current + future)
        extra_tokens: Number of future frames to predict
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        AutonomyPi3WithDetection model
    """
    return AutonomyPi3WithDetection(
        num_classes=num_classes,
        num_queries=num_queries,
        decoder_size=decoder_size,
        full_N=full_N,
        extra_tokens=extra_tokens,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_autonomy_pi3_detection(
        num_classes=80,    # COCO classes
        num_queries=100,   # Detection queries
        decoder_size='large'
    ).to(device)
    
    # Example input: batch of 2, 3 frames, 3 channels, 224x224
    imgs = torch.randn(2, 3, 3, 224, 224).to(device)
    
    # Forward pass with all outputs
    with torch.no_grad():
        outputs = model(imgs, return_detection=True)
        
        print("Output shapes:")
        for key, value in outputs.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
        
        # Detection-only forward pass (faster)
        detection_only = model.get_detection_only(imgs)
        print("\nDetection-only output shapes:")
        for key, value in detection_only.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")