import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from copy import deepcopy
import math

# Import existing Pi3 components
from Pi3.pi3.models.dinov2.layers import Mlp
from Pi3.pi3.utils.geometry import homogenize_points
from Pi3.pi3.models.layers.pos_embed import RoPE2D, PositionGetter
from Pi3.pi3.models.layers.block import BlockRope
from Pi3.pi3.models.layers.attention import FlashAttentionRope
from Pi3.pi3.models.layers.transformer_head import TransformerDecoder, LinearPts3d, FutureLinearPts3d
from Pi3.pi3.models.layers.camera_head import CameraHead, FutureCameraHead
from Pi3.pi3.models.dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin


class MultiHeadAttention(nn.Module):
    """Multi-head attention module for DETR-style decoder"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(out)


class DecoderLayer(nn.Module):
    """DETR-style decoder layer with self-attention and cross-attention"""
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, memory_mask=None, tgt_mask=None):
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        
        # Cross-attention
        tgt2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        
        return tgt


class ObjectDetectionDecoder(nn.Module):
    """DETR-style object detection decoder"""
    def __init__(self, d_model=256, n_heads=8, n_layers=6, n_queries=100, n_classes=80):
        super().__init__()
        self.d_model = d_model
        self.n_queries = n_queries
        self.n_classes = n_classes
        
        # Object queries (learnable embeddings)
        self.query_embed = nn.Embedding(n_queries, d_model)
        
        # Decoder layers
        decoder_layer = DecoderLayer(d_model, n_heads)
        self.decoder = nn.ModuleList([
            deepcopy(decoder_layer) for _ in range(n_layers)
        ])
        
        # Output heads
        self.class_embed = nn.Linear(d_model, n_classes)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)  # MLP for bbox regression
        
        # Auxiliary heads for intermediate supervision
        self.aux_loss = True
        if self.aux_loss:
            self.class_embed = nn.ModuleList([self.class_embed])
            self.bbox_embed = nn.ModuleList([self.bbox_embed])
            for _ in range(n_layers - 1):
                self.class_embed.append(nn.Linear(d_model, n_classes))
                self.bbox_embed.append(MLP(d_model, d_model, 4, 3))

    def forward(self, src, pos_embed=None):
        """
        Args:
            src: [B, HW, d_model] - flattened image features from encoder
            pos_embed: [B, HW, d_model] - positional embeddings
        Returns:
            outputs_class: [B, n_queries, n_classes] - classification logits
            outputs_coord: [B, n_queries, 4] - box coordinates (cx, cy, w, h)
        """
        B, HW, _ = src.shape
        
        # Initialize queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, n_queries, d_model]
        
        # Add positional encoding to source features
        if pos_embed is not None:
            src = src + pos_embed
        
        # Decoder forward pass
        tgt = torch.zeros_like(query_embed)
        outputs_class = []
        outputs_coord = []
        
        for i, layer in enumerate(self.decoder):
            tgt = layer(tgt, src)
            
            # Apply output heads
            if self.aux_loss:
                outputs_class.append(self.class_embed[i](tgt))
                outputs_coord.append(self.bbox_embed[i](tgt).sigmoid())
            elif i == len(self.decoder) - 1:
                outputs_class.append(self.class_embed(tgt))
                outputs_coord.append(self.bbox_embed(tgt).sigmoid())
        
        if self.aux_loss:
            # Return all intermediate outputs for auxiliary loss
            outputs_class = torch.stack(outputs_class)  # [n_layers, B, n_queries, n_classes]
            outputs_coord = torch.stack(outputs_coord)  # [n_layers, B, n_queries, 4]
        else:
            outputs_class = outputs_class[0]
            outputs_coord = outputs_coord[0]
            
        return outputs_class, outputs_coord


class MLP(nn.Module):
    """Simple MLP"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Pi3WithObjectDetection(nn.Module, PyTorchModelHubMixin):
    """
    Pi3 model extended with Grounding DINO-style object detection capabilities.
    
    This model maintains all original Pi3 functionality (3D points, camera poses, confidence)
    while adding a parallel object detection head that can detect and localize objects
    in 2D image space.
    """
    def __init__(
        self,
        pos_type='rope100',
        decoder_size='large',
        full_N=6,
        extra_tokens=3,
        # Object detection parameters
        detection_d_model=256,
        detection_n_heads=8,
        detection_n_layers=6,
        detection_n_queries=100,
        detection_n_classes=80,  # COCO classes
        enable_detection=True,
    ):
        super().__init__()
        import pdb; pdb.set_trace()
        
        # Store detection config
        self.enable_detection = enable_detection
        self.detection_n_queries = detection_n_queries
        self.detection_n_classes = detection_n_classes
        
        # ----------------------
        #    Original Pi3 Components (AutonomyPi3 architecture)
        # ----------------------
        
        # Encoder
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        self.full_N = full_N
        self.extra_tokens = extra_tokens

        # Positional Encoding
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):
            if RoPE2D is None: 
                raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # Decoder configuration
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

        # Register tokens
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # Original Pi3 heads
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = FutureLinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3, extra_tokens=self.extra_tokens)

        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = FutureLinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1, extra_tokens=self.extra_tokens)

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
        #    Object Detection Components
        # ----------------------
        
        if self.enable_detection:
            # Feature projection for detection (Pi3 features -> detection features)
            self.detection_feature_proj = nn.Linear(2*self.dec_embed_dim, detection_d_model)
            
            # Positional embedding for detection
            self.detection_pos_embed = nn.Parameter(torch.randn(1, 1024, detection_d_model))  # Max 1024 patches
            nn.init.normal_(self.detection_pos_embed, std=0.02)
            
            # Object detection decoder
            self.object_detection_decoder = ObjectDetectionDecoder(
                d_model=detection_d_model,
                n_heads=detection_n_heads,
                n_layers=detection_n_layers,
                n_queries=detection_n_queries,
                n_classes=detection_n_classes
            )

        # Image normalization
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    def decode(self, hidden, N, H, W):
        """Original Pi3 decode method (unchanged)"""
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

    def forward(self, imgs, return_detection=None):
        """
        Forward pass with optional object detection.
        
        Args:
            imgs: [B, N, C, H, W] - input images
            return_detection: bool - whether to compute detection outputs (overrides self.enable_detection)
            
        Returns:
            dict containing:
                - Original Pi3 outputs: points, local_points, conf, camera_poses
                - Detection outputs (if enabled): detection_class_logits, detection_bbox_coords
        """
        # Determine if detection should be computed
        compute_detection = return_detection if return_detection is not None else self.enable_detection
        
        # Normalize images
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # Encode with DINOv2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        # Pi3 decode
        hidden, pos = self.decode(hidden, N, H, W)

        # ----------------------
        #    Original Pi3 Outputs
        # ----------------------
        
        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # Local points - returns [B*(N+M), H, W, output_dim]
            point_hidden = point_hidden.float()
            local_points_flat = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W), B, N)
            total_frames = N + self.extra_tokens
            local_points_raw = local_points_flat.reshape(B, total_frames, H, W, -1)
            
            xy, z = local_points_raw.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # Confidence
            conf_hidden = conf_hidden.float()
            conf_flat = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W), B, N)
            conf = conf_flat.reshape(B, total_frames, H, W, -1)

            # Camera poses
            camera_hidden = camera_hidden.float()
            camera_poses_flat = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w, B, N)
            camera_poses = camera_poses_flat.reshape(B, total_frames, 4, 4)

            # Unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        # Prepare output dictionary
        outputs = dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )

        # ----------------------
        #    Object Detection Head
        # ----------------------
        
        if compute_detection:
            # Use the same hidden features for detection
            # hidden: [B*N, hw, 2*dec_embed_dim] where hw includes register tokens
            
            # Remove register tokens for detection (use only patch tokens)
            detection_features = hidden[:, self.patch_start_idx:]  # [B*N, patch_h*patch_w, 2*dec_embed_dim]
            
            # Project to detection feature dimension
            detection_features = self.detection_feature_proj(detection_features)  # [B*N, patch_h*patch_w, detection_d_model]
            
            # Add positional embeddings
            hw_det = detection_features.shape[1]
            pos_embed = self.detection_pos_embed[:, :hw_det, :]  # [1, hw, detection_d_model]
            
            # Process each frame independently for object detection
            detection_outputs_class = []
            detection_outputs_coord = []
            
            for frame_idx in range(N):
                # Extract features for current frame
                frame_start = frame_idx * B
                frame_end = (frame_idx + 1) * B
                frame_features = detection_features[frame_start:frame_end]  # [B, hw, detection_d_model]
                
                # Run object detection decoder
                outputs_class, outputs_coord = self.object_detection_decoder(frame_features, pos_embed)
                
                detection_outputs_class.append(outputs_class)
                detection_outputs_coord.append(outputs_coord)
            
            # Stack frame outputs: [N, ...] -> [B, N, ...]
            if self.object_detection_decoder.aux_loss:
                # outputs_class: [N, n_layers, B, n_queries, n_classes] -> [n_layers, B, N, n_queries, n_classes]
                detection_outputs_class = torch.stack(detection_outputs_class, dim=2)
                detection_outputs_coord = torch.stack(detection_outputs_coord, dim=2)
            else:
                # outputs_class: [N, B, n_queries, n_classes] -> [B, N, n_queries, n_classes]
                detection_outputs_class = torch.stack(detection_outputs_class, dim=1)
                detection_outputs_coord = torch.stack(detection_outputs_coord, dim=1)
            
            outputs.update({
                'detection_class_logits': detection_outputs_class,
                'detection_bbox_coords': detection_outputs_coord,
            })

        return outputs

    def get_detection_loss(self, outputs, targets):
        """
        Compute object detection loss similar to DETR/Grounding DINO.
        
        Args:
            outputs: model outputs containing detection_class_logits and detection_bbox_coords
            targets: list of dicts, each containing 'labels' and 'boxes' for each image
            
        Returns:
            dict containing loss components
        """
        if not self.enable_detection:
            raise ValueError("Detection is not enabled in this model")
            
        from losses import ObjectDetectionLosses  # Import your detection losses
        return ObjectDetectionLosses.detr_loss(
            outputs['detection_class_logits'], 
            outputs['detection_bbox_coords'], 
            targets,
            num_classes=self.detection_n_classes
        )

    def disable_detection(self):
        """Disable detection head to save computation"""
        self.enable_detection = False
    
    def enable_detection_head(self):
        """Enable detection head"""
        self.enable_detection = True


# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    model = Pi3WithObjectDetection(
        pos_type='rope100',
        decoder_size='large',
        full_N=6,
        extra_tokens=3,
        detection_n_queries=100,
        detection_n_classes=80,
        enable_detection=True
    )
    
    # Test forward pass
    B, N, C, H, W = 2, 6, 3, 224, 224
    imgs = torch.randn(B, N, C, H, W)
    
    with torch.no_grad():
        outputs = model(imgs)
        print("Model outputs:")
        for k, v in outputs.items():
            if torch.is_tensor(v):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {type(v)}")
        
        # Test with detection disabled
        outputs_no_detection = model(imgs, return_detection=False)
        print("\nOutputs without detection:")
        for k, v in outputs_no_detection.items():
            if torch.is_tensor(v):
                print(f"  {k}: {v.shape}")