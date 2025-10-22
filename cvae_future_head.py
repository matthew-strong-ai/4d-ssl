"""
Conditional Variational Autoencoder for future frame prediction in Pi3.
Models uncertainty in future predictions through latent distributions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MotionEncoder(nn.Module):
    """Encode motion patterns from consecutive frames."""
    
    def __init__(self, input_dim, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.motion_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Variational components
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, motion_sequence):
        # motion_sequence: [B, N-1, D] (motion between consecutive frames)
        B, T, D = motion_sequence.shape
        
        # Reshape for conv1d: [B, D, T]
        motion_sequence = motion_sequence.transpose(1, 2)
        
        # Encode motion patterns
        motion_features = self.motion_conv(motion_sequence).squeeze(-1)  # [B, hidden_dim]
        
        # Compute latent distribution parameters
        mu = self.mu_head(motion_features)
        logvar = self.logvar_head(motion_features)
        
        return mu, logvar


class ConditionEncoder(nn.Module):
    """Encode conditioning information (current frame features, camera poses, etc)."""
    
    def __init__(self, feature_dim, pose_dim=16, hidden_dim=256):
        super().__init__()
        # Current frame feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Camera pose encoder (flattened 4x4 matrix)
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, current_features, camera_poses=None):
        # Encode current features
        feat_encoded = self.feature_encoder(current_features)
        
        if camera_poses is not None:
            # Flatten and encode camera poses
            B, N = camera_poses.shape[:2]
            poses_flat = camera_poses.view(B, -1)
            pose_encoded = self.pose_encoder(poses_flat)
            
            # Fuse features and poses
            combined = torch.cat([feat_encoded, pose_encoded], dim=-1)
            condition = self.fusion(combined)
        else:
            condition = feat_encoded
            
        return condition


class MotionDecoder(nn.Module):
    """Decode latent motion + conditions into future point predictions."""
    
    def __init__(self, latent_dim, condition_dim, output_dim, patch_size, extra_tokens=3):
        super().__init__()
        self.extra_tokens = extra_tokens
        self.patch_size = patch_size
        self.output_dim = output_dim
        
        # Latent projection
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Condition projection  
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Temporal decoder for each future frame
        self.temporal_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, output_dim * patch_size**2)
            ) for _ in range(extra_tokens)
        ])
        
        # Motion refinement network
        self.motion_refiner = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward(self, latent_z, conditions, spatial_tokens):
        """
        Args:
            latent_z: [B, latent_dim] - sampled latent motion
            conditions: [B, condition_dim] - conditioning information
            spatial_tokens: [B, S, D] - spatial features for each location
        """
        B, S, D = spatial_tokens.shape
        
        # Project latent and conditions
        latent_feat = self.latent_proj(latent_z)  # [B, 256]
        condition_feat = self.condition_proj(conditions)  # [B, 256]
        
        # Combine latent motion with conditions
        combined = torch.cat([latent_feat, condition_feat], dim=-1)  # [B, 512]
        
        # Generate predictions for each future frame
        future_predictions = []
        
        for t in range(self.extra_tokens):
            # Time-specific decoding
            temporal_combined = combined + self.motion_refiner(combined) * (t / self.extra_tokens)
            
            # Expand to all spatial locations
            temporal_expanded = temporal_combined.unsqueeze(1).expand(-1, S, -1)  # [B, S, 512]
            
            # Decode to point predictions
            frame_pred = self.temporal_decoders[t](temporal_expanded)  # [B, S, output_dim*patch_size^2]
            future_predictions.append(frame_pred)
        
        # Stack: [extra_tokens, B, S, output_dim*patch_size^2] -> [B, extra_tokens, S, output_dim*patch_size^2]
        future_predictions = torch.stack(future_predictions, dim=1)
        
        return future_predictions


class CVAEFuturePts3d(nn.Module):
    """
    Conditional Variational Autoencoder for future point prediction.
    Models uncertainty and multiple plausible futures.
    """
    
    def __init__(self, patch_size, dec_embed_dim, output_dim=3, extra_tokens=3,
                 latent_dim=128, condition_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.extra_tokens = extra_tokens
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        # Current frame projection (deterministic)
        self.current_proj = nn.Linear(dec_embed_dim, output_dim * patch_size**2)
        
        # C-VAE components
        self.motion_encoder = MotionEncoder(dec_embed_dim, latent_dim=latent_dim)
        self.condition_encoder = ConditionEncoder(dec_embed_dim, hidden_dim=condition_dim)
        self.motion_decoder = MotionDecoder(latent_dim, condition_dim, output_dim, patch_size, extra_tokens)
        
        # Motion flow computation
        self.motion_flow_net = nn.Sequential(
            nn.Linear(dec_embed_dim * 2, dec_embed_dim),
            nn.ReLU(),
            nn.Linear(dec_embed_dim, dec_embed_dim)
        )
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def compute_motion_flow(self, tokens, num_current_frames):
        """Compute motion flow between consecutive frames."""
        B, N, S, D = tokens.shape
        
        motion_flows = []
        for i in range(N - 1):
            current = tokens[:, i]      # [B, S, D]
            next_frame = tokens[:, i+1] # [B, S, D]
            
            # Compute motion representation
            combined = torch.cat([current, next_frame], dim=-1)  # [B, S, 2*D]
            motion = self.motion_flow_net(combined)  # [B, S, D]
            motion_flows.append(motion)
        
        # Stack and average across spatial locations for global motion
        motion_flows = torch.stack(motion_flows, dim=1)  # [B, N-1, S, D]
        global_motion = motion_flows.mean(dim=2)  # [B, N-1, D]
        
        return global_motion
        
    def forward(self, decout, img_shape, batch_size, num_current_frames, camera_poses=None):
        H, W = img_shape
        tokens = decout[-1]  # [B*N, S, D]
        BN, S, D = tokens.shape
        B, N = batch_size, num_current_frames
        M = self.extra_tokens
        
        # Reshape to batch format
        tokens = tokens.view(B, N, S, D)
        
        # Current frame predictions (deterministic)
        current_tokens = tokens.view(B * N, S, D)
        current_feat = self.current_proj(current_tokens)
        current_feat = current_feat.transpose(-1, -2).view(B * N, -1, H//self.patch_size, W//self.patch_size)
        current_feat = F.pixel_shuffle(current_feat, self.patch_size)
        current_points = current_feat.permute(0, 2, 3, 1)  # [B*N, H, W, output_dim]
        
        # Future frame predictions (probabilistic)
        if N > 1:
            # Compute motion flow between consecutive frames
            global_motion = self.compute_motion_flow(tokens, N)  # [B, N-1, D]
            
            # Encode motion into latent distribution
            mu, logvar = self.motion_encoder(global_motion)
            
            # Sample from latent distribution
            z = self.reparameterize(mu, logvar) if self.training else mu
        else:
            # No motion information available, use prior
            z = torch.randn(B, self.latent_dim, device=tokens.device)
            mu = torch.zeros(B, self.latent_dim, device=tokens.device)
            logvar = torch.zeros(B, self.latent_dim, device=tokens.device)
        
        # Encode conditioning information
        last_frame_features = tokens[:, -1].mean(dim=1)  # [B, D] - global context from last frame
        conditions = self.condition_encoder(last_frame_features, camera_poses)
        
        # Decode future predictions
        future_predictions = []
        for s in range(S):  # For each spatial location
            spatial_context = tokens[:, :, s, :].mean(dim=1, keepdim=True)  # [B, 1, D]
            future_pred = self.motion_decoder(z, conditions, spatial_context)  # [B, M, 1, output_dim*patch_size^2]
            future_predictions.append(future_pred.squeeze(2))  # [B, M, output_dim*patch_size^2]
        
        # Stack spatial predictions: [S, B, M, output_dim*patch_size^2] -> [B, M, S, output_dim*patch_size^2]
        future_predictions = torch.stack(future_predictions, dim=2)  # [B, M, S, output_dim*patch_size^2]
        future_predictions = future_predictions.view(B * M, S, -1)
        
        # Convert to spatial format
        future_predictions = future_predictions.transpose(-1, -2).view(B * M, -1, H//self.patch_size, W//self.patch_size)
        future_predictions = F.pixel_shuffle(future_predictions, self.patch_size)
        future_points = future_predictions.permute(0, 2, 3, 1)  # [B*M, H, W, output_dim]
        
        # Combine current and future predictions
        all_points = torch.cat([current_points, future_points], dim=0)
        
        # Return latent parameters for loss computation
        if self.training:
            return all_points, mu, logvar
        else:
            return all_points
            
    def sample_multiple_futures(self, decout, img_shape, batch_size, num_current_frames, 
                               camera_poses=None, num_samples=5):
        """Sample multiple plausible futures for uncertainty estimation."""
        self.eval()
        
        all_samples = []
        with torch.no_grad():
            for _ in range(num_samples):
                if self.training:
                    points, _, _ = self.forward(decout, img_shape, batch_size, num_current_frames, camera_poses)
                else:
                    points = self.forward(decout, img_shape, batch_size, num_current_frames, camera_poses)
                all_samples.append(points)
        
        return torch.stack(all_samples)  # [num_samples, B*(N+M), H, W, output_dim]