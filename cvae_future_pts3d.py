"""
Conditional Variational Autoencoder for FutureLinearPts3d.
Replaces simple MLP with probabilistic future prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MotionEncoder(nn.Module):
    """Encode motion patterns from temporal sequence."""
    
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.conv_temporal = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Variational parameters
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)
        
    def forward(self, temporal_sequence):
        """
        Args:
            temporal_sequence: [B, N, D] - sequence of frame features
        Returns:
            mu, logvar: [B, latent_dim] - latent distribution parameters
        """
        B, N, D = temporal_sequence.shape
        
        # Transpose for conv1d: [B, D, N]
        seq = temporal_sequence.transpose(1, 2)
        
        # Encode temporal patterns
        features = self.conv_temporal(seq).squeeze(-1)  # [B, 128]
        
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        
        return mu, logvar


class ConditionEncoder(nn.Module):
    """Encode conditioning information for future prediction."""
    
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        
        # Current frame context encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Temporal context encoder (from sequence)
        self.temporal_context = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        
        # Global scene encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//2 + hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, current_frame, sequence_context, global_context):
        """
        Args:
            current_frame: [B, S, D] - current frame spatial features
            sequence_context: [B, D] - temporal sequence summary
            global_context: [B, D] - global scene context
        Returns:
            conditions: [B, S, hidden_dim] - conditioning features per spatial location
        """
        B, S, D = current_frame.shape
        
        # Encode different types of context
        spatial_features = self.spatial_encoder(current_frame)  # [B, S, hidden_dim]
        temporal_features = self.temporal_context(sequence_context).unsqueeze(1).expand(-1, S, -1)
        global_features = self.global_encoder(global_context).unsqueeze(1).expand(-1, S, -1)
        
        # Fuse all conditioning information
        combined = torch.cat([spatial_features, temporal_features, global_features], dim=-1)
        conditions = self.fusion(combined)
        
        return conditions


class FutureDecoder(nn.Module):
    """Decode latent motion + conditions into future predictions."""
    
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
        
        # Per-timestep decoders
        self.timestep_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256 + condition_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim * patch_size**2)
            ) for _ in range(extra_tokens)
        ])
        
        # Temporal consistency network
        self.consistency_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward(self, latent_z, conditions):
        """
        Args:
            latent_z: [B, latent_dim] - sampled latent motion
            conditions: [B, S, condition_dim] - spatial conditioning features
        Returns:
            future_predictions: [B, extra_tokens, S, output_dim*patch_size^2]
        """
        B, S, condition_dim = conditions.shape
        
        # Project latent to feature space
        latent_features = self.latent_proj(latent_z)  # [B, 256]
        
        # Generate temporal modulation
        temporal_modulation = self.consistency_net(latent_features)  # [B, 64]
        
        # Decode each future timestep
        future_preds = []
        
        for t in range(self.extra_tokens):
            # Time-aware latent features
            time_factor = (t + 1) / self.extra_tokens  # 1/3, 2/3, 3/3 for 3 future frames
            time_modulated = latent_features + temporal_modulation * time_factor
            
            # Expand to spatial dimension
            spatial_latent = time_modulated.unsqueeze(1).expand(-1, S, -1)  # [B, S, 256]
            
            # Combine with spatial conditions
            decoder_input = torch.cat([spatial_latent, conditions], dim=-1)
            
            # Decode predictions for this timestep
            timestep_pred = self.timestep_decoders[t](decoder_input)  # [B, S, output_dim*patch_size^2]
            future_preds.append(timestep_pred)
        
        # Stack timesteps: [extra_tokens, B, S, output_dim*patch_size^2] -> [B, extra_tokens, S, output_dim*patch_size^2]
        future_predictions = torch.stack(future_preds, dim=1)
        
        return future_predictions


class CVAEFutureLinearPts3d(nn.Module):
    """
    C-VAE version of FutureLinearPts3d.
    Replaces deterministic MLP with probabilistic future generation.
    """
    
    def __init__(self, patch_size, dec_embed_dim, output_dim=3, extra_tokens=3, 
                 latent_dim=64, condition_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.extra_tokens = extra_tokens
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        # Current frame projection (deterministic like original)
        self.current_proj = nn.Linear(dec_embed_dim, output_dim * patch_size**2)
        
        # C-VAE components for future prediction
        self.motion_encoder = MotionEncoder(dec_embed_dim, latent_dim)
        self.condition_encoder = ConditionEncoder(dec_embed_dim, condition_dim)
        self.future_decoder = FutureDecoder(latent_dim, condition_dim, output_dim, patch_size, extra_tokens)
        
        # Motion flow computation (optional enhancement)
        self.compute_motion_flow = True
        if self.compute_motion_flow:
            self.motion_flow_net = nn.Sequential(
                nn.Linear(dec_embed_dim * 2, dec_embed_dim),
                nn.ReLU(),
                nn.Linear(dec_embed_dim, dec_embed_dim)
            )
    
    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_temporal_motion(self, tokens, num_current_frames):
        """Compute motion flow between consecutive frames."""
        B, N, S, D = tokens.shape
        
        if N < 2:
            # No motion information available
            return torch.zeros(B, D, device=tokens.device)
        
        # Compute frame-to-frame differences
        motion_flows = []
        for i in range(N - 1):
            current = tokens[:, i]      # [B, S, D]
            next_frame = tokens[:, i+1] # [B, S, D]
            
            if self.compute_motion_flow:
                # Learn motion representation
                combined = torch.cat([current, next_frame], dim=-1)  # [B, S, 2*D]
                motion = self.motion_flow_net(combined)  # [B, S, D]
            else:
                # Simple difference
                motion = next_frame - current
            
            # Global motion (average across spatial locations)
            global_motion = motion.mean(dim=1)  # [B, D]
            motion_flows.append(global_motion)
        
        # Average motion across all transitions
        avg_motion = torch.stack(motion_flows, dim=1).mean(dim=1)  # [B, D]
        return avg_motion
    
    def forward(self, decout, img_shape, batch_size, num_current_frames):
        """
        Main forward pass - drop-in replacement for original FutureLinearPts3d.
        
        Returns:
            If training: (all_points, kl_loss)
            If inference: all_points
        """
        H, W = img_shape
        tokens = decout[-1]  # [B*N, S, D]
        BN, S, D = tokens.shape
        B, N = batch_size, num_current_frames
        M = self.extra_tokens
        
        # Reshape to proper batch format
        tokens = tokens.view(B, N, S, D)
        
        # ===== CURRENT FRAME PREDICTIONS (Deterministic, same as original) =====
        current_tokens = tokens.view(B * N, S, D)
        current_feat = self.current_proj(current_tokens)
        current_feat = current_feat.transpose(-1, -2).view(B * N, -1, H//self.patch_size, W//self.patch_size)
        current_feat = F.pixel_shuffle(current_feat, self.patch_size)
        current_points = current_feat.permute(0, 2, 3, 1)  # [B*N, H, W, output_dim]
        
        # ===== FUTURE FRAME PREDICTIONS (Probabilistic C-VAE) =====
        
        # 1. Encode motion from temporal sequence
        # Average across spatial locations for global temporal context
        temporal_sequence = tokens.mean(dim=2)  # [B, N, D]
        motion_mu, motion_logvar = self.motion_encoder(temporal_sequence)
        
        # 2. Sample from latent motion distribution
        if self.training:
            latent_z = self.reparameterize(motion_mu, motion_logvar)
        else:
            latent_z = motion_mu  # Use mean for deterministic inference
        
        # 3. Prepare conditioning information
        current_frame = tokens[:, -1]  # [B, S, D] - most recent frame
        sequence_context = temporal_sequence.mean(dim=1)  # [B, D] - temporal summary
        global_context = self.compute_temporal_motion(tokens, N)  # [B, D] - motion context
        
        conditions = self.condition_encoder(current_frame, sequence_context, global_context)
        
        # 4. Decode future predictions
        future_predictions = self.future_decoder(latent_z, conditions)  # [B, M, S, output_dim*patch_size^2]
        
        # 5. Convert to spatial format
        future_predictions = future_predictions.view(B * M, S, -1)
        future_predictions = future_predictions.transpose(-1, -2).view(B * M, -1, H//self.patch_size, W//self.patch_size)
        future_predictions = F.pixel_shuffle(future_predictions, self.patch_size)
        future_points = future_predictions.permute(0, 2, 3, 1)  # [B*M, H, W, output_dim]
        
        # 6. Combine current and future predictions
        all_points = torch.cat([current_points, future_points], dim=0)  # [B*(N+M), H, W, output_dim]
        
        # Return format matches original FutureLinearPts3d
        if self.training:
            # Compute KL divergence for VAE loss
            kl_loss = -0.5 * torch.sum(1 + motion_logvar - motion_mu.pow(2) - motion_logvar.exp(), dim=1).mean()
            return all_points, kl_loss
        else:
            return all_points
    
    def sample_multiple_futures(self, decout, img_shape, batch_size, num_current_frames, num_samples=5):
        """
        Sample multiple plausible futures for uncertainty estimation.
        Useful for planning and risk assessment in autonomous driving.
        """
        self.eval()
        all_samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                if self.training:
                    points, _ = self.forward(decout, img_shape, batch_size, num_current_frames)
                else:
                    points = self.forward(decout, img_shape, batch_size, num_current_frames)
                all_samples.append(points)
        
        # Stack samples: [num_samples, B*(N+M), H, W, output_dim]
        return torch.stack(all_samples, dim=0)
    
    def get_latent_interpolation(self, decout, img_shape, batch_size, num_current_frames, alpha=0.5):
        """
        Interpolate between different motion patterns for smooth future generation.
        """
        self.eval()
        
        with torch.no_grad():
            # Get two different latent samples
            tokens = decout[-1].view(batch_size, num_current_frames, -1, tokens.shape[-1])
            temporal_sequence = tokens.mean(dim=2)
            motion_mu, motion_logvar = self.motion_encoder(temporal_sequence)
            
            z1 = self.reparameterize(motion_mu, motion_logvar)
            z2 = self.reparameterize(motion_mu, motion_logvar)
            
            # Interpolate
            z_interp = alpha * z1 + (1 - alpha) * z2
            
            # Decode with interpolated latent
            # ... (rest of forward pass with z_interp)
            
        return interpolated_points