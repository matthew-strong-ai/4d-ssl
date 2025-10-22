"""
Loss functions for C-VAE future prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cvae_future_loss(pred_points, gt_points, kl_loss, current_frames, 
                     kl_weight=0.1, future_weight=2.0, consistency_weight=0.05):
    """
    Combined loss for C-VAE future prediction.
    
    Args:
        pred_points: [B*(N+M), H, W, 3] - predicted points (current + future)
        gt_points: [B*(N+M), H, W, 3] - ground truth points
        kl_loss: scalar - KL divergence from VAE
        current_frames: int - number of current frames (N)
        kl_weight: weight for KL divergence regularization
        future_weight: weight for future frame accuracy
        consistency_weight: weight for temporal consistency
        
    Returns:
        total_loss: combined loss
        loss_dict: dictionary of individual loss components
    """
    BNM, H, W, _ = pred_points.shape
    B = BNM // (current_frames + 3)  # Assuming 3 future frames
    N = current_frames
    M = 3  # future frames
    
    # Reshape to separate current and future
    pred_points = pred_points.view(B, N + M, H, W, 3)
    gt_points = gt_points.view(B, N + M, H, W, 3)
    
    pred_current = pred_points[:, :N]
    pred_future = pred_points[:, N:]
    gt_current = gt_points[:, :N]
    gt_future = gt_points[:, N:]
    
    # 1. Current frame reconstruction loss (L1 + L2)
    current_l1 = F.l1_loss(pred_current, gt_current)
    current_l2 = F.mse_loss(pred_current, gt_current)
    current_loss = current_l1 + 0.5 * current_l2
    
    # 2. Future frame prediction loss (higher weight)
    future_l1 = F.l1_loss(pred_future, gt_future)
    future_l2 = F.mse_loss(pred_future, gt_future)
    future_loss = future_l1 + 0.5 * future_l2
    
    # 3. Temporal consistency loss (smooth motion)
    pred_all = pred_points  # [B, N+M, H, W, 3]
    temporal_diff = pred_all[:, 1:] - pred_all[:, :-1]  # Frame differences
    # Encourage smooth changes (second derivative)
    temporal_smooth = F.mse_loss(temporal_diff[:, 1:], temporal_diff[:, :-1])
    
    # 4. Depth consistency loss (ensure positive depths)
    depth_pred = pred_points[..., 2]  # Z coordinates
    depth_loss = F.relu(-depth_pred + 0.1).mean()  # Penalize negative/tiny depths
    
    # 5. Motion magnitude consistency (prevent explosive motion)
    motion_magnitudes = torch.norm(temporal_diff, dim=-1)  # [B, N+M-1, H, W]
    motion_loss = F.mse_loss(motion_magnitudes[1:], motion_magnitudes[:-1])
    
    # Combine losses
    total_loss = (current_loss + 
                  future_weight * future_loss + 
                  kl_weight * kl_loss +
                  consistency_weight * temporal_smooth +
                  0.1 * depth_loss +
                  0.05 * motion_loss)
    
    loss_dict = {
        'current_loss': current_loss.item(),
        'future_loss': future_loss.item(), 
        'kl_loss': kl_loss.item(),
        'temporal_consistency': temporal_smooth.item(),
        'depth_loss': depth_loss.item(),
        'motion_loss': motion_loss.item(),
        'total_loss': total_loss.item()
    }
    
    return total_loss, loss_dict


def cvae_uncertainty_loss(multiple_predictions, gt_points, current_frames):
    """
    Loss that encourages diverse but plausible predictions.
    
    Args:
        multiple_predictions: [num_samples, B*(N+M), H, W, 3] - multiple sampled futures
        gt_points: [B*(N+M), H, W, 3] - ground truth
        current_frames: int - number of current frames
        
    Returns:
        diversity_loss: encourages diverse predictions
        accuracy_loss: best prediction should match ground truth
    """
    num_samples, BNM, H, W, _ = multiple_predictions.shape
    
    # 1. Diversity loss - encourage different predictions
    # Compute pairwise distances between predictions
    pred_flat = multiple_predictions.view(num_samples, -1)  # [num_samples, BNM*H*W*3]
    
    diversity_loss = 0
    count = 0
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            # L2 distance between predictions
            dist = F.mse_loss(pred_flat[i], pred_flat[j])
            diversity_loss -= dist  # Negative because we want to maximize diversity
            count += 1
    
    diversity_loss /= count if count > 0 else 1
    
    # 2. Accuracy loss - best prediction should be close to ground truth
    # Compute loss for each sample
    sample_losses = []
    for s in range(num_samples):
        sample_loss = F.mse_loss(multiple_predictions[s], gt_points)
        sample_losses.append(sample_loss)
    
    # Use minimum loss (best prediction)
    accuracy_loss = torch.min(torch.stack(sample_losses))
    
    return diversity_loss, accuracy_loss


class CVAEFutureLosses:
    """Collection of C-VAE specific loss functions."""
    
    @staticmethod
    def elbo_loss(pred_points, gt_points, mu, logvar, current_frames,
                  reconstruction_weight=1.0, kl_weight=0.1):
        """
        Evidence Lower BOund (ELBO) loss for VAE.
        
        Args:
            pred_points: [B*(N+M), H, W, 3] - predicted points
            gt_points: [B*(N+M), H, W, 3] - ground truth points
            mu: [B, latent_dim] - latent mean
            logvar: [B, latent_dim] - latent log variance
            current_frames: int - number of current frames
            reconstruction_weight: weight for reconstruction term
            kl_weight: weight for KL divergence term
            
        Returns:
            elbo_loss: negative ELBO (to minimize)
            loss_components: dict of individual terms
        """
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(pred_points, gt_points)
        
        # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0,I)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # ELBO = -[log p(x|z) - KL(q(z|x)||p(z))]
        # We minimize negative ELBO
        elbo = reconstruction_weight * reconstruction_loss + kl_weight * kl_divergence
        
        loss_components = {
            'reconstruction': reconstruction_loss.item(),
            'kl_divergence': kl_divergence.item(),
            'elbo': elbo.item()
        }
        
        return elbo, loss_components
    
    @staticmethod
    def progressive_loss(pred_points, gt_points, current_frames, 
                        base_weight=1.0, progression_factor=1.5):
        """
        Progressive loss that weights future frames more heavily the further they are.
        
        Args:
            pred_points: [B*(N+M), H, W, 3] - predicted points
            gt_points: [B*(N+M), H, W, 3] - ground truth points
            current_frames: int - number of current frames
            base_weight: base weight for current frames
            progression_factor: multiplicative factor for future frames
            
        Returns:
            weighted_loss: progressively weighted loss
        """
        BNM, H, W, _ = pred_points.shape
        B = BNM // (current_frames + 3)  # Assuming 3 future frames
        N = current_frames
        M = 3
        
        pred_points = pred_points.view(B, N + M, H, W, 3)
        gt_points = gt_points.view(B, N + M, H, W, 3)
        
        total_loss = 0
        for t in range(N + M):
            if t < N:
                # Current frames
                weight = base_weight
            else:
                # Future frames - weight increases with time
                future_idx = t - N
                weight = base_weight * (progression_factor ** (future_idx + 1))
            
            frame_loss = F.mse_loss(pred_points[:, t], gt_points[:, t])
            total_loss += weight * frame_loss
        
        return total_loss
    
    @staticmethod
    def motion_aware_loss(pred_points, gt_points, current_frames, motion_weight=0.2):
        """
        Loss that explicitly models motion consistency.
        
        Args:
            pred_points: [B*(N+M), H, W, 3] - predicted points
            gt_points: [B*(N+M), H, W, 3] - ground truth points
            current_frames: int - number of current frames
            motion_weight: weight for motion consistency term
            
        Returns:
            total_loss: motion-aware loss
        """
        BNM, H, W, _ = pred_points.shape
        B = BNM // (current_frames + 3)
        N = current_frames
        M = 3
        
        pred_points = pred_points.view(B, N + M, H, W, 3)
        gt_points = gt_points.view(B, N + M, H, W, 3)
        
        # Standard reconstruction loss
        reconstruction_loss = F.mse_loss(pred_points, gt_points)
        
        # Motion consistency loss
        pred_motion = pred_points[:, 1:] - pred_points[:, :-1]  # [B, N+M-1, H, W, 3]
        gt_motion = gt_points[:, 1:] - gt_points[:, :-1]
        
        motion_loss = F.mse_loss(pred_motion, gt_motion)
        
        # Motion acceleration (smoothness)
        pred_accel = pred_motion[:, 1:] - pred_motion[:, :-1]
        gt_accel = gt_motion[:, 1:] - gt_motion[:, :-1]
        accel_loss = F.mse_loss(pred_accel, gt_accel)
        
        total_loss = reconstruction_loss + motion_weight * (motion_loss + 0.5 * accel_loss)
        
        return total_loss