#!/usr/bin/env python3
"""
Validation utilities for model evaluation and testing.

This module contains functions for running validation, computing metrics,
and handling validation-specific preprocessing and visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import wandb
from PIL import Image as PILImage

from losses import Pi3Losses, PointCloudLosses, NormalLosses
from .visualization_utils import tensor_to_image


def align_prediction_shapes(predictions, pseudo_gt):
    """
    Align predictions and pseudo_gt to have same H, W dimensions.
    
    Args:
        predictions: Model predictions dict
        pseudo_gt: Ground truth dict
        
    Returns:
        Tuple of (aligned_predictions, aligned_pseudo_gt)
    """
    target_h, target_w = pseudo_gt['local_points'].shape[2:4]
    
    for key in ['conf', 'points', 'local_points']:
        if key in predictions:
            tensor = predictions[key]  # (B, N, H, W, C)
            B, N, H, W, C = tensor.shape
            if H != target_h or W != target_w:
                reshaped = tensor.permute(0, 1, 4, 2, 3).reshape(B * N, C, H, W)
                resized = torch.nn.functional.interpolate(reshaped, size=(target_h, target_w), mode='bilinear', align_corners=False)
                predictions[key] = resized.reshape(B, N, C, target_h, target_w).permute(0, 1, 3, 4, 2)
    
    return predictions, pseudo_gt


def denormalize_intrinsics(intrinsics, image_shape):
    """
    More robust denormalization handling different tensor shapes.
    
    Args:
        intrinsics: Camera intrinsics tensor
        image_shape: Target image shape (H, W)
        
    Returns:
        Denormalized intrinsics tensor
    """
    H, W = image_shape[-2:]
    
    # Handle batch and temporal dimensions
    original_shape = intrinsics.shape
    intrinsics_flat = intrinsics.view(-1, *intrinsics.shape[-2:])
    
    if intrinsics.shape[-2:] == (3, 3):
        # Scale the intrinsics matrix
        scale_matrix = torch.tensor([
            [W, 1, W],
            [1, H, H], 
            [1, 1, 1]
        ], device=intrinsics.device, dtype=intrinsics.dtype)
        
        intrinsics_flat = intrinsics_flat * scale_matrix[None, ...]
    
    return intrinsics_flat.view(original_shape)


def compute_validation_metrics(predictions, pseudo_gt):
    """
    Compute validation metrics between predictions and ground truth.
    
    Args:
        predictions: Model predictions dict
        pseudo_gt: Ground truth dict
        
    Returns:
        Dict of computed metrics
    """
    metrics = {}
    
    # Simple L1 loss between predicted and ground truth 3D points
    pred_points = predictions['local_points']  # [B, T, H, W, 3]
    gt_points = pseudo_gt['local_points']      # [B, T, H, W, 3]
    metrics['unweighted_l1_points'] = torch.nn.functional.l1_loss(pred_points, gt_points, reduction='mean')
    
    # Scale-corrected point cloud loss (L1 loss after applying optimal scale)
    scale_factor = PointCloudLosses.optimal_scale_factor(pred_points, gt_points)
    scaled_pred_points = scale_factor * pred_points
    metrics['scale_corrected_l1_points'] = torch.nn.functional.l1_loss(scaled_pred_points, gt_points, reduction='mean')
    
    # Simple pose loss (L1 loss between camera poses)
    pred_poses = predictions['camera_poses']  # [B, T, 4, 4]
    gt_poses = pseudo_gt['camera_poses']      # [B, T, 4, 4]
    metrics['unweighted_pose_loss'] = torch.nn.functional.l1_loss(pred_poses, gt_poses, reduction='mean')
    
    # Confidence metrics if available
    if 'conf' in predictions and 'conf' in pseudo_gt:
        pred_conf = predictions['conf']
        gt_conf = pseudo_gt['conf']
        metrics['conf_l1_loss'] = torch.nn.functional.l1_loss(pred_conf, gt_conf, reduction='mean')
    
    # Depth metrics
    pred_depth = pred_points[..., 2]  # Z component
    gt_depth = gt_points[..., 2]
    metrics['depth_l1_loss'] = torch.nn.functional.l1_loss(pred_depth, gt_depth, reduction='mean')
    metrics['depth_l2_loss'] = torch.nn.functional.mse_loss(pred_depth, gt_depth, reduction='mean')
    
    # Relative depth error
    valid_mask = gt_depth > 0.1  # Avoid division by very small depths
    if valid_mask.any():
        rel_error = torch.abs(pred_depth[valid_mask] - gt_depth[valid_mask]) / (gt_depth[valid_mask] + 1e-8)
        metrics['relative_depth_error'] = rel_error.mean()
    
    return metrics


def save_validation_visualizations(predictions, pseudo_gt, rgb_frames, step, cfg, wandb_run=None):
    """
    Save validation visualizations to disk and WandB.
    
    Args:
        predictions: Model predictions dict
        pseudo_gt: Ground truth dict  
        rgb_frames: RGB input frames
        step: Current training step
        cfg: Configuration object
        wandb_run: WandB run object
        
    Returns:
        Dict of saved visualization paths
    """
    saved_files = {}
    
    try:
        # Convert to numpy
        val_predictions = {}
        for key in predictions.keys():
            if key not in ['all_decoder_features', 'all_positional_encoding']:
                if isinstance(predictions[key], torch.Tensor):
                    val_predictions[key] = predictions[key].clone().detach().cpu().numpy().squeeze(0)  # remove batch dimension

        # === DEPTH VISUALIZATION ===
        if 'local_points' in val_predictions:
            local_points = val_predictions['local_points']  # shape (T, H, W, 3)
            depth_maps = local_points[..., 2]  # shape (T, H, W)
            val_depth_images_for_wandb = []
            
            for t in range(min(depth_maps.shape[0], 6)):  # Limit to 6 frames
                depth = depth_maps[t]
                vmin, vmax = np.percentile(depth, 2), np.percentile(depth, 98)
                norm_depth = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
                # Apply viridis colormap
                colored = plt.get_cmap('viridis')(norm_depth)[:, :, :3]  # shape (H, W, 3), drop alpha
                colored_uint8 = (colored * 255).astype(np.uint8)
                
                # Save to disk with validation prefix
                filename = f"val_depth_frame_{t}_viridis.png"
                imageio.imwrite(filename, colored_uint8)
                saved_files[f'depth_frame_{t}'] = filename
                print(f"💙 Saved validation depth map for frame {t} to {filename}")
                
                # Prepare for WandB
                if cfg.WANDB.USE_WANDB and wandb_run:
                    pil_image = PILImage.fromarray(colored_uint8)
                    val_depth_images_for_wandb.append(wandb.Image(pil_image, caption=f"Val Depth Frame {t}"))
            
            if wandb_run and val_depth_images_for_wandb:
                wandb_run.log({"val_visualizations/depth_maps": val_depth_images_for_wandb}, step=step)

        # === CONFIDENCE VISUALIZATION ===
        if 'conf' in val_predictions:
            conf_maps = val_predictions['conf']  # shape (T, H, W, 1) or (T, H, W)
            if conf_maps.ndim == 4:  # (T, H, W, 1)
                conf_maps = conf_maps.squeeze(-1)  # (T, H, W)
            
            val_confidence_images_for_wandb = []
            for t in range(min(conf_maps.shape[0], 6)):  # Limit to 6 frames
                conf = conf_maps[t]  # (H, W)
                # Apply sigmoid if values are not in [0,1] range
                if conf.min() < 0 or conf.max() > 1:
                    conf = 1 / (1 + np.exp(-conf))  # sigmoid
                
                # Apply hot colormap for confidence (red=high confidence, blue=low confidence)
                colored = plt.get_cmap('hot')(conf)[:, :, :3]  # shape (H, W, 3), drop alpha
                colored_uint8 = (colored * 255).astype(np.uint8)
                
                # Save to disk with validation prefix
                filename = f"val_confidence_frame_{t}_hot.png"
                imageio.imwrite(filename, colored_uint8)
                saved_files[f'confidence_frame_{t}'] = filename
                print(f"💙 Saved validation confidence map for frame {t} to {filename}")
                
                # Prepare for WandB
                if cfg.WANDB.USE_WANDB and wandb_run:
                    pil_image = PILImage.fromarray(colored_uint8)
                    val_confidence_images_for_wandb.append(wandb.Image(pil_image, caption=f"Val Confidence Frame {t}"))
            
            if wandb_run and val_confidence_images_for_wandb:
                wandb_run.log({"val_visualizations/confidence_maps": val_confidence_images_for_wandb}, step=step)

        # === NORMAL MAP VISUALIZATION ===
        if 'local_points' in val_predictions:
            try:
                # Convert back to torch tensor for normal computation
                points_tensor = torch.from_numpy(val_predictions['local_points']).float()  # (T, H, W, 3)
                
                val_normal_images_for_wandb = []
                for t in range(min(points_tensor.shape[0], 6)):  # Limit to 6 frames
                    # Compute normals for this frame
                    frame_points = points_tensor[t:t+1]  # (1, H, W, 3) - add batch dimension
                    normals = NormalLosses.compute_normals_from_grid(frame_points)  # (1, H-2, W-2, 3)
                    normals = normals.squeeze(0).numpy()  # (H-2, W-2, 3)
                    
                    # Convert normals from [-1,1] to [0,1] for visualization
                    # RGB channels represent X, Y, Z components of normal vectors
                    normals_vis = (normals + 1.0) * 0.5  # [-1,1] -> [0,1]
                    normals_vis = np.clip(normals_vis, 0, 1)
                    normals_uint8 = (normals_vis * 255).astype(np.uint8)
                    
                    # Save to disk with validation prefix
                    filename = f"val_normal_frame_{t}_rgb.png"
                    imageio.imwrite(filename, normals_uint8)
                    saved_files[f'normal_frame_{t}'] = filename
                    print(f"💙 Saved validation normal map for frame {t} to {filename}")
                    
                    # Prepare for WandB
                    if cfg.WANDB.USE_WANDB and wandb_run:
                        pil_image = PILImage.fromarray(normals_uint8)
                        val_normal_images_for_wandb.append(wandb.Image(pil_image, caption=f"Val Normal Frame {t}"))
                
                if wandb_run and val_normal_images_for_wandb:
                    wandb_run.log({"val_visualizations/normal_maps": val_normal_images_for_wandb}, step=step)
                    
            except Exception as e:
                print(f"⚠️ Error generating validation normal maps: {e}")

        # === RGB FRAME LOGGING ===
        if rgb_frames is not None and wandb_run:
            rgb_images = []
            for t in range(min(len(rgb_frames), 6)):  # Log up to 6 frames
                img = tensor_to_image(rgb_frames[t])
                rgb_images.append(wandb.Image(img, caption=f"Frame {t}"))
            
            wandb_run.log({"val_visualizations/rgb_frames": rgb_images}, step=step)
            print(f"📸 Logged {len(rgb_images)} RGB frames to WandB")

    except Exception as e:
        print(f"❌ Error in validation visualization: {e}")
        import traceback
        traceback.print_exc()
    
    return saved_files


def run_validation(train_model, frozen_model, val_dataloader, cfg, accelerator, 
                   preprocess_image, dtype=torch.bfloat16, global_step=0, wandb_run=None):
    """
    Run validation and return validation metrics.
    
    Args:
        train_model: The training model
        frozen_model: The frozen Pi3 model for ground truth
        val_dataloader: Validation dataloader
        cfg: YACS configuration
        accelerator: Accelerate accelerator
        preprocess_image: Image preprocessing function
        dtype: Data type for mixed precision
        global_step: Global training step
        wandb_run: WandB run object
        
    Returns:
        dict: Validation metrics
    """
    train_model.eval()
    
    total_val_loss = 0.0
    total_point_loss = 0.0
    total_camera_loss = 0.0
    total_conf_loss = 0.0
    total_frozen_decoder_loss = 0.0
    
    # Add unweighted validation losses for tracking
    total_unweighted_l1_points = 0.0
    total_unweighted_pose_loss = 0.0
    total_scale_corrected_l1_points = 0.0
    num_batches = 0
    max_batches = cfg.VALIDATION.VAL_SAMPLES if cfg.VALIDATION.VAL_SAMPLES > 0 else len(val_dataloader)
    
    print(f"🔍 Running validation on {min(max_batches, len(val_dataloader))} samples...")
    
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            if step >= max_batches:
                break
                
            # Process batch properly for any batch size
            X = batch[0]  # (B, m, C, H, W) - current frames
            y = batch[1]  # (B, n, C, H, W) - future frames
            X_all = torch.cat([X, y], dim=1)  # (B, T, C, H, W) where T = m + n
            
            # For validation, both models use unaugmented data
            batch_size = X_all.shape[0]
            if batch_size == 1:
                # Optimize for batch_size=1 (most common case)
                if cfg.MODEL.ENCODER_NAME == "dinov2":
                    video_tensor_unaugmented = preprocess_image(X_all[0], target_size=518, patch_size=14).unsqueeze(0)
                else:
                    video_tensor_unaugmented = preprocess_image(X_all[0]).unsqueeze(0)

                video_tensor = preprocess_image(X_all[0], target_size=518, patch_size=14).unsqueeze(0)
            else:
                # Handle larger batch sizes
                video_tensors = []
                for b in range(batch_size):
                    # Get single sample: (T, C, H, W)
                    sample = X_all[b]  # (T, C, H, W)
                    # Preprocess this sample without augmentations
                    processed_sample = preprocess_image(sample)  # (T, C, H, W)
                    video_tensors.append(processed_sample)
                
                # Stack to create batch: (B, T, C, H, W)
                video_tensor = torch.stack(video_tensors, dim=0)
            
            subset_video_tensor = video_tensor_unaugmented[:, :cfg.MODEL.M]  # (B, m, C, H, W)
            
            # Get ground truth from frozen model
            with torch.amp.autocast('cuda', dtype=dtype):
                pseudo_gt = frozen_model(video_tensor)
            
            # Get predictions from training model
            with torch.amp.autocast('cuda', dtype=dtype):
                predictions = train_model(subset_video_tensor)
            
            # Align prediction and pseudo_gt shapes before loss computation
            predictions, pseudo_gt = align_prediction_shapes(predictions, pseudo_gt)
            
            # Generate detection targets if detection head is enabled
            detection_targets = None
            if cfg.MODEL.USE_DETECTION_HEAD and cfg.LOSS.DETECTION_LOSS_WEIGHT > 0:
                pass
            
            # Compute validation loss with optional confidence weighting
            if cfg.LOSS.USE_CONF_WEIGHTED_POINTS:
                point_map_loss, camera_pose_loss, conf_loss, normal_loss, segmentation_loss, motion_loss, frozen_decoder_loss = Pi3Losses.pi3_loss_with_confidence_weighting(
                    predictions, pseudo_gt, m_frames=cfg.MODEL.M, future_frame_weight=cfg.LOSS.FUTURE_FRAME_WEIGHT,
                    gamma=cfg.LOSS.CONF_GAMMA, alpha=cfg.LOSS.CONF_ALPHA, use_conf_weighted_points=True, gradient_weight=cfg.LOSS.GRADIENT_WEIGHT,
                    normal_loss_weight=cfg.LOSS.NORMAL_LOSS_WEIGHT
                )
            else:
                point_map_loss, camera_pose_loss, conf_loss, normal_loss, segmentation_loss, motion_loss, frozen_decoder_loss = Pi3Losses.pi3_loss(
                    predictions, pseudo_gt, m_frames=cfg.MODEL.M, future_frame_weight=cfg.LOSS.FUTURE_FRAME_WEIGHT, gradient_weight=cfg.LOSS.GRADIENT_WEIGHT,
                    normal_loss_weight=cfg.LOSS.NORMAL_LOSS_WEIGHT
                )
            
            val_loss = (
                cfg.LOSS.PC_LOSS_WEIGHT * point_map_loss
                + cfg.LOSS.POSE_LOSS_WEIGHT * camera_pose_loss
                + cfg.LOSS.CONF_LOSS_WEIGHT * conf_loss
                + cfg.LOSS.NORMAL_LOSS_WEIGHT * normal_loss
                + cfg.LOSS.SEGMENTATION_LOSS_WEIGHT * segmentation_loss
                + cfg.LOSS.MOTION_LOSS_WEIGHT * motion_loss
                + cfg.LOSS.FROZEN_DECODER_SUPERVISION_WEIGHT * frozen_decoder_loss
            )
            
            # Compute additional metrics
            metrics = compute_validation_metrics(predictions, pseudo_gt)
            
            # Accumulate losses
            total_val_loss += val_loss.item()
            total_point_loss += point_map_loss.item()
            total_camera_loss += camera_pose_loss.item()
            total_conf_loss += conf_loss.item() if torch.is_tensor(conf_loss) else conf_loss
            total_frozen_decoder_loss += frozen_decoder_loss.item() if torch.is_tensor(frozen_decoder_loss) else frozen_decoder_loss
            
            # Accumulate unweighted losses
            total_unweighted_l1_points += metrics['unweighted_l1_points'].item()
            total_unweighted_pose_loss += metrics['unweighted_pose_loss'].item()
            total_scale_corrected_l1_points += metrics['scale_corrected_l1_points'].item()
            
            num_batches += 1
            
            # Save validation visualizations (first batch only)
            if step == 0 and cfg.OUTPUT.SAVE_DEPTHS:
                rgb_frames = X_all[0] if len(X_all) > 0 else None  # First batch
                save_validation_visualizations(predictions, pseudo_gt, rgb_frames, global_step, cfg, wandb_run)
    
    # Calculate average metrics
    if num_batches > 0:
        avg_metrics = {
            'val_loss': total_val_loss / num_batches,
            'val_point_loss': total_point_loss / num_batches,
            'val_camera_loss': total_camera_loss / num_batches,
            'val_conf_loss': total_conf_loss / num_batches,
            'val_frozen_decoder_loss': total_frozen_decoder_loss / num_batches,
            'val_unweighted_l1_points': total_unweighted_l1_points / num_batches,
            'val_unweighted_pose_loss': total_unweighted_pose_loss / num_batches,
            'val_scale_corrected_l1_points': total_scale_corrected_l1_points / num_batches,
        }
    else:
        avg_metrics = {}
    
    train_model.train()  # Return to training mode
    return avg_metrics