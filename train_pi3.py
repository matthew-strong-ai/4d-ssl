import time

import argparse
import os
import torch

import numpy as np

def check_for_nans(tensor, name, step=None):
    """
    Check tensor for NaN values and print detailed diagnostics.
    
    Args:
        tensor: Tensor to check
        name: Name for logging
        step: Optional step number for context
    
    Returns:
        bool: True if NaNs found, False otherwise
    """
    if torch.isnan(tensor).any():
        step_info = f" at step {step}" if step is not None else ""
        print(f"❌ NaN detected in {name}{step_info}")
        print(f"   Shape: {tensor.shape}")
        print(f"   Min: {tensor.min().item()}, Max: {tensor.max().item()}")
        print(f"   NaN count: {torch.isnan(tensor).sum().item()}")
        if tensor.numel() < 100:  # Only print small tensors
            print(f"   Values: {tensor}")
        return True
    
    # Also check for inf values
    if torch.isinf(tensor).any():
        step_info = f" at step {step}" if step is not None else ""
        print(f"⚠️ Inf detected in {name}{step_info}")
        print(f"   Shape: {tensor.shape}")
        print(f"   Min: {tensor.min().item()}, Max: {tensor.max().item()}")
        print(f"   Inf count: {torch.isinf(tensor).sum().item()}")
        return True
    
    return False

def check_model_parameters(model, name, step=None):
    """Check model parameters for NaN/Inf values."""
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            if check_for_nans(param.grad, f"{name}.{param_name}.grad", step):
                return True
        if check_for_nans(param, f"{name}.{param_name}", step):
            return True
    return False

from accelerate import Accelerator
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb
import torchvision.transforms.functional as TF
import random

from utils.geometry_torch import recover_focal_shift
import utils3d

from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image, get_default_transforms
from SpaTrackerV2.ssl_image_dataset import SequenceLearningDataset


# add to path where pi3 is located (one folder deep relative to this file)
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pi3"))

from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3, AutonomyPi3

# import pi3 losses
from losses import Pi3Losses

# GSAM2 for mask generation (optional)
try:
    from vision.gsam2_class import GSAM2
    GSAM2_AVAILABLE = True
except ImportError:
    GSAM2_AVAILABLE = False
    print("⚠️ GSAM2 not available - install required dependencies or check vision/gsam2_class.py")

def apply_random_augmentations(images, training=True):
    """
    Apply random augmentations to images with different amounts per image.
    
    Args:
        images: Tensor of shape (T, C, H, W) where T is number of frames
        training: Whether to apply augmentations (only during training)
        
    Returns:
        Augmented images tensor of same shape
    """
    if not training:
        return images
        
    T, C, H, W = images.shape
    augmented_images = []
    
    for t in range(T):
        img = images[t]  # (C, H, W)
        
        # Random color jittering with different amounts per image
        brightness_factor = random.uniform(0.7, 1.3)
        contrast_factor = random.uniform(0.7, 1.3) 
        saturation_factor = random.uniform(0.7, 1.3)
        hue_factor = random.uniform(-0.1, 0.1)
        
        # Apply color jittering
        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)
        img = TF.adjust_saturation(img, saturation_factor)
        img = TF.adjust_hue(img, hue_factor)
        
        # Random Gaussian blur (20% chance)
        if random.random() < 0.2:
            sigma = random.uniform(0.1, 1.0)
            img = TF.gaussian_blur(img, kernel_size=3, sigma=sigma)
        
        # Random grayscale (10% chance)
        if random.random() < 0.1:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
        
        # Clamp values to valid range
        img = torch.clamp(img, 0.0, 1.0)
        
        augmented_images.append(img)
    
    return torch.stack(augmented_images, dim=0)

def run_validation(train_model, frozen_model, val_dataloader, args, accelerator, dtype=torch.bfloat16):
    """
    Run validation and return validation metrics.
    
    Args:
        train_model: The training model
        frozen_model: The frozen Pi3 model for ground truth
        val_dataloader: Validation dataloader
        args: Training arguments
        accelerator: Accelerate accelerator
        dtype: Data type for mixed precision
        
    Returns:
        dict: Validation metrics
    """
    train_model.eval()
    
    total_val_loss = 0.0
    total_point_loss = 0.0
    total_camera_loss = 0.0
    total_conf_loss = 0.0
    
    # Add unweighted validation losses for tracking
    total_unweighted_l1_points = 0.0
    total_unweighted_pose_loss = 0.0
    num_batches = 0
    max_batches = args.val_samples if args.val_samples > 0 else len(val_dataloader)
    
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
                video_tensor = preprocess_image(X_all[0]).unsqueeze(0)  # (1, T, C, H, W)
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
            subset_video_tensor = video_tensor[:, :args.m]  # (B, m, C, H, W)
            
            # Get ground truth from frozen model
            with torch.amp.autocast('cuda', dtype=dtype):
                pseudo_gt = frozen_model(video_tensor)
            
            # Get predictions from training model
            with torch.amp.autocast('cuda', dtype=dtype):
                predictions = train_model(subset_video_tensor)
            
            # Compute validation loss with configurable precision and loss type
            val_dtype = torch.float32 if args.use_fp32_for_losses else dtype
            with torch.amp.autocast('cuda', dtype=val_dtype):
                if args.use_conf_weighted_points:
                    point_map_loss, camera_pose_loss, conf_loss = Pi3Losses.pi3_loss_with_confidence_weighting(
                        predictions, pseudo_gt, m_frames=args.m, future_frame_weight=args.future_frame_weight,
                        gamma=args.conf_gamma, alpha=args.conf_alpha, use_conf_weighted_points=True, gradient_weight=args.gradient_weight
                    )
                else:
                    point_map_loss, camera_pose_loss, conf_loss = Pi3Losses.pi3_loss(
                        predictions, pseudo_gt, m_frames=args.m, future_frame_weight=args.future_frame_weight, gradient_weight=args.gradient_weight
                    )
                
            # Check for NaNs in validation loss components (only if detection enabled)
            nan_detected = False
            if args.detect_nans:
                nan_detected |= check_for_nans(point_map_loss, "val_point_map_loss")
                nan_detected |= check_for_nans(camera_pose_loss, "val_camera_pose_loss") 
                if not args.use_conf_weighted_points:
                    nan_detected |= check_for_nans(conf_loss, "val_conf_loss")
            
            if nan_detected:
                print(f"🚨 NaN detected in validation losses! Using fallback values...")
                val_loss = torch.tensor(float('inf'), device=point_map_loss.device)
            else:
                val_loss = (
                    args.pc_loss_weight * point_map_loss
                    + args.pose_loss_weight * camera_pose_loss
                    + args.conf_loss_weight * conf_loss
                )
            
            # Compute unweighted validation losses for tracking performance
            # Simple L1 loss between predicted and ground truth 3D points
            pred_points = predictions['points']  # [B, T, H, W, 3]
            gt_points = pseudo_gt['points']      # [B, T, H, W, 3]
            unweighted_l1_points = torch.nn.functional.l1_loss(pred_points, gt_points, reduction='mean')
            
            # Simple pose loss (L1 loss between camera poses)
            pred_poses = predictions['camera_poses']  # [B, T, 4, 4]
            gt_poses = pseudo_gt['camera_poses']      # [B, T, 4, 4]
            unweighted_pose_loss = torch.nn.functional.l1_loss(pred_poses, gt_poses, reduction='mean')
            
            # Accumulate losses
            total_val_loss += val_loss.item()
            total_point_loss += point_map_loss.item()
            total_camera_loss += camera_pose_loss.item()
            total_conf_loss += conf_loss.item() if torch.is_tensor(conf_loss) else conf_loss
            
            # Accumulate unweighted losses
            total_unweighted_l1_points += unweighted_l1_points.item()
            total_unweighted_pose_loss += unweighted_pose_loss.item()
            
            num_batches += 1
            
            # Cleanup
            torch.cuda.empty_cache()
    
    # Calculate average losses
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
    avg_point_loss = total_point_loss / num_batches if num_batches > 0 else float('inf')
    avg_camera_loss = total_camera_loss / num_batches if num_batches > 0 else float('inf')
    avg_conf_loss = total_conf_loss / num_batches if num_batches > 0 else float('inf')
    
    # Calculate average unweighted losses
    avg_unweighted_l1_points = total_unweighted_l1_points / num_batches if num_batches > 0 else float('inf')
    avg_unweighted_pose_loss = total_unweighted_pose_loss / num_batches if num_batches > 0 else float('inf')
    
    train_model.train()  # Back to training mode
    
    return {
        'val_loss': avg_val_loss,
        'val_point_loss': avg_point_loss,
        'val_camera_loss': avg_camera_loss,
        'val_conf_loss': avg_conf_loss,
        # Unweighted losses for tracking actual performance
        'val_unweighted_l1_points': avg_unweighted_l1_points,
        'val_unweighted_pose_loss': avg_unweighted_pose_loss,
        'num_samples': num_batches
    }


def denormalize_intrinsics(intrinsics, image_shape):
    """More robust denormalization handling different tensor shapes."""
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


# ================================================================================================
# GSAM2 Integration Functions
# ================================================================================================

def initialize_gsam2(args, accelerator):
    """
    Initialize GSAM2 model if enabled and available.
    
    Args:
        args: Training arguments containing GSAM2 configuration
        accelerator: Accelerate object for device handling
        
    Returns:
        GSAM2 instance or None if disabled/unavailable
    """
    if not args.use_gsam2:
        return None
        
    if not GSAM2_AVAILABLE:
        print("❌ GSAM2 requested but not available. Please install dependencies.")
        return None
    
    if accelerator.is_main_process:
        print(f"🎯 Initializing GSAM2 for mask generation...")
        print(f"   Prompt: '{args.gsam2_prompt}'")
        print(f"   Frequency: Every {args.gsam2_frequency} steps (0=every step)")
        print(f"   Thresholds: box={args.gsam2_box_threshold}, text={args.gsam2_text_threshold}")
    
    try:
        gsam2 = GSAM2()
        
        # Create save directory if needed
        if args.gsam2_save_masks and accelerator.is_main_process:
            os.makedirs(args.gsam2_save_dir, exist_ok=True)
            print(f"📁 GSAM2 masks will be saved to: {args.gsam2_save_dir}")
        
        return gsam2
        
    except Exception as e:
        print(f"❌ Failed to initialize GSAM2: {e}")
        return None


def convert_torch_to_rgb_frames(tensor_batch):
    """
    Convert torch tensor batch to list of RGB numpy arrays for GSAM2.
    
    Args:
        tensor_batch: Torch tensor of shape (B, T, C, H, W) or (T, C, H, W)
        
    Returns:
        List of RGB numpy arrays each of shape (H, W, 3)
    """
    # Handle different input shapes
    if tensor_batch.dim() == 5:  # (B, T, C, H, W)
        # Take first batch item
        tensor_sequence = tensor_batch[0]  # (T, C, H, W)
    elif tensor_batch.dim() == 4:  # (T, C, H, W)
        tensor_sequence = tensor_batch
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor_batch.shape}")
    
    # Convert to numpy and denormalize
    frames = []
    for t in range(tensor_sequence.shape[0]):
        frame_tensor = tensor_sequence[t]  # (C, H, W)
        
        # Move to CPU and convert to numpy
        frame_np = frame_tensor.detach().cpu().numpy()
        
        # Transpose from (C, H, W) to (H, W, C)
        frame_np = np.transpose(frame_np, (1, 2, 0))
        
        # Denormalize from [-1, 1] or [0, 1] to [0, 255] uint8
        if frame_np.min() < 0:  # Assume normalized to [-1, 1]
            frame_np = (frame_np + 1.0) / 2.0
        
        # Ensure [0, 1] range
        frame_np = np.clip(frame_np, 0, 1)
        
        # Convert to uint8 RGB
        frame_rgb = (frame_np * 255).astype(np.uint8)
        frames.append(frame_rgb)
    
    return frames


def process_batch_with_gsam2(gsam2, tensor_batch, args, step_num, accelerator):
    """
    Process a batch of frames with GSAM2 to generate object masks.
    
    Args:
        gsam2: GSAM2 instance
        tensor_batch: Torch tensor batch (B, T, C, H, W) or (T, C, H, W)
        args: Training arguments
        step_num: Current training step number
        accelerator: Accelerate object
        
    Returns:
        Dictionary containing GSAM2 results or None if processing failed/skipped
    """
    if gsam2 is None:
        return None
    
    # Check if we should process this step
    if args.gsam2_frequency > 0 and step_num % args.gsam2_frequency != 0:
        return None
    
    try:
        # Convert torch tensors to RGB frames
        rgb_frames = convert_torch_to_rgb_frames(tensor_batch)
        
        if accelerator.is_main_process and step_num % 100 == 0:  # Log occasionally
            print(f"🎯 Processing step {step_num} with GSAM2 ({len(rgb_frames)} frames)")
        
        # Process frames with GSAM2
        results = gsam2.process_frames(
            rgb_frames,
            args.gsam2_prompt,
            box_threshold=args.gsam2_box_threshold,
            text_threshold=args.gsam2_text_threshold,
            prompt_type="box",  # Use box prompts for stability
            cleanup_temp=True
        )
        
        # Save detailed masks if requested (only for first few steps if max_steps is set)
        should_save_detailed = (
            args.gsam2_save_masks and 
            accelerator.is_main_process and 
            results['num_objects'] > 0 and
            (args.gsam2_save_masks_max_steps == 0 or step_num < args.gsam2_save_masks_max_steps)
        )
        
        if should_save_detailed:
            save_gsam2_masks(results, rgb_frames, args, step_num)
        
        return results
        
    except Exception as e:
        if accelerator.is_main_process:
            print(f"⚠️ GSAM2 processing failed at step {step_num}: {e}")
        return None


def save_gsam2_masks(results, rgb_frames, args, step_num):
    """
    Save GSAM2 masks and visualization to disk with enhanced PNG visualizations.
    
    Args:
        results: GSAM2 results dictionary
        rgb_frames: Original RGB frames
        args: Training arguments
        step_num: Current training step
    """
    try:
        import os
        import cv2
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from PIL import Image
        
        # Create step directory
        step_dir = os.path.join(args.gsam2_save_dir, f"step_{step_num:06d}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Save mask data as numpy
        mask_file = os.path.join(step_dir, "masks.npz")
        np.savez_compressed(
            mask_file,
            masks=[results['masks'][i] for i in range(len(results['masks']))],
            labels=results['labels'],
            boxes=results['boxes'],
            scores=results['scores'],
            num_objects=results['num_objects'],
            num_frames=results['num_frames']
        )
        
        # Create color palette for different objects
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, results['num_objects'])))
        
        # Save visualizations for first 3 frames
        for frame_idx in range(min(3, len(rgb_frames))):
            frame = rgb_frames[frame_idx]
            frame_masks = results['masks'][frame_idx]
            
            if not frame_masks:  # Skip frames with no detections
                continue
            
            # 1. Save original frame
            orig_path = os.path.join(step_dir, f"frame_{frame_idx:02d}_original.png")
            Image.fromarray(frame).save(orig_path)
            
            # 2. Save individual masks (grayscale)
            for obj_id, mask in frame_masks.items():
                mask_path = os.path.join(step_dir, f"frame_{frame_idx:02d}_mask_obj{obj_id}.png")
                
                # Ensure mask is 2D
                mask_2d = mask
                if mask_2d.ndim > 2:
                    mask_2d = np.squeeze(mask_2d)
                    if mask_2d.ndim > 2:
                        mask_2d = mask_2d[0] if mask_2d.shape[0] == 1 else mask_2d[:, :, 0]
                
                mask_img = (mask_2d * 255).astype(np.uint8)
                Image.fromarray(mask_img, mode='L').save(mask_path)
            
            # 3. Create colored mask overlay
            overlay = frame.copy()
            combined_mask = np.zeros((*frame.shape[:2], 4), dtype=np.uint8)  # RGBA
            
            for i, (obj_id, mask) in enumerate(frame_masks.items()):
                # Ensure mask is 2D
                mask_2d = mask
                if mask_2d.ndim > 2:
                    mask_2d = np.squeeze(mask_2d)
                    if mask_2d.ndim > 2:
                        mask_2d = mask_2d[0] if mask_2d.shape[0] == 1 else mask_2d[:, :, 0]
                
                if mask_2d.sum() == 0:  # Skip empty masks
                    continue
                    
                # Get color for this object
                color = colors[i % len(colors)]
                color_rgb = (np.array(color[:3]) * 255).astype(np.uint8)
                
                # Apply colored mask
                mask_bool = mask_2d.astype(bool)
                overlay[mask_bool] = overlay[mask_bool] * 0.7 + color_rgb * 0.3
                
                # Add to combined mask with alpha
                combined_mask[mask_bool] = [*color_rgb, 128]  # Semi-transparent
            
            # Save colored overlay
            overlay_path = os.path.join(step_dir, f"frame_{frame_idx:02d}_overlay.png")
            Image.fromarray(overlay).save(overlay_path)
            
            # 4. Create side-by-side comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original
            axes[0].imshow(frame)
            axes[0].set_title("Original Frame")
            axes[0].axis('off')
            
            # Masks only
            if len(frame_masks) > 0:
                all_masks = np.zeros(frame.shape[:2])
                for i, (obj_id, mask) in enumerate(frame_masks.items()):
                    # Ensure mask is 2D
                    mask_2d = mask
                    if mask_2d.ndim > 2:
                        mask_2d = np.squeeze(mask_2d)
                        if mask_2d.ndim > 2:
                            mask_2d = mask_2d[0] if mask_2d.shape[0] == 1 else mask_2d[:, :, 0]
                    all_masks += mask_2d * (i + 1)
                
                axes[1].imshow(all_masks, cmap='tab10', vmin=0, vmax=10)
                axes[1].set_title(f"Detected Objects ({len(frame_masks)})")
            else:
                axes[1].imshow(np.zeros(frame.shape[:2]), cmap='gray')
                axes[1].set_title("No Objects Detected")
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay with Labels")
            axes[2].axis('off')
            
            # Add object labels and scores
            label_text = []
            for i, (obj_id, label) in enumerate(zip(frame_masks.keys(), results['labels'])):
                if obj_id <= len(results['scores']):
                    score = results['scores'][obj_id-1] if obj_id-1 < len(results['scores']) else 0.0
                    label_text.append(f"Obj{obj_id}: {label} ({score:.2f})")
            
            if label_text:
                fig.suptitle(" | ".join(label_text), fontsize=10)
            
            plt.tight_layout()
            comparison_path = os.path.join(step_dir, f"frame_{frame_idx:02d}_comparison.png")
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 5. Create bounding box visualization
            bbox_frame = frame.copy()
            for i, (obj_id, mask) in enumerate(frame_masks.items()):
                # Ensure mask is 2D
                mask_2d = mask
                if mask_2d.ndim > 2:
                    mask_2d = np.squeeze(mask_2d)
                    if mask_2d.ndim > 2:
                        mask_2d = mask_2d[0] if mask_2d.shape[0] == 1 else mask_2d[:, :, 0]
                
                if mask_2d.sum() == 0:
                    continue
                    
                # Calculate bounding box from mask
                rows, cols = np.where(mask_2d)
                if len(rows) > 0 and len(cols) > 0:
                    min_row, max_row = rows.min(), rows.max()
                    min_col, max_col = cols.min(), cols.max()
                    
                    # Draw bounding box
                    color = (colors[i % len(colors)][:3] * 255).astype(int)
                    cv2.rectangle(bbox_frame, (min_col, min_row), (max_col, max_row), color.tolist(), 2)
                    
                    # Add label
                    label_text = f"Obj{obj_id}"
                    if obj_id-1 < len(results['labels']):
                        label_text += f": {results['labels'][obj_id-1]}"
                    
                    cv2.putText(bbox_frame, label_text, (min_col, min_row-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 1)
            
            bbox_path = os.path.join(step_dir, f"frame_{frame_idx:02d}_bboxes.png")
            Image.fromarray(bbox_frame).save(bbox_path)
        
        # 6. Create summary visualization showing all frames
        if len(rgb_frames) > 1:
            n_frames = min(6, len(rgb_frames))
            fig, axes = plt.subplots(2, n_frames, figsize=(3*n_frames, 6))
            if n_frames == 1:
                axes = axes.reshape(2, 1)
            
            for f_idx in range(n_frames):
                frame = rgb_frames[f_idx]
                frame_masks = results['masks'][f_idx]
                
                # Top row: original frames
                axes[0, f_idx].imshow(frame)
                axes[0, f_idx].set_title(f"Frame {f_idx}")
                axes[0, f_idx].axis('off')
                
                # Bottom row: masks
                if frame_masks:
                    all_masks = np.zeros(frame.shape[:2])
                    for i, (obj_id, mask) in enumerate(frame_masks.items()):
                        # Ensure mask is 2D
                        mask_2d = mask
                        if mask_2d.ndim > 2:
                            mask_2d = np.squeeze(mask_2d)
                            if mask_2d.ndim > 2:
                                mask_2d = mask_2d[0] if mask_2d.shape[0] == 1 else mask_2d[:, :, 0]
                        all_masks += mask_2d * (i + 1)
                    axes[1, f_idx].imshow(all_masks, cmap='tab10', vmin=0, vmax=10)
                    axes[1, f_idx].set_title(f"{len(frame_masks)} objects")
                else:
                    axes[1, f_idx].imshow(np.zeros(frame.shape[:2]), cmap='gray')
                    axes[1, f_idx].set_title("No objects")
                axes[1, f_idx].axis('off')
            
            plt.suptitle(f"Step {step_num}: {results['num_objects']} unique objects detected")
            plt.tight_layout()
            sequence_path = os.path.join(step_dir, "sequence_summary.png")
            plt.savefig(sequence_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # Log summary
        with open(os.path.join(step_dir, "summary.txt"), 'w') as f:
            f.write(f"Step: {step_num}\n")
            f.write(f"Prompt: {args.gsam2_prompt}\n")
            f.write(f"Objects detected: {results['num_objects']}\n")
            f.write(f"Labels: {results['labels']}\n")
            f.write(f"Frames processed: {results['num_frames']}\n")
            f.write(f"Detection scores: {results['scores'].tolist()}\n")
            f.write(f"\nFiles created:\n")
            f.write(f"- masks.npz: Raw mask data\n")
            f.write(f"- frame_XX_original.png: Original frames\n")
            f.write(f"- frame_XX_mask_objY.png: Individual object masks\n")
            f.write(f"- frame_XX_overlay.png: Colored mask overlays\n")
            f.write(f"- frame_XX_comparison.png: Side-by-side comparisons\n")
            f.write(f"- frame_XX_bboxes.png: Bounding box visualizations\n")
            f.write(f"- sequence_summary.png: Multi-frame overview\n")
            
    except Exception as e:
        print(f"⚠️ Failed to save GSAM2 masks for step {step_num}: {e}")
        import traceback
        traceback.print_exc()


def log_gsam2_metrics(results, args, step_num, accelerator):
    """
    Log GSAM2 metrics to WandB or console.
    
    Args:
        results: GSAM2 results dictionary
        args: Training arguments
        step_num: Current training step
        accelerator: Accelerate object
    """
    if results is None or not accelerator.is_main_process:
        return
    
    try:
        # Calculate metrics
        total_masks = sum(len(frame_masks) for frame_masks in results['masks'])
        avg_masks_per_frame = total_masks / results['num_frames'] if results['num_frames'] > 0 else 0
        avg_detection_score = np.mean(results['scores']) if len(results['scores']) > 0 else 0
        
        # Log to WandB if available
        if args.use_wandb:
            import wandb
            wandb.log({
                "gsam2/num_objects": results['num_objects'],
                "gsam2/total_masks": total_masks,
                "gsam2/avg_masks_per_frame": avg_masks_per_frame,
                "gsam2/avg_detection_score": avg_detection_score,
                "gsam2/processing_step": step_num
            }, step=step_num)
        
        # Log to console occasionally
        if step_num % 500 == 0:
            print(f"🎯 GSAM2 @ step {step_num}: {results['num_objects']} objects, "
                  f"{avg_masks_per_frame:.1f} masks/frame, "
                  f"score={avg_detection_score:.3f}")
            
    except Exception as e:
        print(f"⚠️ Failed to log GSAM2 metrics: {e}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Training script with Accelerate")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing subfolders of images")
    parser.add_argument("--batch_size", type=int, default=20,
                      help="Batch size for training (number of consecutive images per batch)")
    
    # m current frames, n future frames
    parser.add_argument('--m', type=int, default=3, help='Number of input frames')
    parser.add_argument('--n', type=int, default=3, help='Number of target frames') 

    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping (reduced for stability)")
    parser.add_argument("--use_fp32_for_losses", action='store_true', default=True, help="Use FP32 for loss computations (more stable than BF16)")
    parser.add_argument("--detect_nans", action='store_true', default=True, help="Enable NaN detection and recovery")
    
    # Confidence-weighted point loss parameters
    parser.add_argument("--use_conf_weighted_points", action='store_true', default=False, help="Use confidence-weighted point loss instead of scale-invariant loss")
    parser.add_argument("--conf_gamma", type=float, default=1.0, help="Weight for confidence-weighted reconstruction loss")
    parser.add_argument("--conf_alpha", type=float, default=0.1, help="Weight for confidence regularization term")
    parser.add_argument("--gradient_weight", type=float, default=0.0, help="Weight for gradient comparison term in confidence loss")

    parser.add_argument("--num_epochs", type=int, default=5,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                      help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--warmup_start_factor", type=float, default=0.1,
                      help="Starting factor for warmup (lr will start at lr * warmup_start_factor)")
    
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                      help="Number of gradient accumulation steps")
    parser.add_argument("--val_freq", type=int, default=1000,
                      help="Validate every N steps")
    parser.add_argument("--val_split", type=float, default=0.1,
                      help="Fraction of data to use for validation")
    parser.add_argument("--pc_loss_weight", type=float, default=1.0,
                      help="Point cloud loss weight")
    parser.add_argument("--pose_loss_weight", type=float, default=50.0,
                      help="Camera pose loss weight")
    parser.add_argument("--conf_loss_weight", type=float, default=0.5,
                      help="Confidence loss weight")
    parser.add_argument("--future_frame_weight", type=float, default=2.0,
                      help="Weight multiplier for future frame supervision (>1.0 emphasizes future frames)")
    parser.add_argument("--val_samples", type=int, default=50,
                      help="Number of validation samples to use (-1 for all)")
    parser.add_argument("--early_stopping_patience", type=int, default=1000,
                      help="Early stopping patience (number of validation checks without improvement)")
    

    parser.add_argument("--grid_size", type=int, default=10, help="Grid size for query points")
    parser.add_argument("--n_visualize", type=int, default=3, help="Number of random batches to visualize before training")
    parser.add_argument("--save_npz", type=bool, default=False, help="Save npz files")
    parser.add_argument("--save_depths", type=bool, default=False, help="Save depths files")
    parser.add_argument("--log_freq", type=int, default=50, help="Log every N steps")
    parser.add_argument("--save_freq", type=int, default=10000, help="Check for best model every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--wandb_project", type=str, default="pi3-training", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--use_wandb", action='store_true', default=True, help="Enable Weights & Biases logging")
    parser.add_argument("--use_augmentations", action='store_true', default=True, help="Apply random augmentations during training")
    
    # GSAM2 Integration Parameters
    parser.add_argument("--use_gsam2", action='store_true', default=False, help="Enable GSAM2 mask generation during training")
    parser.add_argument("--gsam2_prompt", type=str, default="person. vehicle. car.", help="Text prompt for GSAM2 object detection")
    parser.add_argument("--gsam2_frequency", type=int, default=10, help="Generate GSAM2 masks every N steps (0 = every step)")
    parser.add_argument("--gsam2_box_threshold", type=float, default=0.25, help="GSAM2 detection confidence threshold")
    parser.add_argument("--gsam2_text_threshold", type=float, default=0.3, help="GSAM2 text matching threshold")
    parser.add_argument("--gsam2_save_masks", action='store_true', default=True, help="Save GSAM2 masks to disk")
    parser.add_argument("--gsam2_save_masks_max_steps", type=int, default=5, help="Maximum number of steps to save detailed GSAM2 visualizations (0=unlimited)")
    parser.add_argument("--gsam2_save_dir", type=str, default="gsam2_masks", help="Directory to save GSAM2 masks")

    
    args = parser.parse_args()
    
    # Validate GSAM2 configuration
    if args.use_gsam2 and not GSAM2_AVAILABLE:
        print("❌ GSAM2 requested (--use_gsam2) but not available.")
        print("   Please install required dependencies or check vision/gsam2_class.py")
        args.use_gsam2 = False  # Disable to prevent errors

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision='bf16'
    )

    grid_size = args.grid_size

    # Find all subdirectories in root_dir
    image_dirs = [os.path.join(args.root_dir, d) for d in os.listdir(args.root_dir)
                  if os.path.isdir(os.path.join(args.root_dir, d))]
    print(f"Found {len(image_dirs)} subfolders:")
    for d in image_dirs:
        print(f"  {d}")


    # Create full dataset
    full_dataset = SequenceLearningDataset(
        image_dirs=image_dirs,
        m=args.m,
        n=args.n,
        transform=get_default_transforms())

    # Split dataset into train and validation
    if args.val_split > 0:
        total_size = len(full_dataset)
        val_size = int(total_size * args.val_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible split
        )
        
        print(f"📊 Dataset split: {train_size} train, {val_size} validation samples")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print(f"📊 Using full dataset for training: {len(train_dataset)} samples")
    
    # Print augmentation status
    if args.use_augmentations:
        print("🎨 Random augmentations enabled: color jittering, Gaussian blur, grayscale")
    else:
        print("🚫 Random augmentations disabled")

    # Optionally visualize a few random batches
    # from SpaTrackerV2.multi_folder_consecutive_images_dataset import visualize_random_samples
    # visualize_random_samples(dataset, n=args.n_visualize)

    # Create dataloaders with enhanced prefetching
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,  # Each item is a batch of images (T, C, H, W)
        shuffle=True,
        num_workers=4,          # Multiple workers for parallel loading
        prefetch_factor=8,      # Prefetch 8 batches per worker
        pin_memory=True,        # Faster GPU transfer
        persistent_workers=True # Keep workers alive between epochs
    )
    
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,  # Don't shuffle validation
            num_workers=2,          # Fewer workers for validation
            prefetch_factor=4,      # Moderate prefetching for validation
            pin_memory=True,
            persistent_workers=True
        )


    pose_loss_weight = args.pose_loss_weight
    pc_loss_weight = args.pc_loss_weight
    conf_loss_weight = args.conf_loss_weight


    print("Initializing and loading Pi3 model...")
    frozen_model = Pi3.from_pretrained("yyfz233/Pi3")
    frozen_model = frozen_model.to(accelerator.device)
    frozen_model.requires_grad_(False)  # freeze parameters
    
    # Initialize GSAM2 for mask generation (optional)
    gsam2 = initialize_gsam2(args, accelerator)

    # Define training model
    train_model = AutonomyPi3(full_N=args.m + args.n, extra_tokens=args.n)
    # train_model = Pi3.from_pretrained("yyfz233/Pi3")

    # Load encoder and rope
    train_model.encoder.load_state_dict(frozen_model.encoder.state_dict())
    train_model.rope.load_state_dict(frozen_model.rope.state_dict())

    # Load decoders
    train_model.decoder.load_state_dict(frozen_model.decoder.state_dict())
    train_model.point_decoder.load_state_dict(frozen_model.point_decoder.state_dict())
    train_model.conf_decoder.load_state_dict(frozen_model.conf_decoder.state_dict())
    train_model.camera_decoder.load_state_dict(frozen_model.camera_decoder.state_dict())
    
    # Load register token
    train_model.register_token.data.copy_(frozen_model.register_token.data)
    
    # Load point and conf heads (compatible parts only)
    # Note: Pi3 uses LinearPts3d, AutonomyPi3 uses FutureLinearPts3d - load compatible weights
    frozen_point_dict = frozen_model.point_head.state_dict()
    train_point_dict = train_model.point_head.state_dict()
    matched_point_dict = {
        k: v for k, v in frozen_point_dict.items()
        if k in train_point_dict and v.shape == train_point_dict[k].shape
    }
    train_point_dict.update(matched_point_dict)
    train_model.point_head.load_state_dict(train_point_dict)
    
    frozen_conf_dict = frozen_model.conf_head.state_dict()
    train_conf_dict = train_model.conf_head.state_dict()
    matched_conf_dict = {
        k: v for k, v in frozen_conf_dict.items()
        if k in train_conf_dict and v.shape == train_conf_dict[k].shape
    }
    train_conf_dict.update(matched_conf_dict)
    train_model.conf_head.load_state_dict(train_conf_dict)
    
    # Load camera head (compatible parts only)
    frozen_camera_dict = frozen_model.camera_head.state_dict()
    train_camera_dict = train_model.camera_head.state_dict()
    matched_camera_dict = {
        k: v for k, v in frozen_camera_dict.items()
        if k in train_camera_dict and v.shape == train_camera_dict[k].shape
    }
    train_camera_dict.update(matched_camera_dict)
    train_model.camera_head.load_state_dict(train_camera_dict)
    
    # Load image normalization buffers
    train_model.image_mean.data.copy_(frozen_model.image_mean.data)
    train_model.image_std.data.copy_(frozen_model.image_std.data)

    train_model.point_head.current_proj.load_state_dict(frozen_model.point_head.proj.state_dict())
    train_model.conf_head.current_proj.load_state_dict(frozen_model.conf_head.proj.state_dict())

    # Define optimizer for train_model
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.learning_rate)
    
    # Create warmup + cosine annealing scheduler
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = min(args.warmup_steps, total_steps // 10)  # Cap warmup at 10% of total steps
    cosine_steps = total_steps - warmup_steps
    
    if warmup_steps > 0:
        # Create warmup scheduler (linear increase from start_factor to 1.0)
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=args.warmup_start_factor, 
            end_factor=1.0,
            total_iters=warmup_steps
        )
        # Create cosine annealing scheduler for after warmup
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)
        # Combine them with SequentialLR
        scheduler = SequentialLR(
            optimizer, 
            [warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_steps]
        )
        print(f"🔥 Using warmup scheduler: {warmup_steps} warmup steps + {cosine_steps} cosine annealing steps")
    else:
        # Fallback to original cosine annealing if no warmup
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        print(f"📊 Using cosine annealing scheduler: {total_steps} total steps")

    # Prepare training components with Accelerator
    if val_dataloader is not None:
        train_model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
            train_model,
            optimizer,
            scheduler,
            train_dataloader,
            val_dataloader
        )
    else:
        train_model, optimizer, scheduler, train_dataloader = accelerator.prepare(
            train_model,
            optimizer, 
            scheduler,
            train_dataloader
        )

    # Move frozen model manually to accelerator.device (but do NOT prepare it if you don't train it)
    frozen_model.to(accelerator.device)
    device = accelerator.device

    # Create checkpoint directory
    if accelerator.is_main_process:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize Weights & Biases
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "learning_rate": args.learning_rate,
                "batch_size": 1,  # Actual batch size
                "grad_accum_steps": args.grad_accum_steps,
                "effective_batch_size": 1 * args.grad_accum_steps,
                "num_epochs": args.num_epochs,
                "m_frames": args.m,
                "n_frames": args.n,
                "future_frame_weight": args.future_frame_weight,
                "max_grad_norm": args.max_grad_norm,
                "architecture": "AutonomyPi3",
                "optimizer": "AdamW",
                "scheduler": "Warmup+CosineAnnealingLR" if warmup_steps > 0 else "CosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "warmup_start_factor": args.warmup_start_factor,
                "total_steps": total_steps,
                "mixed_precision": "fp16",
                "val_split": args.val_split,
                "val_freq": args.val_freq,
                "val_samples": args.val_samples,
                "early_stopping_patience": args.early_stopping_patience,
                "train_samples": len(train_dataloader),
                "val_samples_total": len(val_dataloader) if val_dataloader else 0,
                # GSAM2 Configuration
                "gsam2_enabled": args.use_gsam2,
                "gsam2_prompt": args.gsam2_prompt if args.use_gsam2 else None,
                "gsam2_frequency": args.gsam2_frequency if args.use_gsam2 else None,
                "gsam2_box_threshold": args.gsam2_box_threshold if args.use_gsam2 else None,
                "gsam2_text_threshold": args.gsam2_text_threshold if args.use_gsam2 else None,
                "gsam2_save_masks": args.gsam2_save_masks if args.use_gsam2 else None,
                "gsam2_save_masks_max_steps": args.gsam2_save_masks_max_steps if args.use_gsam2 else None,
            },
            tags=["pi3", "ssl", "training"] + (["gsam2"] if args.use_gsam2 else [])
        )
        print("🚀 Weights & Biases initialized!")

    # TensorBoard SummaryWriter
    if accelerator.is_main_process:
        writer = SummaryWriter("runs/pi3_optimized")

    # Training loop
    global_step = 0
    total_step = 0
    best_loss = float('inf')
    best_val_loss = float('inf')
    running_loss = 0.0
    loss_history = []
    val_loss_history = []
    steps_without_improvement = 0
    
    # Save sample images at the start
    if accelerator.is_main_process:
        print("🖼️ Saving sample training images...")
        try:
            from PIL import Image as PILImage
            
            sample_dir = "./training_samples"
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save a few random training samples (original and augmented)
            sample_batch = next(iter(train_dataloader))
            X_sample, y_sample = sample_batch
            X_all_sample = torch.cat([X_sample, y_sample], dim=1)  # (B, T, C, H, W)
            
            # Save original images first
            sample_tensor_orig = preprocess_image(X_all_sample[0]).cpu()  # (T, C, H, W)
            
            for frame_idx in range(min(6, sample_tensor_orig.shape[0])):  # Save first 6 frames
                img_tensor = sample_tensor_orig[frame_idx]  # (C, H, W)
                img_array = img_tensor.permute(1, 2, 0).numpy()  # (H, W, C)
                img_array = np.clip(img_array, 0, 1)
                img_array = (img_array * 255).astype(np.uint8)
                
                pil_image = PILImage.fromarray(img_array)
                frame_type = "input" if frame_idx < args.m else "target"
                local_idx = frame_idx if frame_idx < args.m else frame_idx - args.m
                save_path = os.path.join(sample_dir, f"original_{frame_type}_frame_{local_idx:02d}.png")
                pil_image.save(save_path)
            
            # Save augmented images if augmentations are enabled
            if args.use_augmentations:
                # Apply augmentations to the sample
                X_all_sample_aug = apply_random_augmentations(X_all_sample[0], training=True)  # (T, C, H, W)
                sample_tensor_aug = preprocess_image(X_all_sample_aug).cpu()  # (T, C, H, W)
                
                for frame_idx in range(min(6, sample_tensor_aug.shape[0])):  # Save first 6 frames
                    img_tensor = sample_tensor_aug[frame_idx]  # (C, H, W)
                    img_array = img_tensor.permute(1, 2, 0).numpy()  # (H, W, C)
                    img_array = np.clip(img_array, 0, 1)
                    img_array = (img_array * 255).astype(np.uint8)
                    
                    pil_image = PILImage.fromarray(img_array)
                    frame_type = "input" if frame_idx < args.m else "target"
                    local_idx = frame_idx if frame_idx < args.m else frame_idx - args.m
                    save_path = os.path.join(sample_dir, f"augmented_{frame_type}_frame_{local_idx:02d}.png")
                    pil_image.save(save_path)
            
            if args.use_augmentations:
                print(f"✅ Saved original and augmented sample images to {sample_dir}")
            else:
                print(f"✅ Saved original sample images to {sample_dir}")
            
        except Exception as e:
            print(f"⚠️ Could not save sample images: {e}")


    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch+1}/{args.num_epochs}", 
            disable=not accelerator.is_local_main_process
        )
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(train_model):
                # Process batch properly for any batch size
                X = batch[0]  # (B, m, C, H, W) - current frames
                y = batch[1]  # (B, n, C, H, W) - future frames
                X_all = torch.cat([X, y], dim=1)  # (B, T, C, H, W) where T = m + n
                
                # Create unaugmented tensor for frozen model (ground truth)
                batch_size = X_all.shape[0]
                if batch_size == 1:
                    # Optimize for batch_size=1 (most common case)
                    video_tensor_unaugmented = preprocess_image(X_all[0]).unsqueeze(0)  # (1, T, C, H, W)
                    
                    # Process with GSAM2 for mask generation (on unaugmented frames)
                    with torch.no_grad():
                        # Clear cache before GSAM2 inference
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gsam2_results = process_batch_with_gsam2(gsam2, X_all, args, global_step, accelerator)
                        if gsam2_results is not None:
                            log_gsam2_metrics(gsam2_results, args, global_step, accelerator)
                        # Clear cache after GSAM2 inference
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    # Handle larger batch sizes
                    video_tensors_unaugmented = []
                    for b in range(batch_size):
                        # Get single sample: (T, C, H, W)
                        sample = X_all[b]  # (T, C, H, W)
                        # Preprocess this sample without augmentations
                        processed_sample = preprocess_image(sample)  # (T, C, H, W)
                        video_tensors_unaugmented.append(processed_sample)
                    
                    # Stack to create batch: (B, T, C, H, W)
                    video_tensor_unaugmented = torch.stack(video_tensors_unaugmented, dim=0)
                    
                    # Process with GSAM2 for mask generation (on unaugmented frames)
                    with torch.no_grad():
                        # Clear cache before GSAM2 inference
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gsam2_results = process_batch_with_gsam2(gsam2, X_all, args, global_step, accelerator)
                        if gsam2_results is not None:
                            log_gsam2_metrics(gsam2_results, args, global_step, accelerator)
                        # Clear cache after GSAM2 inference
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Apply augmentations to each sample in the batch for training model
                augmented_samples = []
                for b in range(batch_size):
                    sample = X_all[b]  # (T, C, H, W)
                    # Apply random augmentations (different for each image in sequence)
                    augmented_sample = apply_random_augmentations(sample, training=args.use_augmentations)
                    augmented_samples.append(augmented_sample)
                
                # Stack back to batch format
                X_all_augmented = torch.stack(augmented_samples, dim=0)  # (B, T, C, H, W)
                
                # Process augmented samples for training model
                if batch_size == 1:
                    # Optimize for batch_size=1 (most common case)
                    video_tensor_augmented = preprocess_image(X_all_augmented[0]).unsqueeze(0)  # (1, T, C, H, W)
                else:
                    # Handle larger batch sizes
                    video_tensors_augmented = []
                    for b in range(batch_size):
                        # Get single sample: (T, C, H, W)
                        sample = X_all_augmented[b]  # (T, C, H, W)
                        # Preprocess this augmented sample
                        processed_sample = preprocess_image(sample)  # (T, C, H, W)
                        video_tensors_augmented.append(processed_sample)
                    
                    # Stack to create batch: (B, T, C, H, W)
                    video_tensor_augmented = torch.stack(video_tensors_augmented, dim=0)
                
                subset_video_tensor = video_tensor_augmented[:, :args.m]  # (B, m, C, H, W) - augmented for training
                # use bfloat for higher dynamic range, but more memory efficient.
                dtype = torch.bfloat16

                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=dtype):
                        pseudo_gt = frozen_model(video_tensor_unaugmented)  # Use unaugmented data for ground truth
                torch.cuda.empty_cache()

                # with this ground truth, let us know propagate Grounding SAM2 in the frames

                # run inference on the training model
                with torch.amp.autocast('cuda', dtype=dtype):
                    predictions = train_model(subset_video_tensor)

                # compute loss between prediction and pseudo_gt with configurable precision and loss type
                loss_dtype = torch.float32 if args.use_fp32_for_losses else dtype
                with torch.amp.autocast('cuda', dtype=loss_dtype):
                    if args.use_conf_weighted_points:
                        point_map_loss, camera_pose_loss, conf_loss = Pi3Losses.pi3_loss_with_confidence_weighting(
                            predictions, pseudo_gt, m_frames=args.m, future_frame_weight=args.future_frame_weight,
                            gamma=args.conf_gamma, alpha=args.conf_alpha, use_conf_weighted_points=True, gradient_weight=args.gradient_weight
                        )
                    else:
                        point_map_loss, camera_pose_loss, conf_loss = Pi3Losses.pi3_loss(
                            predictions, pseudo_gt, m_frames=args.m, future_frame_weight=args.future_frame_weight, gradient_weight=args.gradient_weight
                        )
                
                # Check for NaNs in training loss components (only if detection enabled)
                nan_detected = False
                if args.detect_nans:
                    nan_detected |= check_for_nans(point_map_loss, "point_map_loss", global_step)
                    nan_detected |= check_for_nans(camera_pose_loss, "camera_pose_loss", global_step)
                    if not args.use_conf_weighted_points:
                        nan_detected |= check_for_nans(conf_loss, "conf_loss", global_step)
                
                pi3_loss = (pc_loss_weight * point_map_loss) + (pose_loss_weight * camera_pose_loss) + (conf_loss_weight * conf_loss)
                
                # Check final loss for NaNs (only if detection enabled)
                if args.detect_nans:
                    nan_detected |= check_for_nans(pi3_loss, "pi3_loss", global_step)
                    
                    if nan_detected:
                        print(f"🚨 NaN detected at training step {global_step}! Skipping this batch...")
                        # Skip backward pass for this batch
                        continue

                # loss handling and optimization
                accelerator.backward(pi3_loss)
                
                if accelerator.sync_gradients:
                    # Check for NaN gradients before clipping (only if detection enabled)
                    if args.detect_nans and check_model_parameters(train_model, "train_model", global_step):
                        print(f"🚨 NaN gradients detected at step {global_step}! Skipping optimizer step...")
                        optimizer.zero_grad()  # Clear gradients and continue
                        continue
                    
                    accelerator.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
                    
                    # Check for NaN gradients after clipping (only if detection enabled)
                    if args.detect_nans and check_model_parameters(train_model, "train_model_after_clip", global_step):
                        print(f"🚨 NaN gradients after clipping at step {global_step}! Skipping optimizer step...")
                        optimizer.zero_grad()  # Clear gradients and continue
                        continue
                
                optimizer.step()
                
                # Check model parameters after optimizer step (only if detection enabled)
                if args.detect_nans and check_model_parameters(train_model, "train_model_after_step", global_step):
                    print(f"🚨 NaN model parameters after optimizer step {global_step}! This indicates a serious numerical issue...")
                    # Could implement model parameter restoration here if needed
                
                scheduler.step()
                optimizer.zero_grad()

            # Store loss value immediately and delete large tensors
            current_loss = pi3_loss.detach().item()
            # del pi3_loss, point_map_loss, camera_pose_loss, predictions
            
            # Aggressive memory cleanup after optimization
            if accelerator.sync_gradients:
                torch.cuda.empty_cache()
                
            # compute K matrix (moved outside accumulation to save memory)
            points = pseudo_gt["local_points"]
            masks = torch.sigmoid(pseudo_gt["conf"][..., 0]) > 0.1
            original_height, original_width = points.shape[-3:-1]
            aspect_ratio = original_width / original_height

            # use recover_focal_shift function from MoGe paper.
            # focal, shift = recover_focal_shift(points, masks)
            # fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
            # intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
            # intrinsics_pixels = denormalize_intrinsics(intrinsics, video_tensor_unaugmented.shape)

            # pseudo_gt["intrinsics"] = intrinsics_pixels
            pseudo_gt['images'] = video_tensor_unaugmented.permute(0, 1, 3, 4, 2)
            pseudo_gt['conf'] = torch.sigmoid(pseudo_gt['conf'])
            edge = depth_edge(pseudo_gt['local_points'][..., 2], rtol=0.03)
            pseudo_gt['conf'][edge] = 0.0
            
            if args.save_npz:
                # get copy and detach
                scene_rgb = video_tensor_unaugmented.cpu().numpy()[0]
                conf = pseudo_gt['conf'].copy()
                # remove last dim
                conf = conf[..., 0]

                depths = pseudo_gt['local_points'][..., 2].copy()
                intrinsics = pseudo_gt['intrinsics'].copy() if 'intrinsics' in pseudo_gt else None

                data_npz_load = {}
                # repat the first cam pose n times
                cam_poses = pseudo_gt['camera_poses'].copy()
                repeated_poses = np.tile(cam_poses[0], (20, 1, 1))

                data_npz_load["extrinsics"] = repeated_poses
                data_npz_load["intrinsics"] = intrinsics
                # depth_save = points_map[:,2,...]
                depths[conf<0.5] = 0
                data_npz_load["depths"] = depths
                data_npz_load["video"] = scene_rgb
                # data_npz_load["unc_metric"] = conf
                # save the data_npz_load to a npz file
                output_file = f"output_{total_step}.npz"
                total_step += 1
                np.savez(output_file, **data_npz_load)
                print(f"Saved data to {output_file}")

            if args.save_depths and 'local_points' in predictions and global_step % 200 == 0:
                # Convert tensors to numpy
                for key in pseudo_gt.keys():
                    if isinstance(pseudo_gt[key], torch.Tensor):
                        pseudo_gt[key] = pseudo_gt[key].cpu().numpy().squeeze(0)  # remove batch dimension

                for key in predictions.keys():
                    if isinstance(predictions[key], torch.Tensor):
                        predictions[key] = predictions[key].clone().detach().cpu().numpy().squeeze(0)  # remove batch dimension

                import matplotlib.pyplot as plt
                import imageio
                
                # === DEPTH VISUALIZATION ===
                local_points = predictions['local_points']  # shape (T, H, W, 3)
                depth_maps = local_points[..., 2]  # shape (T, H, W)
                depth_images_for_wandb = []
                
                for t in range(depth_maps.shape[0]):
                    depth = depth_maps[t]
                    vmin, vmax = np.percentile(depth, 2), np.percentile(depth, 98)
                    norm_depth = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
                    # Apply viridis colormap
                    colored = plt.get_cmap('viridis')(norm_depth)[:, :, :3]  # shape (H, W, 3), drop alpha
                    colored_uint8 = (colored * 255).astype(np.uint8)
                    
                    # Save to disk
                    imageio.imwrite(f"depth_frame_{t}_viridis.png", colored_uint8)
                    print(f"Saved depth map for frame {t} to depth_frame_{t}_viridis.png")
                    
                    # Prepare for WandB (convert to PIL Image for WandB)
                    if args.use_wandb:
                        from PIL import Image as PILImage
                        pil_image = PILImage.fromarray(colored_uint8)
                        depth_images_for_wandb.append(wandb.Image(pil_image, caption=f"Depth Frame {t}"))
                
                # === CONFIDENCE VISUALIZATION ===
                confidence_images_for_wandb = []
                if 'conf' in predictions:
                    conf_maps = predictions['conf']  # shape (T, H, W, 1) or (T, H, W)
                    if conf_maps.ndim == 4:  # (T, H, W, 1)
                        conf_maps = conf_maps.squeeze(-1)  # (T, H, W)
                    
                    for t in range(conf_maps.shape[0]):
                        conf = conf_maps[t]  # (H, W)
                        # Apply sigmoid if values are not in [0,1] range
                        if conf.min() < 0 or conf.max() > 1:
                            conf = 1 / (1 + np.exp(-conf))  # sigmoid
                        
                        # Apply hot colormap for confidence (red=high confidence, blue=low confidence)
                        colored = plt.get_cmap('hot')(conf)[:, :, :3]  # shape (H, W, 3), drop alpha
                        colored_uint8 = (colored * 255).astype(np.uint8)
                        
                        # Save to disk
                        imageio.imwrite(f"confidence_frame_{t}_hot.png", colored_uint8)
                        print(f"Saved confidence map for frame {t} to confidence_frame_{t}_hot.png")
                        
                        # Prepare for WandB
                        if args.use_wandb:
                            from PIL import Image as PILImage
                            pil_image = PILImage.fromarray(colored_uint8)
                            confidence_images_for_wandb.append(wandb.Image(pil_image, caption=f"Confidence Frame {t}"))
                
                # === CAMERA POSE VISUALIZATION ===
                if 'camera_poses' in predictions:
                    camera_poses = predictions['camera_poses']  # shape (T, 4, 4)
                    
                    # Extract translation vectors (position)
                    positions = camera_poses[:, :3, 3]  # (T, 3) - x, y, z positions
                    
                    # Create 3D trajectory plot
                    fig = plt.figure(figsize=(12, 4))
                    
                    # Plot 1: 3D trajectory
                    ax1 = fig.add_subplot(131, projection='3d')
                    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-o', markersize=3)
                    ax1.set_xlabel('X')
                    ax1.set_ylabel('Y')
                    ax1.set_zlabel('Z')
                    ax1.set_title('3D Camera Trajectory')
                    # Mark start and end
                    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='green', s=50, label='Start')
                    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=50, label='End')
                    ax1.legend()
                    
                    # Plot 2: XY trajectory (top view)
                    ax2 = fig.add_subplot(132)
                    ax2.plot(positions[:, 0], positions[:, 1], 'b-o', markersize=3)
                    ax2.scatter(positions[0, 0], positions[0, 1], c='green', s=50, label='Start')
                    ax2.scatter(positions[-1, 0], positions[-1, 1], c='red', s=50, label='End')
                    ax2.set_xlabel('X')
                    ax2.set_ylabel('Y')
                    ax2.set_title('Camera Trajectory (Top View)')
                    ax2.grid(True)
                    ax2.legend()
                    
                    # Plot 3: Position over time
                    ax3 = fig.add_subplot(133)
                    frames = np.arange(len(positions))
                    ax3.plot(frames, positions[:, 0], 'r-', label='X', linewidth=2)
                    ax3.plot(frames, positions[:, 1], 'g-', label='Y', linewidth=2)
                    ax3.plot(frames, positions[:, 2], 'b-', label='Z', linewidth=2)
                    ax3.set_xlabel('Frame')
                    ax3.set_ylabel('Position')
                    ax3.set_title('Camera Position Over Time')
                    ax3.grid(True)
                    ax3.legend()
                    
                    plt.tight_layout()
                    plt.savefig('camera_poses_latest.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"Saved camera poses plot to camera_poses_latest.png (step {global_step})")
                    
                    # Also save rotation matrices as heatmaps
                    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                    for t in range(min(6, len(camera_poses))):  # Show first 6 frames
                        row, col = t // 3, t % 3
                        rotation_matrix = camera_poses[t, :3, :3]  # (3, 3) rotation matrix
                        
                        im = axes[row, col].imshow(rotation_matrix, cmap='RdBu', vmin=-1, vmax=1)
                        axes[row, col].set_title(f'Rotation Matrix Frame {t}')
                        axes[row, col].set_xticks([0, 1, 2])
                        axes[row, col].set_yticks([0, 1, 2])
                        
                        # Add text annotations
                        for i in range(3):
                            for j in range(3):
                                axes[row, col].text(j, i, f'{rotation_matrix[i, j]:.2f}', 
                                                   ha='center', va='center', fontsize=8)
                    
                    plt.tight_layout()
                    plt.savefig('rotation_matrices_latest.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"Saved rotation matrices plot to rotation_matrices_latest.png (step {global_step})")
                
                # === LOG TO WANDB ===
                if args.use_wandb:
                    wandb_log_dict = {}
                    
                    # Log depth images
                    if depth_images_for_wandb:
                        wandb_log_dict["visualizations/depth_maps"] = depth_images_for_wandb
                        print(f"🚀 Logged {len(depth_images_for_wandb)} depth maps to WandB")
                    
                    # Log confidence images  
                    if confidence_images_for_wandb:
                        wandb_log_dict["visualizations/confidence_maps"] = confidence_images_for_wandb
                        print(f"🚀 Logged {len(confidence_images_for_wandb)} confidence maps to WandB")
                    
                    # Log camera pose plots (if they exist)
                    if 'camera_poses' in predictions:
                        # Log the trajectory plot
                        trajectory_plot_path = 'camera_poses_latest.png'
                        if os.path.exists(trajectory_plot_path):
                            wandb_log_dict["visualizations/camera_trajectory"] = wandb.Image(trajectory_plot_path, caption=f"Camera Trajectory Step {global_step}")
                        
                        # Log the rotation matrices plot  
                        rotation_plot_path = 'rotation_matrices_latest.png'
                        if os.path.exists(rotation_plot_path):
                            wandb_log_dict["visualizations/rotation_matrices"] = wandb.Image(rotation_plot_path, caption=f"Rotation Matrices Step {global_step}")
                    
                    # Send all visualizations to WandB
                    if wandb_log_dict:
                        wandb.log(wandb_log_dict, step=global_step)

            epoch_loss += current_loss
            running_loss += current_loss
            loss_history.append(current_loss)
                
            # Logging
            if global_step % args.log_freq == 0 and accelerator.is_main_process:
                current_lr = scheduler.get_last_lr()[0]
                
                # TensorBoard logging
                writer.add_scalar("Loss/Train", pi3_loss.item(), global_step)
                writer.add_scalar("Learning_Rate", current_lr, global_step)
                
                # Weights & Biases logging
                if args.use_wandb:
                    log_dict = {
                        "train/total_loss": pi3_loss.item(),
                        "train/point_map_loss": point_map_loss.item(),
                        "train/camera_pose_loss": camera_pose_loss.item(),
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch,
                        "train/best_loss": best_loss,
                        "train/step": global_step
                    }
                    # Add warmup phase indicator if using warmup
                    if warmup_steps > 0:
                        log_dict["train/warmup_phase"] = 1.0 if global_step < warmup_steps else 0.0
                    wandb.log(log_dict, step=global_step)
                
                postfix_dict = {
                    'loss': f'{pi3_loss.item():.6f}',
                    'lr': f'{current_lr:.2e}',
                    'best': f'{best_loss:.6f}',
                    'val_best': f'{best_val_loss:.6f}' if val_dataloader else 'N/A'
                }
                # Add warmup indicator if we're in warmup phase
                if warmup_steps > 0 and global_step < warmup_steps:
                    postfix_dict['warmup'] = f'{global_step}/{warmup_steps}'
                progress_bar.set_postfix(postfix_dict)
            
            # Validation check
            if (val_dataloader is not None and 
                global_step % args.val_freq == 0 and 
                global_step > 0 and 
                accelerator.is_main_process):
                
                print(f"\n🔍 Running validation at step {global_step}...")
                val_metrics = run_validation(
                    train_model, frozen_model, val_dataloader, args, accelerator, dtype
                )
                
                val_loss_history.append(val_metrics['val_loss'])
                
                # Log validation metrics
                writer.add_scalar("Loss/Validation", val_metrics['val_loss'], global_step)
                writer.add_scalar("Loss/Val_Point", val_metrics['val_point_loss'], global_step)
                writer.add_scalar("Loss/Val_Camera", val_metrics['val_camera_loss'], global_step)
                
                # Log unweighted validation losses for tracking actual performance
                writer.add_scalar("Loss/Val_Unweighted_L1_Points", val_metrics['val_unweighted_l1_points'], global_step)
                writer.add_scalar("Loss/Val_Unweighted_Pose", val_metrics['val_unweighted_pose_loss'], global_step)
                
                if args.use_wandb:
                    wandb.log({
                        "val/total_loss": val_metrics['val_loss'],
                        "val/point_map_loss": val_metrics['val_point_loss'],
                        "val/camera_pose_loss": val_metrics['val_camera_loss'],
                        # Unweighted validation losses for tracking actual performance
                        "val/unweighted_l1_points": val_metrics['val_unweighted_l1_points'],
                        "val/unweighted_pose_loss": val_metrics['val_unweighted_pose_loss'],
                        "val/num_samples": val_metrics['num_samples'],
                        "val/step": global_step
                    }, step=global_step)
                
                print(f"📊 Validation Results:")
                print(f"   Total Loss (weighted): {val_metrics['val_loss']:.6f}")
                print(f"   Point Loss (weighted): {val_metrics['val_point_loss']:.6f}")
                print(f"   Camera Loss (weighted): {val_metrics['val_camera_loss']:.6f}")
                print(f"   L1 Points (unweighted): {val_metrics['val_unweighted_l1_points']:.6f}")
                print(f"   Pose Loss (unweighted): {val_metrics['val_unweighted_pose_loss']:.6f}")
                print(f"   Samples: {val_metrics['num_samples']}")
                
                # Early stopping check
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    steps_without_improvement = 0
                    
                    # Save best validation model
                    best_val_checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': accelerator.unwrap_model(train_model).state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'val_metrics': val_metrics,
                        'loss_history': loss_history,
                        'val_loss_history': val_loss_history,
                        'args': vars(args)
                    }
                    
                    best_val_model_path = os.path.join(args.checkpoint_dir, 'best_val_model.pt')
                    torch.save(best_val_checkpoint, best_val_model_path)
                    print(f"💾 New best validation model saved! Val Loss: {best_val_loss:.6f}")
                    
                    if args.use_wandb:
                        wandb.log({
                            "val/best_model_saved": True,
                            "val/new_best_loss": best_val_loss
                        }, step=global_step)
                else:
                    steps_without_improvement += 1
                    print(f"⚠️  No validation improvement. Steps without improvement: {steps_without_improvement}/{args.early_stopping_patience}")
                
                # Early stopping
                if args.early_stopping_patience > 0 and steps_without_improvement >= args.early_stopping_patience:
                    print(f"🛑 Early stopping triggered after {steps_without_improvement} validation checks without improvement.")
                    if args.use_wandb:
                        wandb.log({
                            "training/early_stopped": True,
                            "training/final_step": global_step,
                            "training/final_val_loss": val_metrics['val_loss']
                        }, step=global_step)
                    # break
            
            # Model saving based on best loss
            if global_step % args.save_freq == 0 and global_step != 0 and accelerator.is_main_process:
                # Calculate average loss over the last save_freq steps
                recent_loss = running_loss / args.save_freq
                
                # Check if this is the best loss so far
                if recent_loss < best_loss:
                    best_loss = recent_loss
                    
                    # Save the best model
                    checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': accelerator.unwrap_model(train_model).state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_loss': best_loss,
                        'best_val_loss': best_val_loss,
                        'loss_history': loss_history,
                        'val_loss_history': val_loss_history,
                        'steps_without_improvement': steps_without_improvement,
                        'args': vars(args)
                    }
                    
                    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                    torch.save(checkpoint, best_model_path)
                    print(f"💾 New best model saved! Loss: {best_loss:.6f} at step {global_step}")
                    
                    # Log best model to wandb
                    if args.use_wandb:
                        wandb.log({
                            "train/best_model_saved": True,
                            "train/new_best_loss": best_loss
                        }, step=global_step)
                
                # Always save a recent checkpoint
                recent_checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': accelerator.unwrap_model(train_model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'current_loss': recent_loss,
                    'best_loss': best_loss,
                    'best_val_loss': best_val_loss,
                    'loss_history': loss_history,
                    'val_loss_history': val_loss_history,
                    'steps_without_improvement': steps_without_improvement,
                    'args': vars(args)
                }
                
                recent_model_path = os.path.join(args.checkpoint_dir, f'checkpoint_step_{global_step}.pt')
                torch.save(recent_checkpoint, recent_model_path)
                
                # Keep only the last 3 checkpoints to save disk space
                checkpoint_files = sorted([f for f in os.listdir(args.checkpoint_dir) if f.startswith('checkpoint_step_')])
                if len(checkpoint_files) > 3:
                    for old_checkpoint in checkpoint_files[:-3]:
                        os.remove(os.path.join(args.checkpoint_dir, old_checkpoint))
                
                # Reset running loss
                running_loss = 0.0
            
            # Optional debug outputs (much less frequent)
            # if global_step % 1000 == 0 and accelerator.is_main_process:
            #     save_debug_outputs(pseudo_gt, video_tensor_unaugmented, global_step, args)
            global_step += 1
            torch.cuda.empty_cache()

        # End of epoch logging
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        if accelerator.is_main_process:
            print(f"📊 Epoch {epoch+1} completed - Avg Loss: {avg_epoch_loss:.6f}")
            if val_dataloader is not None:
                print(f"   Best Val Loss so far: {best_val_loss:.6f}")
                print(f"   Steps without improvement: {steps_without_improvement}")
            
            # Log epoch metrics
            if args.use_wandb:
                epoch_log = {
                    "epoch/avg_loss": avg_epoch_loss,
                    "epoch/number": epoch + 1,
                    "epoch/best_loss_so_far": best_loss
                }
                if val_dataloader is not None:
                    epoch_log.update({
                        "epoch/best_val_loss_so_far": best_val_loss,
                        "epoch/steps_without_improvement": steps_without_improvement
                    })
                wandb.log(epoch_log, step=global_step)
                
        # Break out of epoch loop if early stopping was triggered
        if (args.early_stopping_patience > 0 and 
            steps_without_improvement >= args.early_stopping_patience):
            # break
            pass

    # Training complete - cleanup
    writer.close()
    
    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()
        print("📊 Weights & Biases run finished!")
    
    # Final summary
    if accelerator.is_main_process:
        print("🎉 Training complete!")
        print(f"📊 Best training loss achieved: {best_loss:.6f}")
        if val_dataloader is not None:
            print(f"📊 Best validation loss achieved: {best_val_loss:.6f}")
            print(f"💾 Best validation model saved at: {os.path.join(args.checkpoint_dir, 'best_val_model.pt')}")
            if steps_without_improvement >= args.early_stopping_patience:
                print(f"🛑 Training stopped early due to no validation improvement for {steps_without_improvement} checks")
        print(f"💾 Best training model saved at: {os.path.join(args.checkpoint_dir, 'best_model.pt')}")
        print(f"📈 Total training steps: {global_step}")
        
        # Save final training summary
        summary = {
            'final_loss': avg_epoch_loss,
            'best_loss': best_loss,
            'best_val_loss': best_val_loss,
            'total_steps': global_step,
            'loss_history': loss_history,
            'val_loss_history': val_loss_history,
            'steps_without_improvement': steps_without_improvement,
            'early_stopped': steps_without_improvement >= args.early_stopping_patience,
            'training_args': vars(args)
        }
        summary_path = os.path.join(args.checkpoint_dir, 'training_summary.json')
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📋 Training summary saved at: {summary_path}")

if __name__ == "__main__":
    main()
