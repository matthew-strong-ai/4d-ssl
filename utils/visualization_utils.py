#!/usr/bin/env python3
"""
Visualization utilities for training and model analysis.

This module contains functions for visualizing training data, model outputs,
and analysis results like dynamic object detection.
"""

import os
import numpy as np
import torch
import wandb
from PIL import Image as PILImage


def save_batch_images_to_png(X_all, step, cfg, max_batches=3, max_frames_per_batch=6):
    """
    Save images from X_all tensor to PNG files.
    
    Args:
        X_all: Tensor of shape (B, T, C, H, W) containing image batch
        step: Training step number for filename
        cfg: Configuration object with logging settings
        max_batches: Maximum number of batches to save (to limit disk usage)
        max_frames_per_batch: Maximum frames per batch to save
    """
    # Check if image saving is enabled
    if not cfg.LOGGING.SAVE_IMAGES:
        return
    
    # Only save images for the first few steps to avoid filling disk
    if step > cfg.LOGGING.SAVE_IMAGES_STEPS:
        return
    
    output_dir = cfg.LOGGING.SAVE_IMAGES_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    B, T, C, H, W = X_all.shape
    print(f"🖼️ Saving images from step {step}: {B} batches, {T} frames each")
    
    # ImageNet normalization values for denormalization (if needed)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Limit number of batches and frames to save
    num_batches_to_save = min(B, max_batches)
    num_frames_to_save = min(T, max_frames_per_batch)
    
    for b in range(num_batches_to_save):
        for t in range(num_frames_to_save):
            try:
                # Get single image: (C, H, W)
                img_tensor = X_all[b, t]  # Shape: (C, H, W)
                
                # Convert to numpy and transpose to (H, W, C)
                img_array = img_tensor.cpu().numpy().transpose(1, 2, 0)
                
                # Check if image is normalized (values in [0,1] or [-1,1] range)
                if img_array.min() >= 0 and img_array.max() <= 1:
                    # Assume it's in [0,1] range, convert to [0,255]
                    img_array = (img_array * 255).astype(np.uint8)
                elif img_array.min() >= -1 and img_array.max() <= 1:
                    # Assume it's normalized [-1,1], convert to [0,255]
                    img_array = ((img_array + 1) * 127.5).astype(np.uint8)
                else:
                    # Try ImageNet denormalization
                    img_array = img_array * std + mean
                    img_array = np.clip(img_array, 0, 1)
                    img_array = (img_array * 255).astype(np.uint8)
                
                # Convert to PIL Image and save
                pil_image = PILImage.fromarray(img_array)
                
                # Create filename with step, batch, and frame info
                filename = f"step_{step:04d}_batch_{b:02d}_frame_{t:02d}.png"
                save_path = os.path.join(output_dir, filename)
                pil_image.save(save_path)
                
            except Exception as e:
                print(f"❌ Error saving image step {step}, batch {b}, frame {t}: {e}")
                continue
    
    print(f"✅ Saved {num_batches_to_save}x{num_frames_to_save} images to {output_dir}")


def visualize_dynamic_objects(rgb_frames, gsam2_results, dynamic_analysis, point_maps, step, run):
    """
    Visualize dynamic objects across frames with movement tracking.
    
    Args:
        rgb_frames: List of RGB frame arrays
        gsam2_results: Results from GSAM2 process_frames 
        dynamic_analysis: Analysis results for each object
        point_maps: 3D point maps [T, H, W, 3]
        step: Training step number
        run: WandB run object for logging
        
    Returns:
        str: Filename of saved visualization or None if failed
    """
    try:
        import matplotlib.pyplot as plt

        print(f"🎨 Visualizing dynamic objects: {len(rgb_frames)} frames, {len(dynamic_analysis)} objects")
        
        T = len(rgb_frames)
        num_cols = min(T, 6)
        
        # Create figure
        fig, axes = plt.subplots(2, num_cols, figsize=(4*num_cols, 8))
        if num_cols == 1:
            axes = axes.reshape(2, 1)
        
        # Colors for objects
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
        
        # Process each frame
        for t in range(num_cols):
            # Get axes
            ax_rgb = axes[0, t] if num_cols > 1 else axes[0, 0]
            ax_depth = axes[1, t] if num_cols > 1 else axes[1, 0]
            
            # Show RGB frame
            ax_rgb.imshow(rgb_frames[t])
            ax_rgb.set_title(f'Frame {t} - Objects', fontsize=10)
            ax_rgb.axis('off')
            
            # Show depth map
            depth_map = point_maps[t, :, :, 2]
            ax_depth.imshow(depth_map, cmap='plasma')
            ax_depth.set_title(f'Frame {t} - Depth', fontsize=10)
            ax_depth.axis('off')
            
            # Add object overlays
            for obj_id, analysis in dynamic_analysis.items():
                if obj_id in gsam2_results['masks'][t]:
                    mask = gsam2_results['masks'][t][obj_id]
                    color_idx = (obj_id - 1) % len(colors)
                    color = colors[color_idx]
                    
                    # Get centroid
                    coords = np.where(mask)
                    if len(coords) == 2:
                        y_coords, x_coords = coords
                    elif len(coords) == 3:
                        _, y_coords, x_coords = coords
                    else:
                        continue
                        
                    if len(y_coords) > 0:
                        centroid_y = np.mean(y_coords)
                        centroid_x = np.mean(x_coords)
                        
                        # Draw marker with distinct icons for dynamic vs static
                        if analysis['is_dynamic']:
                            # Dynamic objects: triangle pointing right (arrow/movement)
                            marker = '>'
                            size = 14
                            edge_width = 2
                        else:
                            # Static objects: hexagon (stop sign shape)
                            marker = 'H'
                            size = 10
                            edge_width = 1
                        
                        ax_rgb.plot(centroid_x, centroid_y, marker, color=color, 
                                   markersize=size, markeredgecolor='white', markeredgewidth=edge_width)
                        ax_depth.plot(centroid_x, centroid_y, marker, color=color, 
                                     markersize=size, markeredgecolor='white', markeredgewidth=edge_width)
                        
                        # Add label
                        label = f"#{obj_id} {'DYN' if analysis['is_dynamic'] else 'STAT'}"
                        ax_rgb.text(centroid_x + 10, centroid_y, label, color='white', 
                                   fontsize=8, fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        plt.tight_layout()
        
        # Save
        viz_filename = f"dynamic_objects_step_{step}.png"
        plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
        print(f"🎨 Saved dynamic objects visualization: {viz_filename}")
        
        # Upload to wandb
        if run is not None:
            try:
                dynamic_count = sum(1 for a in dynamic_analysis.values() if a['is_dynamic'])
                static_count = len(dynamic_analysis) - dynamic_count
                
                wandb_image = wandb.Image(viz_filename, 
                                        caption=f"Step {step}: {dynamic_count} dynamic, {static_count} static objects")
                run.log({
                    "dynamic_objects_visualization": wandb_image,
                    "num_dynamic_objects": dynamic_count,
                    "num_static_objects": static_count,
                    "total_tracked_objects": len(dynamic_analysis)
                }, step=step)
                print(f"📊 Uploaded to wandb: {dynamic_count} dynamic, {static_count} static objects")
            except Exception as e:
                print(f"⚠️ Failed to upload to wandb: {e}")
        
        plt.close()
        return viz_filename
        
    except Exception as e:
        print(f"❌ Error in dynamic objects visualization: {e}")
        import traceback
        traceback.print_exc()
        return None


def tensor_to_image(tensor):
    """
    Convert a tensor to a displayable image array.
    
    Args:
        tensor: Tensor of shape (C, H, W) or (H, W, C)
        
    Returns:
        numpy array: Image array in (H, W, C) format, values in [0, 255]
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    # Handle different tensor shapes
    if tensor.ndim == 3:
        if tensor.shape[0] in [1, 3]:  # (C, H, W)
            tensor = tensor.transpose(1, 2, 0)
        # else assume (H, W, C)
    elif tensor.ndim == 2:
        # Grayscale image
        tensor = np.expand_dims(tensor, axis=-1)
        tensor = np.repeat(tensor, 3, axis=-1)  # Convert to RGB
    
    # Normalize to [0, 255]
    if tensor.min() >= 0 and tensor.max() <= 1:
        tensor = (tensor * 255).astype(np.uint8)
    elif tensor.min() >= -1 and tensor.max() <= 1:
        tensor = ((tensor + 1) * 127.5).astype(np.uint8)
    else:
        # Assume ImageNet normalized
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        tensor = tensor * std + mean
        tensor = np.clip(tensor, 0, 1)
        tensor = (tensor * 255).astype(np.uint8)
    
    return tensor


def create_prediction_grid(rgb_frames, predictions, step, save_dir="visualizations"):
    """
    Create a grid visualization of RGB frames and model predictions.
    
    Args:
        rgb_frames: List of RGB frame tensors
        predictions: Model predictions dict with 'local_points', 'conf', etc.
        step: Training step number
        save_dir: Directory to save visualizations
        
    Returns:
        str: Path to saved visualization
    """
    try:
        import matplotlib.pyplot as plt
        
        os.makedirs(save_dir, exist_ok=True)
        
        num_frames = len(rgb_frames)
        fig, axes = plt.subplots(3, num_frames, figsize=(4*num_frames, 12))
        
        if num_frames == 1:
            axes = axes.reshape(3, 1)
        
        for t in range(num_frames):
            # RGB frame
            rgb_img = tensor_to_image(rgb_frames[t])
            axes[0, t].imshow(rgb_img)
            axes[0, t].set_title(f'Frame {t} - RGB')
            axes[0, t].axis('off')
            
            # Depth prediction
            if 'local_points' in predictions:
                depth = predictions['local_points'][0, t, :, :, 2]  # Z component
                axes[1, t].imshow(depth.cpu().numpy(), cmap='viridis')
                axes[1, t].set_title(f'Frame {t} - Depth')
                axes[1, t].axis('off')
            
            # Confidence
            if 'conf' in predictions:
                conf = predictions['conf'][0, t, :, :, 0]  # First channel
                axes[2, t].imshow(conf.cpu().numpy(), cmap='hot')
                axes[2, t].set_title(f'Frame {t} - Confidence')
                axes[2, t].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"predictions_step_{step}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"🎨 Saved prediction grid: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"❌ Error creating prediction grid: {e}")
        return None


def visualize_motion_maps(motion_maps, step, save_dir="motion_visualizations"):
    """
    Visualize 3D motion maps and save as PNG files.
    
    Args:
        motion_maps: List of motion maps [T-1, H, W, 3] with 3D displacement vectors
        step: Training step number
        save_dir: Directory to save visualizations
        
    Returns:
        list: Paths to saved visualization files
    """
    try:
        import matplotlib.pyplot as plt
        
        os.makedirs(save_dir, exist_ok=True)
        saved_files = []
        
        print(f"🎨 Visualizing {len(motion_maps)} motion maps for step {step}")
        
        for t, motion_map in enumerate(motion_maps):
            H, W, _ = motion_map.shape
            
            # Compute motion magnitude for overall visualization
            motion_magnitude = np.linalg.norm(motion_map, axis=-1)
            
            # Compute adaptive color ranges based on actual data
            x_range = np.max(np.abs(motion_map[:, :, 0]))
            y_range = np.max(np.abs(motion_map[:, :, 1]))
            z_range = np.max(np.abs(motion_map[:, :, 2]))
            mag_range = np.max(motion_magnitude)
            
            # Use 95th percentile to avoid outliers affecting visualization
            x_range = np.percentile(np.abs(motion_map[:, :, 0]), 95)
            y_range = np.percentile(np.abs(motion_map[:, :, 1]), 95)
            z_range = np.percentile(np.abs(motion_map[:, :, 2]), 95)
            mag_range = np.percentile(motion_magnitude, 95)
            
            # Create figure with subplots for each component and magnitude
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # X displacement
            im1 = axes[0, 0].imshow(motion_map[:, :, 0], cmap='RdBu_r', vmin=-x_range, vmax=x_range)
            axes[0, 0].set_title(f'Frame {t}→{t+1}: X Displacement (m)\nRange: ±{x_range:.3f}')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
            
            # Y displacement
            im2 = axes[0, 1].imshow(motion_map[:, :, 1], cmap='RdBu_r', vmin=-y_range, vmax=y_range)
            axes[0, 1].set_title(f'Frame {t}→{t+1}: Y Displacement (m)\nRange: ±{y_range:.3f}')
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
            
            # Z displacement
            im3 = axes[1, 0].imshow(motion_map[:, :, 2], cmap='RdBu_r', vmin=-z_range, vmax=z_range)
            axes[1, 0].set_title(f'Frame {t}→{t+1}: Z Displacement (m)\nRange: ±{z_range:.3f}')
            axes[1, 0].axis('off')
            plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
            
            # Motion magnitude
            im4 = axes[1, 1].imshow(motion_magnitude, cmap='hot', vmin=0, vmax=mag_range)
            axes[1, 1].set_title(f'Frame {t}→{t+1}: Motion Magnitude (m)\nMax: {mag_range:.3f}')
            axes[1, 1].axis('off')
            plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
            
            # Add motion statistics as text
            avg_mag = np.mean(motion_magnitude)
            max_mag = np.max(motion_magnitude)
            fig.suptitle(f'3D Motion Map - Step {step}, Transition {t}→{t+1}\n'
                        f'Avg: {avg_mag:.4f}m, Max: {max_mag:.4f}m', fontsize=14)
            
            plt.tight_layout()
            
            # Save the visualization
            filename = f"motion_map_step_{step}_transition_{t}to{t+1}.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            saved_files.append(save_path)
            print(f"💾 Saved motion map: {save_path}")
        
        print(f"✅ Saved {len(saved_files)} motion map visualizations")
        return saved_files
        
    except Exception as e:
        print(f"❌ Error visualizing motion maps: {e}")
        import traceback
        traceback.print_exc()
        return []


def visualize_motion_flow_overlay(rgb_frames, motion_maps, step, save_dir="motion_overlays", downsample=8):
    """
    Visualize motion maps as flow vectors overlaid on RGB frames.
    
    Args:
        rgb_frames: List of RGB frame arrays
        motion_maps: List of motion maps [T-1, H, W, 3]
        step: Training step number
        save_dir: Directory to save visualizations
        downsample: Factor to downsample arrows for clarity
        
    Returns:
        list: Paths to saved overlay files
    """
    try:
        import matplotlib.pyplot as plt
        
        os.makedirs(save_dir, exist_ok=True)
        saved_files = []
        
        print(f"🎨 Creating motion flow overlays for {len(motion_maps)} transitions")
        
        for t, motion_map in enumerate(motion_maps):
            if t >= len(rgb_frames) - 1:
                break
                
            H, W, _ = motion_map.shape
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Show RGB frame
            ax.imshow(rgb_frames[t])
            ax.set_title(f'Motion Flow Overlay - Step {step}, Frame {t}→{t+1}')
            ax.axis('off')
            
            # Create downsampled grid for arrows
            y_coords, x_coords = np.meshgrid(
                np.arange(0, H, downsample),
                np.arange(0, W, downsample),
                indexing='ij'
            )
            
            # Get motion vectors at grid points (project 3D to 2D for visualization)
            u = motion_map[y_coords, x_coords, 0]  # X displacement
            v = motion_map[y_coords, x_coords, 1]  # Y displacement
            w = motion_map[y_coords, x_coords, 2]  # Z displacement
            
            # Compute 3D magnitude for color coding
            magnitude = np.sqrt(u**2 + v**2 + w**2)
            
            # Scale arrows for visibility (convert from meters to pixels approximately)
            scale_factor = 100  # Adjust this to make arrows visible
            u_scaled = u * scale_factor
            v_scaled = v * scale_factor
            
            # Plot motion vectors as arrows
            quiver = ax.quiver(x_coords, y_coords, u_scaled, v_scaled, 
                             magnitude, scale=1, scale_units='xy', angles='xy',
                             cmap='hot', alpha=0.8, width=0.003)
            
            # Add colorbar
            cbar = plt.colorbar(quiver, ax=ax, shrink=0.8)
            cbar.set_label('Motion Magnitude (m)', rotation=270, labelpad=15)
            
            plt.tight_layout()
            
            # Save the overlay
            filename = f"motion_overlay_step_{step}_transition_{t}to{t+1}.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            saved_files.append(save_path)
            print(f"💾 Saved motion overlay: {save_path}")
        
        print(f"✅ Saved {len(saved_files)} motion overlay visualizations")
        return saved_files
        
    except Exception as e:
        print(f"❌ Error creating motion flow overlays: {e}")
        import traceback
        traceback.print_exc()
        return []


def log_wandb_images(images_dict, step, run, prefix=""):
    """
    Log multiple images to Weights & Biases.
    
    Args:
        images_dict: Dict of {name: image_path} or {name: image_array}
        step: Training step number
        run: WandB run object
        prefix: Optional prefix for log keys
    """
    if run is None:
        return
        
    try:
        log_dict = {}
        for name, image in images_dict.items():
            key = f"{prefix}_{name}" if prefix else name
            
            if isinstance(image, str):  # File path
                if os.path.exists(image):
                    log_dict[key] = wandb.Image(image)
            else:  # Array or tensor
                log_dict[key] = wandb.Image(tensor_to_image(image))
        
        if log_dict:
            run.log(log_dict, step=step)
            print(f"📊 Logged {len(log_dict)} images to wandb")
            
    except Exception as e:
        print(f"⚠️ Failed to log images to wandb: {e}")