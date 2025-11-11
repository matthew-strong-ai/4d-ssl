# Copyright (c) 2025 Matt Strong. Created for Self Supervised Learning from In the Wild Driving Videos

import argparse
import os
import cv2
import torch


from accelerate import Accelerator
from tqdm import tqdm
import torchvision.transforms as Tr
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb
import torchvision.transforms.functional as TF

# Configuration management
from yacs.config import CfgNode as CN
from config.defaults import get_cfg_defaults, update_config

import utils3d

# all imports for spatracker.
from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image, get_default_transforms, preprocess_numpy_image
from SpaTrackerV2.ssl_image_dataset import SequenceLearningDataset
from simple_s3_dataset import S3Dataset

import sys

from cotracker.utils.visualizer import Visualizer
from vision.gsam2_class import GSAM2

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pi3"))

from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3, AutonomyPi3, AutoregressivePi3

# Import distilled ViT
from distilled_vit import create_distilled_vit, DistillationLoss, visualize_distilled_pca

# import pi3 losses
from losses import Pi3Losses, NormalLosses, PointCloudLosses, normalize_pred_gt
from s3_utils import download_from_s3_uri
from debug_utils import check_for_nans, check_model_parameters

# Import refactored utility modules
from utils.youtube_s3_dataset import YouTubeS3Dataset
from utils.s3_utils import save_state_dict_to_s3, upload_file_to_s3
from utils.augmentation_utils import apply_random_augmentations
from utils.visualization_utils import save_batch_images_to_png, visualize_dynamic_objects, visualize_motion_maps, visualize_motion_flow_overlay
from utils.motion_generation import generate_motion_gt_from_flow
from utils.validation_utils import run_validation, align_prediction_shapes, denormalize_intrinsics
from utils.analysis_utils import analyze_object_dynamics
from utils.model_factory import create_model, validate_model_config, get_model_info
from utils.raft_utils import initialize_raft_model, generate_optical_flow_gt


def generate_optical_flow_gt(frames, raft_model, InputPadder, device='cuda', visualize=False):
    """
    Generate optical flow ground truth using pre-loaded RAFT model.
    
    Args:
        frames: Input frames tensor [B, T, C, H, W] or numpy array
        raft_model: Pre-loaded RAFT model
        InputPadder: RAFT's InputPadder class
        device: Device to run computation on
        visualize: Whether to show flow visualization during generation
        
    Returns:
        flow_gt: Optical flow ground truth [B, T-1, H, W, 2] with (dx, dy) flow vectors
    """

    if raft_model is None:
        return None
    try:
        # Convert frames to correct format
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames).to(device)
        else:
            frames = frames.to(device)
            
        # Ensure frames are in [B, T, C, H, W] format
        if frames.dim() == 4:  # [T, C, H, W]
            frames = frames.unsqueeze(0)  # [1, T, C, H, W]
            
        B, T, C, H, W = frames.shape
        flow_list = []
        visualize = False  # Disable visualization for now
        
        # Compute flow between consecutive frames
        with torch.no_grad():
            for t in range(T - 1):
                # Get consecutive frames
                frame1 = frames[:, t]  # [B, C, H, W]
                frame2 = frames[:, t + 1]  # [B, C, H, W]

                # Pad images for RAFT (divisible by 8)
                padder = InputPadder(frame1.shape)
                frame1, frame2 = padder.pad(frame1, frame2)

                frame1 = (frame1 * 255.0)
                frame2 = (frame2 * 255.0)
                
                # Compute flow
                _, flow_up = raft_model(frame1, frame2, iters=20, test_mode=True)

                # Unpad flow
                flow = padder.unpad(flow_up)  # [B, 2, H, W]
                
                # Keep flow in pixel units (no normalization)
                # flow = flow / 200.0  # Removed normalization
                
                # Optional: Clamp extreme values to [-1, 1] range
                # flow = torch.clamp(flow, -1.0, 1.0)
                
                # Transpose to [B, H, W, 2]
                flow = flow.permute(0, 2, 3, 1)
                flow_list.append(flow)
                
                # Add visualization
                if visualize:  # Only visualize first 3 transitions
                    import sys
                    import os
                    sys.path.append('/home/matthew_strong/Desktop/autonomy-wild/RAFT/core/utils')
                    from flow_viz import flow_to_image
                    import imageio
                    
                    # Convert to numpy for visualization
                    flow_np = flow[0].cpu().numpy()  # Take first batch item [H, W, 2]
                    
                    # Use RAFT's flow visualization
                    flow_img = flow_to_image(flow_np)  # Returns [H, W, 3] RGB image

                    # Save visualization
                    filename = f"flow_gt_gen_frame_{t}_to_{t+1}.png"
                    imageio.imwrite(filename, flow_img)
                
        # Stack flows: [B, T-1, H, W, 2]
        flow_gt = torch.stack(flow_list, dim=1)
        
        return flow_gt
        
    except Exception as e:
        print(f"Error generating optical flow GT: {e}")
        return None


def generate_motion_gt_from_flow(flow_gt, point_maps, cfg, rgb_frames=None, object_segformer=None, dynamic_classes=None, visualize=True):
    """
    Generate motion ground truth from optical flow and 3D point maps.
    Tracks pixels across ALL timesteps to detect cumulative motion.
    
    Args:
        flow_gt: Optical flow ground truth [B, T-1, H, W, 2] (pixel units)
        point_maps: 3D point predictions [B, T, H, W, 3] (world coordinates)
        cfg: Configuration object
        rgb_frames: RGB frames [T, 3, H, W] for segmentation (optional)
        object_segformer: SegFormer model for object detection (optional)
        dynamic_classes: List of class IDs to consider for motion (e.g. [1,2] for vehicle,person)
        
    Returns:
        motion_gt: Binary motion masks [B, T, H, W, 1] where 1 = dynamic
    """
    B, T_minus_1, H, W, _ = flow_gt.shape
    T = T_minus_1 + 1  # Full temporal dimension
    device = flow_gt.device

    
    # Initialize cumulative motion tracking
    # Track the path length (sum of all frame-to-frame motions)
    cumulative_path_length = torch.zeros(B, H, W, device=device)
    motion_gt = torch.zeros(B, T, H, W, 1, device=device)
    
    # Create initial pixel coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    
    # Run segmentation if provided
    object_masks = None
    if rgb_frames is not None and object_segformer is not None and dynamic_classes is not None:
        import numpy as np
        
        with torch.no_grad():
            # Convert torch tensors to numpy arrays for the segmentation model
            rgb_frames_np = []
            for t in range(T):
                # Convert from [3, H, W] to [H, W, 3] numpy array
                frame_np = rgb_frames[t].cpu().numpy()  # [3, H, W]
                frame_np = np.transpose(frame_np, (1, 2, 0))  # [H, W, 3]
                frame_np = (frame_np * 255).astype(np.uint8)  # Convert to 0-255 range
                rgb_frames_np.append(frame_np)
            
            # Run segmentation using process_frames method
            seg_results = object_segformer.process_frames(rgb_frames_np, text_prompt="car. vehicle. person. road sign. traffic light.", verbose=False)
            
            # Get composite masks with class IDs
            seg_predictions = seg_results['composite_masks']  # List of [H, W] numpy arrays
            
            # Convert to torch tensor and create mask for dynamic object classes
            object_masks = torch.zeros(T, H, W, device=device, dtype=torch.bool)
            for t in range(T):
                seg_t = torch.from_numpy(seg_predictions[t]).to(device)
                for class_id in dynamic_classes:
                    object_masks[t] = object_masks[t] | (seg_t == class_id)
            
            print(f"🚗 Segmentation complete. Dynamic pixels per frame:")
            for t_seg in range(T):
                dynamic_count = object_masks[t_seg].sum().item()
                total_count = object_masks[t_seg].numel()
                print(f"   Frame {t_seg}: {dynamic_count}/{total_count} ({100*dynamic_count/total_count:.1f}%)")
    
    # Track previous 3D positions for computing motion
    previous_points_3d = None

    # Process each timestep
    for t in range(T):
        if t == 0:
            # For first frame, use original pixel coordinates
            tracked_x = x_coords[None, :, :].expand(B, -1, -1)
            tracked_y = y_coords[None, :, :].expand(B, -1, -1)
        else:
            # For subsequent frames, use flow per frame directly from previous frame
            flow_t = flow_gt[:, t-1]  # [B, H, W, 2]
            flow_pixels = flow_t  # Already in pixel units

            # Apply flow from previous frame position
            tracked_x = tracked_x + flow_pixels[..., 0]
            tracked_y = tracked_y + flow_pixels[..., 1]
        
        # Sample current 3D positions at tracked locations
        grid_x = 2.0 * torch.clamp(tracked_x, 0, W-1) / (W - 1) - 1.0
        grid_y = 2.0 * torch.clamp(tracked_y, 0, H-1) / (H - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]
        
        # Get current frame's 3D points
        current_points = point_maps[:, t].permute(0, 3, 1, 2)  # [B, 3, H, W]
        
        # Sample 3D positions at tracked pixel locations
        tracked_points_3d = torch.nn.functional.grid_sample(
            current_points,
            grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='zeros'
        ).permute(0, 2, 3, 1)  # [B, H, W, 3]
        
        if t > 0 and previous_points_3d is not None:
            # Compute frame-to-frame 3D motion
            frame_motion_3d = tracked_points_3d - previous_points_3d  # [B, H, W, 3]
            
            # Compute frame-to-frame motion magnitude
            frame_motion_magnitude = torch.sqrt(
                frame_motion_3d[..., 0]**2 + 
                frame_motion_3d[..., 1]**2 + 
                frame_motion_3d[..., 2]**2
            )  # [B, H, W]
            
            # Add to cumulative path length
            cumulative_path_length = cumulative_path_length + frame_motion_magnitude
        
        # Update previous points for next iteration
        previous_points_3d = tracked_points_3d
        
        # Store raw cumulative path length for median computation
        if t == T - 1:  # On the last frame, compute final motion masks
            # Pixel-by-pixel classification based on individual motion
            motion_threshold = cfg.POINT_MOTION.MOTION_THRESHOLD
            
            # Classify each pixel as dynamic if its cumulative motion exceeds threshold
            is_dynamic = (cumulative_path_length > motion_threshold).float()  # [B, H, W]
            
            if object_masks is not None:
                # Create mask of valid object classes to consider
                valid_object_mask = torch.zeros_like(object_masks, dtype=torch.bool)
                for obj_id in dynamic_classes:
                    valid_object_mask = valid_object_mask | (object_masks == obj_id)
                
                # Apply pixel-wise motion classification only to valid object pixels
                for t_update in range(T):
                    # Get mask of valid objects for this frame
                    obj_mask_t = valid_object_mask[t_update].float().unsqueeze(0)  # [1, H, W]
                    
                    # Pixel is dynamic if: (1) it moved more than threshold AND (2) it belongs to a valid object class
                    motion_gt[:, t_update, :, :, 0] = is_dynamic * obj_mask_t
                
                # Print statistics per object class
                print(f"\n📊 Pixel-wise motion statistics:")
                for obj_id in dynamic_classes:
                    obj_mask_all_t = object_masks == obj_id  # [T, H, W]
                    obj_union_mask = obj_mask_all_t.any(dim=0)  # [H, W]
                    
                    if obj_union_mask.sum() > 0:
                        # Extract motion values for this object
                        obj_motion_values = cumulative_path_length[0][obj_union_mask]
                        
                        # Count dynamic pixels for this object
                        obj_dynamic_pixels = (obj_motion_values > motion_threshold).sum().item()
                        total_obj_pixels = obj_union_mask.sum().item()
                        dynamic_ratio = obj_dynamic_pixels / total_obj_pixels if total_obj_pixels > 0 else 0
                        
                        # Compute statistics
                        median_motion = torch.median(obj_motion_values).item()
                        mean_motion = obj_motion_values.mean().item()
                        max_motion = obj_motion_values.max().item()
                        
                        print(f"   Object class {obj_id}:")
                        print(f"     - Dynamic pixels: {obj_dynamic_pixels}/{total_obj_pixels} ({dynamic_ratio:.1%})")
                        print(f"     - Motion stats: median={median_motion:.3f}m, mean={mean_motion:.3f}m, max={max_motion:.3f}m")
                
                # Summary statistics
                total_motion_pixels = motion_gt.sum().item()
                print(f"\n   SUMMARY: Total dynamic pixels = {total_motion_pixels} (pixel-wise classification)")
                    
            else:
                # No object masks - apply pixel-wise threshold to all pixels
                for t_update in range(T):
                    motion_gt[:, t_update, :, :, 0] = is_dynamic
        
        # Visualization
        if visualize and t < 3 and t > 0:  # Visualize first few frames with motion
            import matplotlib.pyplot as plt
            import numpy as np
            import imageio
            import sys
            sys.path.append('/home/matthew_strong/Desktop/autonomy-wild/RAFT/core/utils')
            from flow_viz import flow_to_image
            
            viz_batch = 0  # Visualize first batch item
            
            # Create a figure with multiple subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Flow visualization
            flow_viz = flow_gt[viz_batch, t-1].cpu().numpy()  # [H, W, 2]
            flow_img = flow_to_image(flow_viz)
            axes[0, 0].imshow(flow_img)
            axes[0, 0].set_title(f'Optical Flow {t-1}→{t}')
            axes[0, 0].axis('off')
            
            # 2. Current 3D points (depth visualization)
            points_current = point_maps[viz_batch, t].cpu().numpy()  # [H, W, 3]
            depth_current = points_current[..., 2]  # Z coordinate as depth
            im_depth = axes[0, 1].imshow(depth_current, cmap='viridis')
            axes[0, 1].set_title(f'Depth at Frame {t}')
            axes[0, 1].axis('off')
            plt.colorbar(im_depth, ax=axes[0, 1], fraction=0.046)
            
            # 3. Tracked points visualization (show displacement)
            tracked_points = tracked_points_3d[viz_batch].cpu().numpy()  # [H, W, 3]
            tracked_depth = tracked_points[..., 2]
            im_tracked = axes[0, 2].imshow(tracked_depth, cmap='viridis')
            axes[0, 2].set_title(f'Tracked Points Depth at Frame {t}')
            axes[0, 2].axis('off')
            plt.colorbar(im_tracked, ax=axes[0, 2], fraction=0.046)
            
            # 4. Frame-to-frame motion magnitude
            if t > 0:
                motion_mag = frame_motion_magnitude[viz_batch].cpu().numpy()  # [H, W]
                im_motion = axes[1, 0].imshow(motion_mag, cmap='hot')
                axes[1, 0].set_title(f'Motion Magnitude {t-1}→{t} (meters)')
                axes[1, 0].axis('off')
                plt.colorbar(im_motion, ax=axes[1, 0], fraction=0.046)
            
            # 5. Cumulative path length
            cum_motion = cumulative_path_length[viz_batch].cpu().numpy()  # [H, W]
            im_cum = axes[1, 1].imshow(cum_motion, cmap='hot')
            axes[1, 1].set_title(f'Cumulative Motion up to Frame {t} (meters)')
            axes[1, 1].axis('off')
            plt.colorbar(im_cum, ax=axes[1, 1], fraction=0.046)
            
            # 6. Binary motion mask (show object-constrained motion only)
            if object_masks is not None:
                # Show object-constrained motion even during intermediate frames
                motion_threshold = cfg.POINT_MOTION.MOTION_THRESHOLD
                raw_motion_mask = (cumulative_path_length[viz_batch] > motion_threshold).cpu().numpy()
                
                # Apply object constraint to raw motion using dynamic_classes
                obj_mask_t = torch.zeros_like(object_masks[t], dtype=torch.bool)
                for class_id in dynamic_classes:
                    obj_mask_t = obj_mask_t | (object_masks[t] == class_id)
                obj_constrained_motion = raw_motion_mask * obj_mask_t.cpu().numpy()
                
                axes[1, 2].imshow(obj_constrained_motion, cmap='RdYlBu_r', vmin=0, vmax=1)
                class_names = ['road', 'vehicle', 'person', 'traffic light', 'traffic sign', 'sky', 'building/background']
                dynamic_class_names = [class_names[i] for i in dynamic_classes if i < len(class_names)]
                class_str = '+'.join(dynamic_class_names)
                if t < T - 1:
                    axes[1, 2].set_title(f'Object Motion at Frame {t} ({class_str} only)')
                else:
                    axes[1, 2].set_title(f'Final Object Motion Mask ({class_str})')
            else:
                # No object masks - show raw thresholded motion
                motion_threshold = cfg.POINT_MOTION.MOTION_THRESHOLD
                temp_motion_mask = (cumulative_path_length[viz_batch] > motion_threshold).cpu().numpy()
                axes[1, 2].imshow(temp_motion_mask, cmap='RdYlBu_r', vmin=0, vmax=1)
                axes[1, 2].set_title(f'Raw Motion Mask at Frame {t}')
            axes[1, 2].axis('off')
            
            # Add flow arrows overlaid on depth
            if t < 3:  # Only for first few frames
                # Subsample for arrow visualization
                step = 20
                y_sub, x_sub = np.mgrid[0:H:step, 0:W:step]
                flow_x_sub = flow_viz[::step, ::step, 0]
                flow_y_sub = flow_viz[::step, ::step, 1]
                
                # Create new subplot for flow arrows on depth
                plt.figure(figsize=(10, 8))
                plt.imshow(depth_current, cmap='viridis', alpha=0.8)
                plt.quiver(x_sub, y_sub, flow_x_sub, flow_y_sub, 
                          angles='xy', scale_units='xy', scale=1, 
                          color='red', width=0.003, headwidth=3, headlength=4)
                plt.title(f'Flow Arrows on Depth - Frame {t}')
                plt.colorbar(label='Depth (m)')
                plt.savefig(f'flow_on_depth_frame_{t}.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"💾 Saved flow_on_depth_frame_{t}.png")
                
                # Also visualize segmentation if available
                if object_masks is not None:
                    plt.figure(figsize=(20, 10))
                    
                    # Get RGB frame
                    rgb_frame_t = rgb_frames[t].cpu().numpy()  # [3, H, W]
                    rgb_frame_t = np.transpose(rgb_frame_t, (1, 2, 0))  # [H, W, 3]
                    rgb_frame_t = np.clip(rgb_frame_t, 0, 1)  # Ensure in [0,1] range
                    
                    # Show segmentation predictions
                    seg_pred_t = seg_predictions[t]
                    obj_mask_t = object_masks[t].cpu().numpy()
                    
                    # Define colors for each class (matching CityscapesAsGSAM2)
                    class_colors = np.array([
                        [128, 64, 128],   # 0: road - purple
                        [64, 0, 128],     # 1: vehicle - dark blue
                        [255, 128, 0],    # 2: person - orange
                        [255, 255, 0],    # 3: traffic light - yellow
                        [192, 128, 255],  # 4: traffic sign - light purple
                        [128, 128, 255],  # 5: sky - light blue
                        [128, 128, 128]   # 6: background - gray
                    ]) / 255.0  # Normalize to [0,1]
                    
                    # Create colored segmentation overlay
                    seg_colored = np.zeros((H, W, 3))
                    for class_id in range(7):
                        mask = seg_pred_t == class_id
                        seg_colored[mask] = class_colors[class_id]
                    
                    plt.subplot(2, 3, 1)
                    plt.imshow(rgb_frame_t)
                    plt.title(f'RGB Frame {t}')
                    plt.axis('off')
                    
                    plt.subplot(2, 3, 2)
                    plt.imshow(seg_pred_t, cmap='tab10')
                    plt.title(f'Segmentation Classes')
                    plt.colorbar(label='Class ID', ticks=range(7))
                    plt.axis('off')
                    
                    plt.subplot(2, 3, 3)
                    # RGB with segmentation overlay
                    alpha = 0.5
                    seg_overlay = rgb_frame_t * (1 - alpha) + seg_colored * alpha
                    plt.imshow(seg_overlay)
                    plt.title(f'RGB + Segmentation Overlay')
                    plt.axis('off')
                    
                    plt.subplot(2, 3, 4)
                    plt.imshow(obj_mask_t, cmap='gray')
                    class_names = ['road', 'vehicle', 'person', 'traffic light', 'traffic sign', 'sky', 'building/background']
                    dynamic_class_names_vis = [class_names[i] for i in dynamic_classes if i < len(class_names)]
                    class_str_vis = ', '.join(dynamic_class_names_vis)
                    plt.title(f'Dynamic Objects ({class_str_vis})')
                    plt.axis('off')
                    
                    plt.subplot(2, 3, 5)
                    # Overlay motion on object mask
                    overlay = np.zeros((H, W, 3))
                    # Get motion mask for visualization
                    if t < T - 1:
                        # Use temporary threshold for visualization, but constrain to object classes
                        motion_threshold = cfg.POINT_MOTION.MOTION_THRESHOLD
                        raw_vis_motion_mask = (cumulative_path_length[viz_batch] > motion_threshold).cpu().numpy()
                        # Apply object constraint to visualization
                        obj_mask_vis = torch.zeros_like(object_masks[t], dtype=torch.bool)
                        for class_id in dynamic_classes:
                            obj_mask_vis = obj_mask_vis | (object_masks[t] == class_id)
                        vis_motion_mask = raw_vis_motion_mask * obj_mask_vis.cpu().numpy()
                    else:
                        # Use final motion mask
                        vis_motion_mask = motion_gt[viz_batch, t, :, :, 0].cpu().numpy()
                    
                    overlay[..., 0] = vis_motion_mask  # Red channel for motion
                    overlay[..., 1] = obj_mask_t   # Green channel for objects
                    plt.imshow(overlay)
                    plt.title('Motion (red) on Objects (green)')
                    plt.axis('off')
                    
                    plt.subplot(2, 3, 6)
                    # RGB with motion overlay (only on objects)
                    motion_overlay = rgb_frame_t.copy()
                    motion_pixels = vis_motion_mask > 0.5
                    motion_overlay[motion_pixels, 0] = 1.0  # Red tint for motion
                    motion_overlay[motion_pixels, 1] *= 0.5  # Reduce green
                    motion_overlay[motion_pixels, 2] *= 0.5  # Reduce blue
                    plt.imshow(motion_overlay)
                    plt.title('RGB + Motion (on objects only)')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f'segmentation_viz_frame_{t}.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"💾 Saved segmentation_viz_frame_{t}.png")
            
            plt.suptitle(f'Motion Detection Visualization - Frame {t}', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'motion_detection_viz_frame_{t}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"🎨 Motion detection visualization saved: motion_detection_viz_frame_{t}.png")
            print(f"   Flow range: x=[{flow_viz[..., 0].min():.1f}, {flow_viz[..., 0].max():.1f}], y=[{flow_viz[..., 1].min():.1f}, {flow_viz[..., 1].max():.1f}] pixels")
            print(f"   Depth range: [{depth_current.min():.2f}, {depth_current.max():.2f}] meters")
            print(f"   Motion range: [{motion_mag.min():.3f}, {motion_mag.max():.3f}] meters")
            print(f"   Cumulative motion: [{cum_motion.min():.3f}, {cum_motion.max():.3f}] meters")
            # Get current motion mask for stats
            if t < T - 1:
                temp_motion_mask = (cumulative_path_length[viz_batch] > motion_threshold).cpu().numpy()
                print(f"   Dynamic pixels (temp): {temp_motion_mask.sum()} / {temp_motion_mask.size} ({100*temp_motion_mask.mean():.1f}%)")
            else:
                final_motion_mask = motion_gt[viz_batch, t, :, :, 0].cpu().numpy()
                print(f"   Dynamic pixels (final): {final_motion_mask.sum()} / {final_motion_mask.size} ({100*final_motion_mask.mean():.1f}%)")
    
    return motion_gt


def convert_mapanything_to_pi3_format(mapanything_output, B, T, H, W, cfg):
    """
    Convert MapAnything output to Pi3-compatible format for unified loss computation.
    
    Args:
        mapanything_output: Raw output from MapAnything model
        B, T, H, W: Batch size, time frames, height, width
        cfg: Configuration object
        
    Returns:
        dict: Pi3-compatible predictions dictionary
    """
    predictions = {}
    
    # Extract depth and confidence from MapAnything output
    # MapAnything typically outputs disparity/depth maps
    if 'depth' in mapanything_output or 'disparity' in mapanything_output:
        # Convert depth to local points format [B, T, H, W, 3]
        if 'depth' in mapanything_output:
            depth = mapanything_output['depth']  # [B*T, 1, H, W] or [B*T, H, W]
        else:
            # Convert disparity to depth
            disparity = mapanything_output['disparity']
            depth = 1.0 / (disparity + 1e-8)  # Basic disparity to depth conversion
            
        # Reshape depth to [B, T, H, W, 1]
        if depth.dim() == 3:  # [B*T, H, W]
            depth = depth.unsqueeze(-1)  # [B*T, H, W, 1]
        elif depth.dim() == 4 and depth.shape[1] == 1:  # [B*T, 1, H, W]
            depth = depth.permute(0, 2, 3, 1)  # [B*T, H, W, 1]
            
        depth = depth.reshape(B, T, H, W, 1)
        
        # Create xy coordinates for local points
        device = depth.device
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        xy = torch.stack([x, y], dim=-1).float()  # [H, W, 2]
        xy = xy.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)  # [B, T, H, W, 2]
        
        # Combine xy and depth to create local_points [B, T, H, W, 3]
        local_points = torch.cat([xy, depth], dim=-1)
        predictions['local_points'] = local_points
        
        # Generate dummy camera poses (identity matrices) for compatibility
        eye_matrix = torch.eye(4, device=device, dtype=local_points.dtype)
        camera_poses = eye_matrix.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        predictions['camera_poses'] = camera_poses
        
        # Transform local points to world coordinates using identity transform
        predictions['points'] = local_points  # Same as local points for identity transform
    
    # Generate confidence maps if not provided by MapAnything
    if 'confidence' in mapanything_output:
        conf = mapanything_output['confidence'].reshape(B, T, H, W, 1)
        predictions['conf'] = conf
    else:
        # Generate uniform confidence for compatibility
        conf = torch.ones(B, T, H, W, 1, device=depth.device, dtype=depth.dtype)
        predictions['conf'] = conf
    
    # Handle segmentation if MapAnything provides semantic output
    if cfg.MODEL.USE_SEGMENTATION_HEAD and 'semantics' in mapanything_output:
        semantics = mapanything_output['semantics']  # [B*T, num_classes, H, W]
        num_classes = semantics.shape[1]
        semantics = semantics.permute(0, 2, 3, 1)  # [B*T, H, W, num_classes]
        semantics = semantics.reshape(B, T, H, W, num_classes)
        predictions['segmentation'] = semantics
    
    # Handle motion if available (MapAnything may not have motion by default)
    if cfg.MODEL.USE_MOTION_HEAD:
        # Generate zero motion for compatibility (MapAnything doesn't predict motion by default)
        motion = torch.zeros(B, T, H, W, 3, device=depth.device, dtype=depth.dtype)
        predictions['motion'] = motion
    
    return predictions

# GroundingDINO imports (optional)
_GDINO_AVAILABLE = True
try:
    from groundingdino.util.inference import load_model as gdino_load_model
    from groundingdino.util.inference import load_image as gdino_load_image
    from groundingdino.util.inference import predict as gdino_predict
    from groundingdino.util.inference import annotate as gdino_annotate
    print("✅ GroundingDINO imports successful")
except ImportError as e:
    _GDINO_AVAILABLE = False
    print(f"⚠️ GroundingDINO not available: {e}")

#################################################################################################3

from rich import print
import random
import numpy as np

import datetime
import subprocess
from io import BytesIO
import boto3
import json

import os
import time

def train_model(train_config=None, experiment_tracker=None):
    """
    Main training function that uses YACS configuration.
    
    Args:
        train_config: Dictionary containing training configuration (from Ray Tune or direct call)
        experiment_tracker: Experiment tracker (if any)
    """
    # sample args
    # Get default config and load from config.yaml automatically
    cfg = get_cfg_defaults()
    cfg = update_config(cfg)  # This will automatically load config.yaml
    
    # If train_config is provided (e.g., from Ray Tune), update the config
    if train_config and isinstance(train_config, dict):
        cfg.defrost()
        # Map Ray Tune config to YACS config structure
        if 'learning_rate' in train_config:
            cfg.TRAINING.LEARNING_RATE = train_config['learning_rate']
        if 'num_epochs' in train_config:
            cfg.TRAINING.NUM_EPOCHS = train_config['num_epochs']
        if 'batch_size' in train_config:
            cfg.DATASET.BATCH_SIZE = train_config['batch_size']
        if 'future_frame_weight' in train_config:
            cfg.LOSS.FUTURE_FRAME_WEIGHT = train_config['future_frame_weight']
        if 'pc_loss_weight' in train_config:
            cfg.LOSS.PC_LOSS_WEIGHT = train_config['pc_loss_weight']
        if 'pose_loss_weight' in train_config:
            cfg.LOSS.POSE_LOSS_WEIGHT = train_config['pose_loss_weight']
        if 'conf_loss_weight' in train_config:
            cfg.LOSS.CONF_LOSS_WEIGHT = train_config['conf_loss_weight']
        # Add more mappings as needed
        cfg.freeze()
    
    # Initialize wandb
    if cfg.WANDB.USE_WANDB:
        wandb_kwargs = {
            "entity": "research-interns",
            "project": cfg.WANDB.PROJECT,
            "config": {
                "learning_rate": cfg.TRAINING.LEARNING_RATE,
                "batch_size": cfg.DATASET.BATCH_SIZE,
                "grad_accum_steps": cfg.TRAINING.GRAD_ACCUM_STEPS,
                "effective_batch_size": cfg.DATASET.BATCH_SIZE * cfg.TRAINING.GRAD_ACCUM_STEPS,
                "num_epochs": cfg.TRAINING.NUM_EPOCHS,
                "m_frames": cfg.MODEL.M,
                "n_frames": cfg.MODEL.N,
                "future_frame_weight": cfg.LOSS.FUTURE_FRAME_WEIGHT,
                "pc_loss_weight": cfg.LOSS.PC_LOSS_WEIGHT,
                "pose_loss_weight": cfg.LOSS.POSE_LOSS_WEIGHT,
                "conf_loss_weight": cfg.LOSS.CONF_LOSS_WEIGHT,
                "use_conf_weighted_points": cfg.LOSS.USE_CONF_WEIGHTED_POINTS,
                "conf_gamma": cfg.LOSS.CONF_GAMMA,
                "conf_alpha": cfg.LOSS.CONF_ALPHA,
                "max_grad_norm": cfg.TRAINING.MAX_GRAD_NORM,
                "architecture": cfg.MODEL.ARCHITECTURE,
                "optimizer": "Adam",
                "scheduler": "Warmup+CosineAnnealingLR",
                "warmup_steps": cfg.TRAINING.WARMUP_STEPS,
                "warmup_start_factor": cfg.TRAINING.WARMUP_START_FACTOR,
                "mixed_precision": "bf16",
                "val_split": cfg.DATASET.VAL_SPLIT,
                "val_freq": cfg.VALIDATION.VAL_FREQ,
                "val_samples": cfg.VALIDATION.VAL_SAMPLES,
                "early_stopping_patience": cfg.VALIDATION.EARLY_STOPPING_PATIENCE,
                "dataset_type": "S3" if cfg.DATASET.USE_S3 else "Local",
                "s3_bucket": cfg.DATASET.S3_BUCKET if cfg.DATASET.USE_S3 else None,
                "s3_preload_bytes": cfg.DATASET.S3_PRELOAD_BYTES if cfg.DATASET.USE_S3 else None,
                "save_images": cfg.LOGGING.SAVE_IMAGES,
                "save_images_steps": cfg.LOGGING.SAVE_IMAGES_STEPS,
            },
            "tags": ["pi3", "ssl", "cluster-training", "s3" if cfg.DATASET.USE_S3 else "local"]
        }
        
        # Add custom run name if specified
        if cfg.WANDB.RUN_NAME:
            wandb_kwargs["name"] = cfg.WANDB.RUN_NAME
            
        run = wandb.init(**wandb_kwargs)
        # Get the actual run name (either custom or auto-generated)
        actual_run_name = run.name if run else None
    else:
        run = None
        actual_run_name = None
    
    print("[training.train_model] Using configuration:")
    print(cfg)

    # Set NCCL environment variables to prevent cluster communication issues
    os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand
    os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable peer-to-peer communication
    print("🔧 Set NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 for cluster compatibility")

    # start training setup
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.TRAINING.GRAD_ACCUM_STEPS,
        mixed_precision='bf16',  # Use bfloat16 for better stability and performance
        kwargs_handlers=[ddp_kwargs]
    )
    
    # Override DDP error handling if distributed
    if accelerator.num_processes > 1:
        print(f"🔧 Running distributed training with {accelerator.num_processes} processes")
        print("🔍 DDP unused parameter debugging enabled (every 100 steps)")
    

    # Create dataset based on configuration
    if cfg.DATASET.get('USE_YOUTUBE', False):
        print(f"📺 Using YouTube S3 dataset from bucket: {cfg.DATASET.S3_BUCKET}")
        print(f"   Root prefix: {cfg.DATASET.get('YOUTUBE_ROOT_PREFIX', 'openDV-YouTube/full_images/')}")
        print(f"   Cache directory: {cfg.DATASET.get('YOUTUBE_CACHE_DIR', './youtube_cache')}")
        print(f"   Skip frames: {cfg.DATASET.get('YOUTUBE_SKIP_FRAMES', 300)}")
        print(f"   Min sequence length: {cfg.DATASET.get('YOUTUBE_MIN_SEQUENCE_LENGTH', 50)}")
        print(f"   Max workers: {cfg.DATASET.get('YOUTUBE_MAX_WORKERS', 8)}")

        # with s3, download s3://research-datasets/youtube_cache/youtube_dataset_df7b4701e6ade36698417531f6d163f2.pkl
        youtube_cache_path = 'youtube_cache/youtube_dataset_df7b4701e6ade36698417531f6d163f2.pkl'
        download_success = download_from_s3_uri(
            "s3://research-datasets/youtube_cache/youtube_dataset_df7b4701e6ade36698417531f6d163f2.pkl",
            youtube_cache_path,
            create_dirs=True,
            overwrite=False
        )
        
        # Create YouTube S3 dataset with optimizations
        full_dataset = YouTubeS3Dataset(
            bucket_name=cfg.DATASET.S3_BUCKET,
            root_prefix=cfg.DATASET.get('YOUTUBE_ROOT_PREFIX', 'openDV-YouTube/full_images/'),
            m=cfg.MODEL.M,
            n=cfg.MODEL.N,
            transform=None,
            region_name=cfg.DATASET.get('S3_REGION', 'us-phoenix-1'),
            cache_dir=cfg.DATASET.get('YOUTUBE_CACHE_DIR', './youtube_cache'),
            refresh_cache=cfg.DATASET.get('YOUTUBE_REFRESH_CACHE', False),
            min_sequence_length=cfg.DATASET.get('YOUTUBE_MIN_SEQUENCE_LENGTH', 50),
            skip_frames=cfg.DATASET.get('YOUTUBE_SKIP_FRAMES', 300),
            max_workers=cfg.DATASET.get('YOUTUBE_MAX_WORKERS', 8),
            verbose=True,
            frame_sampling_rate=cfg.DATASET.get('FRAME_SAMPLING_RATE', 1)  # 1=10Hz, 5=2Hz
        )
        
        print(f"✅ YouTube dataset loaded: {len(full_dataset):,} training samples")
        
        # Limit dataset size if requested
        if hasattr(cfg.DATASET, 'MAX_SAMPLES') and cfg.DATASET.MAX_SAMPLES > 0:
            original_size = len(full_dataset)
            # Create a subset using torch.utils.data.Subset
            from torch.utils.data import Subset
            import random
            
            # Create random subset - same samples across epochs, different per run
            indices = list(range(original_size))
            random.shuffle(indices)  # Random order, different each run
            indices = indices[:min(cfg.DATASET.MAX_SAMPLES, original_size)]
            
            full_dataset = Subset(full_dataset, indices)
            print(f"🎯 Limited dataset to {len(full_dataset):,} samples (from {original_size:,})")
            print(f"   🎲 Using random subset (fixed for this run, shuffled each epoch)")
        
    elif cfg.DATASET.USE_S3:
        print(f"🚀 Using S3 dataset from bucket: {cfg.DATASET.S3_BUCKET}")
        print(f"   Sequence prefixes: {cfg.DATASET.S3_SEQUENCE_PREFIXES}")
        print(f"   Image extension: {cfg.DATASET.S3_IMAGE_EXTENSION}")
        print(f"   AWS region: {cfg.DATASET.S3_REGION}")
        print(f"   Preload bytes: {cfg.DATASET.S3_PRELOAD_BYTES}")
        print(f"   ⏭️ Skipping first 300 frames per sequence, using frames 300+")
        
        # Create S3 dataset with new implementation
        full_dataset = S3Dataset(
            bucket_name=cfg.DATASET.S3_BUCKET,
            sequence_prefixes=cfg.DATASET.S3_SEQUENCE_PREFIXES,
            m=cfg.MODEL.M,
            n=cfg.MODEL.N,
            image_extension=cfg.DATASET.S3_IMAGE_EXTENSION,
            transform=get_default_transforms(),
            region_name=cfg.DATASET.S3_REGION,
            preload_bytes=cfg.DATASET.S3_PRELOAD_BYTES
        )
    else:
        print(f"📁 Using local dataset from: {cfg.DATASET.ROOT_DIR}")
        
        # Find all subdirectories in root_dir
        image_dirs = [os.path.join(cfg.DATASET.ROOT_DIR, d) for d in os.listdir(cfg.DATASET.ROOT_DIR)
                      if os.path.isdir(os.path.join(cfg.DATASET.ROOT_DIR, d))]
        print(f"Found {len(image_dirs)} subfolders:")
        for d in image_dirs:
            print(f"  {d}")

        # Create local dataset
        full_dataset = SequenceLearningDataset(
            image_dirs=image_dirs,
            m=cfg.MODEL.M,
            n=cfg.MODEL.N,
            transform=get_default_transforms())

    # Split dataset into train and validation
    if cfg.DATASET.VAL_SPLIT > 0:
        total_size = len(full_dataset)
        val_size = int(total_size * cfg.DATASET.VAL_SPLIT)
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
    if cfg.AUGMENTATION.USE_AUGMENTATIONS:
        print("🎨 Random augmentations enabled: color jittering, Gaussian blur, grayscale")
    else:
        print("🚫 Random augmentations disabled")


    # Smart DataLoader configuration optimized for multi-GPU training
    num_gpus = accelerator.num_processes if hasattr(accelerator, 'num_processes') else 1
    
    if cfg.DATASET.get('USE_YOUTUBE', False):
        # YouTube dataset optimizations: massive dataset with S3 backend
        num_workers = min(4, 4 * num_gpus)  # More workers for YouTube's large scale
        prefetch_factor = 2  # Moderate prefetching for S3 stability 
        persistent_workers = True
        pin_memory = True
        print(f"📺 YouTube dataset optimization: {num_workers} workers, prefetch_factor={prefetch_factor}")
        print(f"   💾 Using cached metadata ({len(full_dataset):,} samples)")
        
    elif cfg.DATASET.USE_S3:
        if cfg.DATASET.S3_PRELOAD_BYTES:
            # Maximum performance: bytes preloaded, scale workers with GPUs
            num_workers = min(8, 2 * num_gpus)  # 2 workers per GPU, max 8
            prefetch_factor = 16  # Aggressive prefetching for multi-GPU
            persistent_workers = True
            pin_memory = True
            print(f"🚀 Multi-GPU high-performance mode: {num_workers} workers, prefetch_factor={prefetch_factor} (S3 bytes preloaded)")
        else:
            # Balanced: scale workers with GPUs but keep reasonable for S3 credentials
            num_workers = 1
            prefetch_factor = 2  # Good prefetching for multi-GPU
            persistent_workers = True
            pin_memory = True
            print(f"⚖️ Multi-GPU balanced mode: {num_workers} workers, prefetch_factor={prefetch_factor} (S3 on-demand)")
    else:
        # Local dataset: scale with GPUs for file I/O
        num_workers = min(6, 2 * num_gpus)  # 2 workers per GPU, max 6
        prefetch_factor = 8
        persistent_workers = True
        pin_memory = True
        print(f"💾 Multi-GPU local dataset mode: {num_workers} workers, prefetch_factor={prefetch_factor}")
    
    print(f"   🖥️ Detected GPUs: {num_gpus}")
    print(f"   📊 DataLoader config: workers={num_workers}, prefetch={prefetch_factor}, persistent={persistent_workers}")
    print(f"   ⚡ Memory optimizations: pin_memory={pin_memory}")
    
    # Calculate effective throughput
    effective_workers = num_workers * num_gpus if num_gpus > 1 else num_workers
    print(f"   🚀 Effective worker processes across all GPUs: {effective_workers}")


    # num_workers = 8
    # prefetch_factor = 4
    # import ipdb; ipdb.set_trace()
    # Create optimized dataloaders with enhanced configuration
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,  # Ensure consistent batch sizes for training stability
        timeout=600 if num_workers > 0 else 0  # 5 min timeout for S3 downloads
    )
    
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.DATASET.BATCH_SIZE,
            shuffle=False,  # Don't shuffle validation
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,  # Keep all validation samples
            timeout=600 if num_workers > 0 else 0
        )
        
        print(f"📊 DataLoaders created:")
        print(f"   🎯 Train: {len(train_dataloader)} batches of size {cfg.DATASET.BATCH_SIZE}")
        print(f"   ✅ Val: {len(val_dataloader)} batches of size {cfg.DATASET.BATCH_SIZE}")
    else:
        print(f"📊 Train DataLoader created: {len(train_dataloader)} batches of size {cfg.DATASET.BATCH_SIZE}")


    # download sam2 checkpoint
    sam2_local_path = "Grounded-SAM-2/models/sam2.1_hiera_large.pt"
    print(f"📥 Downloading SAM2 checkpoint...")
    download_success = download_from_s3_uri(
        "s3://research-datasets/sam2.1_hiera_large.pt",
        sam2_local_path,
        create_dirs=True,
        overwrite=False
    )
    if not download_success:
        raise RuntimeError("Failed to download SAM2 checkpoint")

    # Download DINOv3 checkpoint from S3
    dinov3_local_path = "dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    print(f"📥 Downloading DINOv3 checkpoint...")
    download_success = download_from_s3_uri(
        "s3://research-datasets/dinov3_matt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        dinov3_local_path,
        create_dirs=True,
        overwrite=False
    )
    if not download_success:
        raise RuntimeError("Failed to download DINOv3 checkpoint")

    print("Initializing and loading Pi3 model...")
    frozen_model = Pi3.from_pretrained("yyfz233/Pi3")
    frozen_model = frozen_model.to(accelerator.device)
    frozen_model.requires_grad_(False)  # freeze parameters

    # Validate model configuration
    validate_model_config(cfg)
    
    # Print model info
    model_info = get_model_info(cfg)
    print(f"🏗️ Creating {cfg.MODEL.ARCHITECTURE} model with configuration:")
    for key, value in model_info.items():
        print(f"   - {key}: {value}")
    
    # Create model using factory
    train_model = create_model(cfg, dinov3_local_path)
    
    # Create distilled ViT if enabled
    distilled_vit = None
    distillation_loss_fn = None
    
    if cfg.MODEL.USE_DISTILLED_VIT:
        print(f"\n🔬 Creating Distilled ViT with configuration:")
        print(f"   - Teacher: {cfg.MODEL.ENCODER_NAME}")
        print(f"   - Student embed_dim: {cfg.MODEL.DISTILLED_VIT.EMBED_DIM}")
        print(f"   - Student depth: {cfg.MODEL.DISTILLED_VIT.DEPTH}")
        print(f"   - Student heads: {cfg.MODEL.DISTILLED_VIT.NUM_HEADS}")
        print(f"   - Distill tokens: {cfg.MODEL.DISTILLED_VIT.DISTILL_TOKENS}")
        

        distilled_vit = create_distilled_vit(
            teacher_model_name=cfg.MODEL.ENCODER_NAME,
            embed_dim=cfg.MODEL.DISTILLED_VIT.EMBED_DIM,
            depth=cfg.MODEL.DISTILLED_VIT.DEPTH,
            num_heads=cfg.MODEL.DISTILLED_VIT.NUM_HEADS,
            distill_tokens=cfg.MODEL.DISTILLED_VIT.DISTILL_TOKENS
        )
        
        distillation_loss_fn = DistillationLoss(
            distill_tokens=cfg.MODEL.DISTILLED_VIT.DISTILL_TOKENS,
            loss_weights={
                'point_features': cfg.LOSS.DISTILLATION_POINT_FEATURES_WEIGHT,
                'camera_features': cfg.LOSS.DISTILLATION_CAMERA_FEATURES_WEIGHT,
                'autonomy_features': cfg.LOSS.DISTILLATION_AUTONOMY_FEATURES_WEIGHT
            },
            temperature=cfg.MODEL.DISTILLED_VIT.TEMPERATURE,
            use_cosine_similarity=cfg.MODEL.DISTILLED_VIT.USE_COSINE_SIMILARITY
        )
        
        print(f"✅ Distilled ViT created with {sum(p.numel() for p in distilled_vit.parameters())/1e6:.1f}M parameters")
    
    # Verify gradient flow setup (especially if using freeze_decoders)
    if hasattr(train_model, 'verify_gradient_flow'):
        print("\n=== Verifying Gradient Flow Configuration ===")
        train_model.verify_gradient_flow()
        
        # Note: Using transformer output directly, no token predictor
        if hasattr(train_model, 'autoregressive_transformer'):
            print("\n--- Autoregressive Transformer Details ---")
            ar_transformer = train_model.autoregressive_transformer
            total_params = sum(p.numel() for p in ar_transformer.parameters())
            trainable_params = sum(p.numel() for p in ar_transformer.parameters() if p.requires_grad)
            print(f"AR Transformer: {trainable_params:,}/{total_params:,} trainable params")
            print("  Using transformer output directly (no token predictor)")
        
        print("==========================================\n")

    # Initialize segmentation model based on config
    if cfg.MODEL.SEGMENTATION_MODEL == "gsam2":
        gsam2 = GSAM2()
        print(f"✅ Initialized GSAM2 for segmentation ({cfg.MODEL.SEGMENTATION_NUM_CLASSES} classes)")
    elif cfg.MODEL.SEGMENTATION_MODEL in ["segformer", "deeplabv3"]:
        from utils.cityscapes_segmentation import CityscapesAsGSAM2
        gsam2 = CityscapesAsGSAM2(model_type=cfg.MODEL.SEGMENTATION_MODEL)
        print(f"✅ Initialized Cityscapes {cfg.MODEL.SEGMENTATION_MODEL} for segmentation ({cfg.MODEL.SEGMENTATION_NUM_CLASSES} classes)")
    else:
        raise ValueError(f"Unknown segmentation model: {cfg.MODEL.SEGMENTATION_MODEL}")

    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").cuda()
    cotracker.eval()
    print("Successfully created training model: CoTracker3")

    # Load SegFormer for object segmentation (independent of training segmentation)
    from utils.cityscapes_segmentation import CityscapesAsGSAM2
    object_segformer = CityscapesAsGSAM2(model_type="segformer")
    print("✅ Loaded SegFormer for object segmentation (7 Cityscapes classes)")
    
    # Cityscapes 7-class system:
    # 0: road, 1: vehicle, 2: person, 3: traffic light, 4: traffic sign, 5: sky, 6: bg/building
    # Dynamic object classes we care about:
    DYNAMIC_CLASSES = [1, 2]  # vehicle, person
    print(f"   Will track motion for: vehicle (1), person (2)")

    from pi3.models.segformer.model import EncoderDecoder
    segformer = EncoderDecoder()
    segformer.load_state_dict(torch.load('segformer.b0.512x512.ade.160k.pth', map_location=torch.device('cpu'), weights_only=False)['state_dict'])
    segformer = segformer.to(accelerator.device)

    # Load pre-trained weights (only for Pi3-based models that use frozen model)
    if cfg.MODEL.ARCHITECTURE.lower() in ["pi3", "autoregressivepi3"]:
        # Load encoder and rope (only load encoder weights for dinov2)
        if cfg.MODEL.ENCODER_NAME == "dinov2":
            print("📥 Loading DINOv2 encoder weights from frozen model...")
            train_model.encoder.load_state_dict(frozen_model.encoder.state_dict())
        else:
            print("📥 Using DINOv3 encoder with custom checkpoint (skipping frozen model weights)")
        
        train_model.rope.load_state_dict(frozen_model.rope.state_dict())
        
        # Freeze encoder parameters during cluster training
        print("🔒 Freezing encoder parameters...")
        for param in train_model.encoder.parameters():
            param.requires_grad = False
        print(f"✅ Froze {sum(p.numel() for p in train_model.encoder.parameters())} encoder parameters")

        # Load decoders (if they exist)
        if hasattr(train_model, 'decoder'):
            train_model.decoder.load_state_dict(frozen_model.decoder.state_dict())

        # commenting this out for now
        if hasattr(train_model, 'point_decoder'):
            train_model.point_decoder.load_state_dict(frozen_model.point_decoder.state_dict())
        if hasattr(train_model, 'conf_decoder'):
            train_model.conf_decoder.load_state_dict(frozen_model.conf_decoder.state_dict())
        if hasattr(train_model, 'camera_decoder'):
            train_model.camera_decoder.load_state_dict(frozen_model.camera_decoder.state_dict())
        
        # Load register token (if it exists)
        if hasattr(train_model, 'register_token'):
            train_model.register_token.data.copy_(frozen_model.register_token.data)
        
        # Load heads (if they exist) - Pi3 uses LinearPts3d, AutonomyPi3 uses FutureLinearPts3d
        if hasattr(train_model, 'point_head'):
            frozen_point_dict = frozen_model.point_head.state_dict()
            train_point_dict = train_model.point_head.state_dict()
            matched_point_dict = {
                k: v for k, v in frozen_point_dict.items()
                if k in train_point_dict and v.shape == train_point_dict[k].shape
            }
            train_point_dict.update(matched_point_dict)
            train_model.point_head.load_state_dict(train_point_dict)
        
        if hasattr(train_model, 'conf_head'):
            frozen_conf_dict = frozen_model.conf_head.state_dict()
            train_conf_dict = train_model.conf_head.state_dict()
            matched_conf_dict = {
                k: v for k, v in frozen_conf_dict.items()
                if k in train_conf_dict and v.shape == train_conf_dict[k].shape
            }
            train_conf_dict.update(matched_conf_dict)
            train_model.conf_head.load_state_dict(train_conf_dict)

        # Load camera head (if it exists)
        if hasattr(train_model, 'camera_head'):
            frozen_camera_dict = frozen_model.camera_head.state_dict()
            train_camera_dict = train_model.camera_head.state_dict()
            matched_camera_dict = {
                k: v for k, v in frozen_camera_dict.items()
                if k in train_camera_dict and v.shape == train_camera_dict[k].shape
            }
            train_camera_dict.update(matched_camera_dict)
            train_model.camera_head.load_state_dict(train_camera_dict)

        # Copy point head weights to motion head for better initialization (if motion head exists)
        if hasattr(train_model, 'motion_head') and train_model.motion_head is not None:
            print("🔄 Copying point head weights to motion head for better initialization...")
            # Both point_head and motion_head have same architecture (FutureLinearPts3d with output_dim=3)
            # so we can copy all weights directly
            # train_model.motion_head.load_state_dict(train_model.point_head.state_dict())
            print("✅ Successfully copied point head weights to motion head")
        
            # other decoders for motion and segmentation
            train_model.motion_decoder.load_state_dict(frozen_model.point_decoder.state_dict())

        if hasattr(train_model, 'segmentation_decoder') and train_model.segmentation_decoder is not None:
            train_model.segmentation_decoder.load_state_dict(frozen_model.point_decoder.state_dict())
        
        # Load image normalization buffers (if they exist)
        if hasattr(train_model, 'image_mean'):
            train_model.image_mean.data.copy_(frozen_model.image_mean.data)
        if hasattr(train_model, 'image_std'):
            train_model.image_std.data.copy_(frozen_model.image_std.data)
        # Additional Pi3 projection layer loading (if available)
        try:
            # if autoregressive
            if True:
                train_model.point_head.proj.load_state_dict(frozen_model.point_head.proj.state_dict())
                train_model.conf_head.proj.load_state_dict(frozen_model.conf_head.proj.state_dict())
                train_model.motion_head.proj.load_state_dict(frozen_model.point_head.proj.state_dict())
            train_model.point_head.current_proj.load_state_dict(frozen_model.point_head.proj.state_dict())
            train_model.conf_head.current_proj.load_state_dict(frozen_model.conf_head.proj.state_dict())
            train_model.motion_head.current_proj.load_state_dict(frozen_model.point_head.proj.state_dict())
        except:
            print("Shape issue with projection layers, no worry!")
    else:
        # MapAnything or other models - no frozen model loading needed
        print(f"✅ {cfg.MODEL.ARCHITECTURE} model created without frozen model initialization")

    # Define optimizer for train_model
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=cfg.TRAINING.LEARNING_RATE)
    total_steps = len(train_dataloader) * cfg.TRAINING.NUM_EPOCHS
    warmup_steps = min(cfg.TRAINING.WARMUP_STEPS, total_steps // 10)  # Cap warmup at 10% of total steps
    cosine_steps = total_steps - warmup_steps
    
    if warmup_steps > 0:
        # Create warmup scheduler (linear increase from start_factor to 1.0)
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=cfg.TRAINING.WARMUP_START_FACTOR, 
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
    distilled_vit_optimizer = None
    distilled_vit_scheduler = None
    
    if distilled_vit is not None:
        # Create separate optimizer for distilled ViT
        distilled_vit_optimizer = torch.optim.AdamW(
            distilled_vit.parameters(),
            lr=cfg.TRAINING.LEARNING_RATE,
            weight_decay=0.1
        )
        distilled_vit_scheduler = CosineAnnealingLR(distilled_vit_optimizer, T_max=total_steps)
        
        if val_dataloader is not None:
            train_model, distilled_vit, optimizer, distilled_vit_optimizer, scheduler, distilled_vit_scheduler, train_dataloader, val_dataloader = accelerator.prepare(
                train_model,
                distilled_vit,
                optimizer,
                distilled_vit_optimizer,
                scheduler,
                distilled_vit_scheduler,
                train_dataloader,
                val_dataloader
            )
        else:
            train_model, distilled_vit, optimizer, distilled_vit_optimizer, scheduler, distilled_vit_scheduler, train_dataloader = accelerator.prepare(
                train_model,
                distilled_vit,
                optimizer,
                distilled_vit_optimizer,
                scheduler,
                distilled_vit_scheduler,
                train_dataloader
            )
    else:
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
    
    # Initialize RAFT model once if flow head is enabled
    raft_model = None
    InputPadder = None
    if cfg.MODEL.USE_FLOW_HEAD:
        print(f"[GPU {accelerator.process_index}] Initializing RAFT model for optical flow...")
        raft_model, InputPadder = initialize_raft_model(device=device)
        if raft_model is None:
            print(f"[GPU {accelerator.process_index}] WARNING: RAFT initialization failed. Flow ground truth will be unavailable.")
    
    # Synchronize all processes after model preparation
    print(f"[GPU {accelerator.process_index}] Waiting for all GPUs after model preparation...")
    accelerator.wait_for_everyone()
    print(f"[GPU {accelerator.process_index}] All GPUs synchronized after model preparation")

    # Create checkpoint directory
    if accelerator.is_main_process:
        os.makedirs(cfg.OUTPUT.CHECKPOINT_DIR, exist_ok=True)

    # TensorBoard SummaryWriter
    if accelerator.is_main_process:
        writer = SummaryWriter("runs/pi3_cluster")

    # Training loop
    global_step = 0
    total_step = 0
    best_loss = float('inf')
    best_val_loss = float('inf')
    running_loss = 0.0
    loss_history = []
    val_loss_history = []
    steps_without_improvement = 0
    
    # Data structure to store pseudo_gt results for first 100 steps
    pseudo_gt_storage = {
        'step': [],
        'point_maps': [],
        'local_point_maps': [],
        'camera_poses': [],
        'confidence': [],
        'segmentation': [] if cfg.MODEL.USE_SEGMENTATION_HEAD else None,
        'motion': [] if cfg.MODEL.USE_MOTION_HEAD else None,
        'images_original': [],  # Original unaugmented images
        'gsam2_composite_masks': [],  # GSAM2 computed masks (all objects)
        # 'gsam2_motion_maps': [],  # Computed 3D motion maps from CoTracker
        'gsam2_labels': []  # GSAM2 detected object labels
    }
    pseudo_gt_save_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, 'pseudo_gt_first_100_steps.pt')


    print(f"[GPU {accelerator.process_index}] Waiting for all GPUs before training loop...")
    accelerator.wait_for_everyone()
    print(f"[GPU {accelerator.process_index}] All GPUs synchronized, starting training!")

    # Model warmup: Do a few forward passes without gradients to stabilize model states across GPUs
    print("🔥 Starting model warmup phase...")
    warmup_steps = 5  # Number of warmup forward passes
    train_model.eval()  # Set to eval mode for warmup
    frozen_model.eval()
    warmup_iterator = iter(train_dataloader)

    for warmup_step in range(warmup_steps):
        try:
            batch = next(warmup_iterator)
            print(f"   Warmup step {warmup_step + 1}/{warmup_steps} on process {accelerator.process_index}")
            
            with torch.no_grad():
                X = batch[0]
                y = batch[1] 
                X_all = torch.cat([X, y], dim=1)
                batch_size = X_all.shape[0]
                if batch_size == 1:
                    video_tensor_unaugmented_14 = preprocess_image(X_all[0], target_size=518, patch_size=14).unsqueeze(0)

                    if cfg.MODEL.ENCODER_NAME == "dinov2":
                        video_tensor_unaugmented = preprocess_image(X_all[0], target_size=518, patch_size=14).unsqueeze(0)
                    else:
                        video_tensor_unaugmented = preprocess_image(X_all[0]).unsqueeze(0)
                    subset_video_tensor = video_tensor_unaugmented[:, :cfg.MODEL.M]
                else:
                    video_tensors = []
                    for b in range(batch_size):
                        sample = X_all[b]
                        processed_sample = preprocess_image(sample)
                        video_tensors.append(processed_sample)
                    video_tensor_unaugmented = torch.stack(video_tensors, dim=0)
                    subset_video_tensor = video_tensor_unaugmented[:, :cfg.MODEL.M]
                
                dtype = torch.bfloat16
                
                # Warmup forward passes
                with torch.amp.autocast('cuda', dtype=dtype):
                    _ = frozen_model(video_tensor_unaugmented_14)
                    _ = train_model(subset_video_tensor)
                
        except StopIteration:
            # If we run out of data, break early
            break
    
    torch.cuda.empty_cache()
    
    # Reset models to training mode
    train_model.train() 
    frozen_model.eval()  # Keep frozen model in eval mode

    # Synchronize all processes after warmup
    print(f"[GPU {accelerator.process_index}] Waiting for all GPUs after warmup...")
    accelerator.wait_for_everyone()
    print(f"[GPU {accelerator.process_index}] ✅ Model warmup completed - all processes synchronized")

    for epoch in range(cfg.TRAINING.NUM_EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch+1}/{cfg.TRAINING.NUM_EPOCHS}", 
            disable=not accelerator.is_local_main_process
        )
        for step, batch in enumerate(progress_bar):
            
            # Periodic memory cleanup to prevent gradual memory increase
            if step % 100 == 0:
                print(f"[GPU {accelerator.process_index}] Step {global_step}: Performing memory cleanup...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"[GPU {accelerator.process_index}] Step {global_step}: Memory cleanup complete")
                
            with accelerator.accumulate(train_model):

                # Process batch properly for any batch size
                X = batch[0]  # (B, m, C, H, W) - current frames
                y = batch[1]  # (B, n, C, H, W) - future frames
                X_all = torch.cat([X, y], dim=1)  # (B, T, C, H, W) where T = m + n
                
                # Create unaugmented tensor for frozen model (ground truth)
                batch_size = X_all.shape[0]
                if batch_size == 1:
                    # Optimize for batch_size=1 (most common case)
                    video_tensor_unaugmented_14 = preprocess_image(X_all[0], target_size=518, patch_size=14).unsqueeze(0)
                    
                else:
                    # Handle larger batch sizes
                    video_tensors_unaugmented = []
                    for b in range(batch_size):
                        sample = X_all[b]  # (T, C, H, W)
                        processed_sample = preprocess_image(sample)  # (T, C, H, W)
                        video_tensors_unaugmented.append(processed_sample)
                    
                    # Stack to create batch: (B, T, C, H, W)
                    video_tensor_unaugmented = torch.stack(video_tensors_unaugmented, dim=0)
                
                # Apply augmentations to each sample in the batch for training model
                augmented_samples = []
                for b in range(batch_size):
                    sample = X_all[b]  # (T, C, H, W)
                    # Apply random augmentations (different for each image in sequence)
                    augmented_sample = apply_random_augmentations(sample, training=cfg.AUGMENTATION.USE_AUGMENTATIONS)
                    augmented_samples.append(augmented_sample)
                
                # Stack back to batch format
                X_all_augmented = torch.stack(augmented_samples, dim=0)  # (B, T, C, H, W)
                
                # Process augmented samples for training model
                if batch_size == 1:
                    # Optimize for batch_size=1 (most common case)
                    if cfg.MODEL.ENCODER_NAME == "dinov2":
                        video_tensor_augmented = preprocess_image(X_all_augmented[0], target_size=518, patch_size=14).unsqueeze(0)
                    else:
                        video_tensor_augmented = preprocess_image(X_all_augmented[0]).unsqueeze(0)
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
                
                subset_video_tensor = video_tensor_augmented[:, :cfg.MODEL.M]  # (B, m, C, H, W) - augmented for training
                dtype = torch.bfloat16

                # Ensure all processes are synchronized before inference to prevent model state issues
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=dtype):
                        pseudo_gt = frozen_model(video_tensor_unaugmented_14)  # Use unaugmented data for ground truth
                
                # if global_step < 100 and accelerator.is_main_process:
                #     pseudo_gt_storage['step'].append(global_step)
                #     pseudo_gt_storage['point_maps'].append(pseudo_gt['points'].cpu().detach())
                #     pseudo_gt_storage['camera_poses'].append(pseudo_gt['camera_poses'].cpu().detach())
                #     pseudo_gt_storage['confidence'].append(pseudo_gt['conf'].cpu().detach())
                #     pseudo_gt_storage['local_point_maps'].append(pseudo_gt['local_points'].cpu().detach())
                    
                #     # Save original images
                #     pseudo_gt_storage['images_original'].append(X_all.cpu().detach())  # (B, T, C, H, W)
                    
                #     # Save every 10 steps to avoid losing data
                #     if (global_step + 1) % 10 == 0 or global_step == 99:
                #         print(f"💾 Saving pseudo_gt data at step {global_step}...")
                #         torch.save(pseudo_gt_storage, pseudo_gt_save_path)
                    
                torch.cuda.empty_cache()
                # run inference on the training model - handle both Pi3 and MapAnything
                with torch.amp.autocast('cuda', dtype=dtype):
                    # autoregressive pi3 model forward pass
                    predictions = train_model(subset_video_tensor)
                    
                # Align prediction and pseudo_gt shapes before loss computation
                predictions, pseudo_gt = align_prediction_shapes(predictions, pseudo_gt)
                predictions, pseudo_gt = normalize_pred_gt(predictions, pseudo_gt)

                motion_maps = None
                segmentation_masks = None
                
                # Generate optical flow ground truth if flow head is enabled (independent of segmentation/motion)
                if cfg.MODEL.USE_FLOW_HEAD:
                    # Prepare frames for RAFT: convert from augmented 14x14 to full resolution RGB
                    flow_frames = video_tensor_unaugmented_14[0]  # [T, 3, H, W]
                    
                    # Generate optical flow using RAFT
                    # Set visualize=True to see flow during training (useful for debugging)
                    visualize_flow = True
                    flow_gt = generate_optical_flow_gt(flow_frames, raft_model, InputPadder, device=device, visualize=visualize_flow)
                    
                    if flow_gt is not None:
                        # flow_gt shape: [B, T-1, H, W, 2]
                        # Pad with zeros for the last frame to match T frames
                        B_flow, T_minus_1, H_flow, W_flow, _ = flow_gt.shape
                        zero_flow = torch.zeros(B_flow, 1, H_flow, W_flow, 2, device=flow_gt.device)
                        flow_gt_padded = torch.cat([flow_gt, zero_flow], dim=1)  # [B, T, H, W, 2]
                        pseudo_gt['flow'] = flow_gt_padded
                        
                        # Generate motion GT from flow if motion head is enabled
                        if cfg.MODEL.USE_MOTION_HEAD:
                            # Get 3D point maps from pseudo GT (world coordinates)
                            point_maps = pseudo_gt['points']  # [B, T, H, W, 3]
                            
                            # Generate motion masks from flow and 3D points
                            # Pass RGB frames for segmentation
                            rgb_frames = video_tensor_unaugmented_14[0]  # [T, 3, H, W]
                            motion_gt = generate_motion_gt_from_flow(flow_gt, point_maps, cfg, 
                                                                    rgb_frames=rgb_frames,
                                                                    object_segformer=object_segformer,
                                                                    dynamic_classes=DYNAMIC_CLASSES)
                            pseudo_gt['motion'] = motion_gt  # [B, T, H, W, 1]
                            
                            print(f"[GPU {accelerator.process_index}] Generated flow-derived motion GT: {motion_gt.shape}")
                            dynamic_pixels = (motion_gt > 0.5).sum().item()
                            total_pixels = motion_gt.numel()
                            print(f"   Dynamic pixels: {dynamic_pixels} / {total_pixels} ({dynamic_pixels/total_pixels*100:.1f}%)")
                    else:
                        # raise exception
                        raise RuntimeError("RAFT failed to generate optical flow ground truth, ore RAFT model not initialized properly.")
                
                if cfg.MODEL.USE_SEGMENTATION_HEAD: # or cfg.POINT_MOTION.TRAIN_MOTION:
                    # Convert predictions to numpy for processing
                    point_maps = pseudo_gt['points'].cpu().numpy()  # (B, T, H, W, 3)
                    rgb_frames = [video_tensor_unaugmented_14[0, t].permute(1, 2, 0).cpu().numpy() for t in range(X_all.shape[1])]
                    # Denormalize RGB frames to [0, 255]
                    rgb_frames = [(frame * 255).astype(np.uint8) for frame in rgb_frames]

                    # Run GSAM2 for object segmentation
                    with torch.no_grad():
                        results = gsam2.process_frames(rgb_frames, "car. vehicle. person. road sign. traffic light", verbose=False)
                        
                    # Create class-aware composite masks for all frames
                    H, W = point_maps.shape[2:4]
                    composite_masks = []
                    
                    # Define class mapping based on segmentation model
                    if cfg.MODEL.SEGMENTATION_MODEL == "gsam2":
                        # GSAM2 uses 6-class system
                        class_mapping = {
                            'car': 1,           # All vehicles -> class 1
                            'vehicle': 1,       # All vehicles -> class 1  
                            'truck': 1,         # All vehicles -> class 1
                            'bus': 1,           # All vehicles -> class 1
                            'motorcycle': 1,    # All vehicles -> class 1
                            'bicycle': 2,       # Bicycle -> class 2
                            'person': 3,        # Person -> class 3
                            'road sign': 4,     # Road signs -> class 4
                            'traffic light': 5, # Traffic lights -> class 5
                            'default': 0        # Background/unrecognized -> class 0
                        }
                    else:
                        # Cityscapes models use 7-class system
                        # Results already contain class IDs, no mapping needed
                        class_mapping = None
                    
                    # Process masks differently based on model type
                    if cfg.MODEL.SEGMENTATION_MODEL == "gsam2":
                        # GSAM2 returns individual masks that need to be composed
                        for t in range(results['num_frames']):
                            frame_masks = results['masks'][t]
                            frame_composite_mask = np.zeros((H, W), dtype=np.uint8)
                            
                            for obj_idx, obj_key in enumerate(frame_masks.keys()):
                                # Get the label for this object
                                if obj_idx < len(results['labels']):
                                    label = results['labels'][obj_idx].lower().strip()
                                    # Map label to class value - try full label first
                                    class_value = class_mapping.get(label, None)
                                    if class_value is None:
                                        # Try first word only as fallback
                                        first_word = label.split(' ')[0].strip()
                                        class_value = class_mapping.get(first_word, class_mapping['default'])
                                else:
                                    class_value = class_mapping['default']
                                
                                segm_mask = frame_masks[obj_key].astype(bool)  # Convert to boolean mask
                                if segm_mask.ndim == 3:
                                    segm_mask = segm_mask[0]  # Remove first dimension if present
                                
                                # Assign class value to this object's pixels
                                frame_composite_mask[segm_mask] = class_value
                            
                            composite_masks.append(frame_composite_mask)
                    else:
                        # Cityscapes models return composite masks directly
                        composite_masks = results['composite_masks']
                    
                    # Run CoTracker for point tracking with binary mask
                    frames = torch.tensor(np.array(rgb_frames)).permute(0, 3, 1, 2)[None].float().to(device)

                    # Create motion masks that EXCLUDE static objects (road signs and traffic lights)
                    # Only include moving objects: vehicles (1), bicycles (2), and persons (3)
                    if cfg.POINT_MOTION.TRAIN_MOTION:
                        moving_object_classes = [1, 2, 3]  # Exclude 4 (road sign) and 5 (traffic light)
                        
                        motion_masks = []
                        for mask in composite_masks:
                            # Create binary mask only for moving objects
                            motion_mask = np.zeros_like(mask, dtype=np.uint8)
                            for class_val in moving_object_classes:
                                motion_mask[mask == class_val] = 1
                            motion_masks.append(motion_mask)
                        
                        # Use motion mask for CoTracker (only moving objects)
                        binary_mask = motion_masks[0]
                        binary_composite_masks = motion_masks
                    
                    if cfg.POINT_MOTION.TRAIN_MOTION:
                        with torch.no_grad():
                            # Reduce grid_size and disable backward tracking to save memory
                            pred_tracks, pred_visibility = cotracker(frames, grid_size=80, 
                                                            segm_mask=torch.from_numpy(binary_mask)[None, None], 
                                                            backward_tracking=True)
                            print(f"[GPU {accelerator.process_index}] Step {global_step}: CoTracker complete")
                            # Immediately move to CPU to free GPU memory
                            pred_tracks = pred_tracks.cpu()
                            pred_visibility = pred_visibility.cpu()
                            torch.cuda.empty_cache()

                        try:
                            _, motion_maps, dynamic_masks = analyze_object_dynamics(results, pred_tracks, pred_visibility, 
                                                                point_maps[0], composite_masks, verbose=False)
                        except Exception as e:
                            print(f"⚠️ analyze_object_dynamics failed: {e}")
                            print(f"   Shapes - pred_tracks: {pred_tracks.shape}, point_maps: {point_maps[0].shape}")
                            motion_maps = None
                            dynamic_masks = None


                        # add dynamic masks to pseudo_gt for object-based motion
                        if dynamic_masks is not None:
                            # dynamic_masks is a list of [H, W] binary masks (0=static, 1=dynamic)
                            dynamic_masks_array = np.stack(dynamic_masks, axis=0)  # (T, H, W)
                            # Add channel dimension to match expected shape
                            dynamic_masks_array = np.expand_dims(dynamic_masks_array, axis=-1)  # (T, H, W, 1)
                            motion_tensor = torch.from_numpy(dynamic_masks_array).float().to(device)  # (T, H, W, 1)
                            pseudo_gt['motion'] = motion_tensor.unsqueeze(0)  # Add batch dimension: (1, T, H, W, 1)

                    # Stack segmentation masks to (T, H, W, 1)
                    segmentation_masks = np.stack(composite_masks, axis=0)
                    segmentation_masks = np.expand_dims(segmentation_masks, axis=-1)
                    
                    # Save GSAM2 masks and motion maps for first 100 steps
                    # if global_step < 100 and accelerator.is_main_process:
                        #     pseudo_gt_storage['gsam2_composite_masks'].append(np.array(composite_masks))
                        #     # if motion_maps is not None:
                        #     #     pseudo_gt_storage['gsam2_motion_maps'].append(np.array(motion_maps))
                        #     pseudo_gt_storage['gsam2_labels'].append(results['labels'])
                        #     # Also save dynamic masks if available
                        #     if 'dynamic_masks' in locals() and dynamic_masks is not None:
                        #         print("💥 Saving GSAM2 dynamic masks for first 100 steps")
                        #         # dynamic_masks shape: [T, H, W] with 0=static, 1=dynamic
                        #         # Stack to match other mask format
                        #         dynamic_masks_array = np.array(dynamic_masks)
                        #         if 'gsam2_dynamic_masks' not in pseudo_gt_storage:
                        #             pseudo_gt_storage['gsam2_dynamic_masks'] = []
                        #         pseudo_gt_storage['gsam2_dynamic_masks'].append(dynamic_masks_array)
                        
                # Add segmentation masks to pseudo_gt if available
                if segmentation_masks is not None:
                    # Convert segmentation masks to tensor and add to pseudo_gt
                    # segmentation_masks are uint8 [0,255], keep as-is since SegmentationLosses.segmentation_bce_loss handles normalization
                    segmentation_tensor = torch.from_numpy(segmentation_masks).float().to(device)  # (T, H, W, 1)
                    pseudo_gt['segmentation'] = segmentation_tensor.unsqueeze(0)  # Add batch dimension: (1, T, H, W, 1)
                
                
                # compute loss between prediction and pseudo_gt with optional confidence weighting
                # Use FP32 for loss computation if enabled for better numerical stability
                loss_dtype = torch.float32 if cfg.TRAINING.USE_FP32_FOR_LOSSES else dtype
                with torch.amp.autocast('cuda', dtype=loss_dtype):
                    if cfg.LOSS.USE_CONF_WEIGHTED_POINTS:
                        point_map_loss, camera_pose_loss, conf_loss, normal_loss, segmentation_loss, motion_loss, flow_loss, frozen_decoder_loss = Pi3Losses.pi3_loss_with_confidence_weighting(
                            predictions, pseudo_gt, m_frames=cfg.MODEL.M, future_frame_weight=cfg.LOSS.FUTURE_FRAME_WEIGHT,
                            gamma=cfg.LOSS.CONF_GAMMA, alpha=cfg.LOSS.CONF_ALPHA, use_conf_weighted_points=True, gradient_weight=cfg.LOSS.GRADIENT_WEIGHT,
                            normal_loss_weight=cfg.LOSS.NORMAL_LOSS_WEIGHT,
                            segformer=segformer, images=video_tensor_unaugmented_14,
                            motion_threshold=cfg.POINT_MOTION.MOTION_THRESHOLD
                        )
                    else:
                        point_map_loss, camera_pose_loss, conf_loss, normal_loss, segmentation_loss, motion_loss, flow_loss, frozen_decoder_loss = Pi3Losses.pi3_loss(
                            predictions, pseudo_gt, m_frames=cfg.MODEL.M, future_frame_weight=cfg.LOSS.FUTURE_FRAME_WEIGHT, gradient_weight=cfg.LOSS.GRADIENT_WEIGHT,
                            normal_loss_weight=cfg.LOSS.NORMAL_LOSS_WEIGHT,
                            segformer=segformer, images=video_tensor_unaugmented_14,
                            motion_threshold=cfg.POINT_MOTION.MOTION_THRESHOLD
                        )
                
                pi3_loss = (cfg.LOSS.PC_LOSS_WEIGHT * point_map_loss) + (cfg.LOSS.POSE_LOSS_WEIGHT * camera_pose_loss) + (cfg.LOSS.CONF_LOSS_WEIGHT * conf_loss) + (cfg.LOSS.NORMAL_LOSS_WEIGHT * normal_loss) + (cfg.LOSS.SEGMENTATION_LOSS_WEIGHT * segmentation_loss) + (cfg.LOSS.MOTION_LOSS_WEIGHT * motion_loss) + (cfg.LOSS.FLOW_LOSS_WEIGHT * flow_loss) + (cfg.LOSS.FROZEN_DECODER_SUPERVISION_WEIGHT * frozen_decoder_loss)
                
                # Backward pass for Pi3 loss (optimize main model only)
                accelerator.backward(pi3_loss)
                
                # Separate distillation loss for distilled ViT only
                distillation_loss = torch.tensor(0.0, device=device, dtype=dtype)
                if distilled_vit is not None and cfg.MODEL.USE_DISTILLED_VIT:
                    with torch.amp.autocast('cuda', dtype=dtype):
                        # Get the last RGB frame fed into Pi3 (the last input frame)
                        last_input_frame = video_tensor_unaugmented_14[0, cfg.MODEL.M-1]  # [3, H, W] - last input frame
                        student_features = distilled_vit(last_input_frame.unsqueeze(0))  # [1, 3, H, W] -> single image
                        
                        # Extract teacher feature differences between M-1 and M indices  
                        teacher_features = {}
                        
                        # Get the teacher model features as mean pool of temporal frames
                        with torch.no_grad():
                            # Extract mean pooled features between consecutive frames
                            if 'point_features' in cfg.MODEL.DISTILLED_VIT.DISTILL_TOKENS:
                                # Mean pool frame M and frame M-1
                                frame_m_minus_1 = predictions['point_features'][cfg.MODEL.M - 1]  # [S, D]
                                frame_m = predictions['point_features'][cfg.MODEL.M]  # [S, D] 
                                teacher_features['point_features'] = ((frame_m + frame_m_minus_1) / 2.0).unsqueeze(0)  # [1, S, D]
                            
                            if 'camera_features' in cfg.MODEL.DISTILLED_VIT.DISTILL_TOKENS:
                                # Mean pool frame M and frame M-1
                                frame_m_minus_1 = predictions['camera_features'][cfg.MODEL.M - 1]  # [S, D]
                                frame_m = predictions['camera_features'][cfg.MODEL.M]  # [S, D]
                                teacher_features['camera_features'] = ((frame_m + frame_m_minus_1) / 2.0).unsqueeze(0)  # [1, S, D]
                                
                            if 'autonomy_features' in cfg.MODEL.DISTILLED_VIT.DISTILL_TOKENS:
                                # Mean pool frame M and frame M-1  
                                frame_m_minus_1 = predictions['autonomy_features'][cfg.MODEL.M - 1]  # [S, D]
                                frame_m = predictions['autonomy_features'][cfg.MODEL.M]  # [S, D]
                                teacher_features['autonomy_features'] = ((frame_m + frame_m_minus_1) / 2.0).unsqueeze(0)  # [1, S, D]
                        
                        # Compute distillation loss
                        if len(teacher_features) > 0:
                            distillation_losses = distillation_loss_fn(student_features, teacher_features)
                            distillation_loss = cfg.LOSS.DISTILLATION_LOSS_WEIGHT * distillation_losses['total_distillation_loss']
                            print(f"   Step {global_step}: Distillation loss = {distillation_loss.item():.6f}")
                            # Separate backward pass for distillation loss (optimize distilled ViT only)
                            accelerator.backward(distillation_loss)
                            
                            # Critical: Clean up distillation variables to prevent memory accumulation
                            del student_features, teacher_features, distillation_losses
                            del last_input_frame
                            torch.cuda.empty_cache()

                if accelerator.sync_gradients:
                    if cfg.TRAINING.DETECT_NANS and check_model_parameters(train_model, "train_model", global_step):
                        print(f"🚨 NaN gradients detected at step {global_step}! Skipping optimizer step...")
                        optimizer.zero_grad()  # Clear gradients and continue
                        continue
                    
                    accelerator.clip_grad_norm_(train_model.parameters(), cfg.TRAINING.MAX_GRAD_NORM)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update distilled ViT if enabled
                if distilled_vit is not None and cfg.MODEL.USE_DISTILLED_VIT:
                    distilled_vit_optimizer.step()
                    distilled_vit_scheduler.step()
                    distilled_vit_optimizer.zero_grad()

            # Store loss value immediately and delete large tensors
            current_loss = pi3_loss.detach().item()  # Only Pi3 loss for main training
            current_distillation_loss = distillation_loss.detach().item()
            
            # Aggressive memory cleanup after optimization
            if accelerator.sync_gradients:
                torch.cuda.empty_cache()
            
            # Sync all processes after memory cleanup (outside conditional to avoid deadlock)
            if cfg.OUTPUT.SAVE_DEPTHS and 'local_points' in predictions and global_step % 100 == 0 and accelerator.is_main_process:
                # Convert tensors to numpy
                points = pseudo_gt["local_points"]
                masks = torch.sigmoid(pseudo_gt["conf"][..., 0]) > 0.1
                original_height, original_width = points.shape[-3:-1]
                aspect_ratio = original_width / original_height

                pseudo_gt['images'] = video_tensor_unaugmented.permute(0, 1, 3, 4, 2)
                pseudo_gt['conf'] = torch.sigmoid(pseudo_gt['conf'])
                edge = depth_edge(pseudo_gt['local_points'][..., 2], rtol=0.03)
                pseudo_gt['conf'][edge] = 0.0

                for key in pseudo_gt.keys():
                    if key not in ['features', 'pos', 'dino_features', 'pi3_features', 'conf_features', 'camera_features',
                                   'point_features']:
                        if isinstance(pseudo_gt[key], torch.Tensor):
                            pseudo_gt[key] = pseudo_gt[key].cpu().numpy().squeeze(0)  # remove batch dimension

                for key in predictions.keys():
                    if key not in ['all_decoder_features', 'all_positional_encoding', 'dino_features',
                                   'pi3_features', 'autonomy_features', 'conf_features', 'camera_features',
                                   'point_features']:
                        if isinstance(predictions[key], torch.Tensor):
                            predictions[key] = predictions[key].clone().detach().cpu().numpy().squeeze(0)  # remove batch dimension

                import matplotlib.pyplot as plt
                import imageio
                import cv2
                
                # === GSAM2 INFERENCE ===
                # Limit visualizations to first 1000 steps to prevent memory accumulation
                # Only run GSAM2 on main process to avoid multi-GPU deadlocks
                if (cfg.MODEL.USE_SEGMENTATION_HEAD and cfg.POINT_MOTION.TRAIN_MOTION and 
                    epoch == 0 and global_step < 1000 and accelerator.is_main_process):
                    B, T, C, H, W = video_tensor_unaugmented_14.shape
                    rgb_frames = []
                    for t in range(min(T, 10)):
                        img = video_tensor_unaugmented_14[0, t].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
                        img = np.clip(img * 255, 0, 255).astype(np.uint8)
                        rgb_frames.append(img)
                    
                    try:
                        results = gsam2.process_frames(rgb_frames, "car. vehicle. person. road sign. traffic light.", verbose=True)
                        print(f"🎭 GSAM2: {results['num_objects']} objects in {results['num_frames']} frames")
                        first_frame_masks = results['masks'][0]  # (num_objects, H, W)
                        
                        # Create composite masks for all frames and stack them
                        composite_masks = []
                        
                        # Process all frames to create composite masks
                        for t in range(results['num_frames']):
                            frame_masks = results['masks'][t]
                            frame_composite_mask = np.zeros((H, W), dtype=np.uint8)
                            
                            for obj_key in frame_masks.keys():
                                segm_mask = frame_masks[obj_key].astype(np.uint8) * 255
                                # OR with frame composite mask
                                frame_composite_mask = cv2.bitwise_or(frame_composite_mask, segm_mask[0])
                            
                            composite_masks.append(frame_composite_mask)
                        
                        # Stack composite masks to shape (N, H, W, 1)
                        stacked_composite_masks = np.stack(composite_masks, axis=0)  # (N, H, W)
                        stacked_composite_masks = np.expand_dims(stacked_composite_masks, axis=-1)  # (N, H, W, 1)
                        
                        print(f"📋 Created stacked composite masks: {stacked_composite_masks.shape}")
                        
                        # Use first frame composite mask for CoTracker (convert to binary)
                        # For the visualization section, we'll also track all objects to show what's happening
                        # But you could change this to motion_masks if you only want to visualize moving objects
                        binary_mask = (composite_masks[0] > 0).astype(np.uint8)
                        frames = torch.tensor(np.array(rgb_frames)).permute(0, 3, 1, 2)[None].float().to(device)

                        if cfg.POINT_MOTION.TRAIN_MOTION:
                            print(f"🎯 CoTracker binary mask: {binary_mask.min()}-{binary_mask.max()}, unique values: {np.unique(binary_mask)}")
                            
                            pred_tracks, pred_visibility = cotracker(frames, grid_size=80, segm_mask=torch.from_numpy(binary_mask)[None, None], backward_tracking=True)
                            
                            # for motion maps: 
                            # shape of pred tracks: (1, T, num_points, H, W)
                            vis = Visualizer(
                                save_dir=f'./videos_{step}',
                                pad_value=100,
                                linewidth=2,
                            )
                            vis.visualize(
                                video=frames,
                                tracks=pred_tracks,
                                visibility=pred_visibility,
                                filename='segm_grid')
                            

                        # Analyze object dynamics using CoTracker2 tracks and point maps
                        point_maps = pseudo_gt['points']
                        if cfg.POINT_MOTION.TRAIN_MOTION:
                            try:
                                dynamic_analysis, motion_maps, dynamic_masks = analyze_object_dynamics(results, pred_tracks, pred_visibility, point_maps, composite_masks, verbose=True)
                            except Exception as e:
                                print(f"⚠️ analyze_object_dynamics visualization failed: {e}")
                                dynamic_analysis = {}
                                motion_maps = []
                                dynamic_masks = []
                        
                            # Log motion map statistics
                            if len(motion_maps) > 0:
                                avg_motion_magnitude = np.mean([np.linalg.norm(m, axis=-1).mean() for m in motion_maps])
                                max_motion_magnitude = np.max([np.linalg.norm(m, axis=-1).max() for m in motion_maps])
                                print(f"🏃 Motion maps: {len(motion_maps)} transitions, avg magnitude: {avg_motion_magnitude:.4f}m, max: {max_motion_magnitude:.4f}m")
                                
                                # Visualize motion maps
                                motion_map_files = visualize_motion_maps(motion_maps, step)
                                motion_overlay_files = visualize_motion_flow_overlay(rgb_frames, motion_maps, step)
                            
                            # Visualize dynamic objects
                            visualize_dynamic_objects(rgb_frames, results, dynamic_analysis, point_maps, step, run)
                            
                        # Clean up visualization memory
                        del rgb_frames
                        del frames
                        if 'pred_tracks' in locals():
                            del pred_tracks
                        if 'pred_visibility' in locals():
                            del pred_visibility
                        if 'results' in locals():
                            del results
                        if 'composite_masks' in locals():
                            del composite_masks
                        if 'motion_maps' in locals():
                            del motion_maps
                        if 'dynamic_masks' in locals():
                            del dynamic_masks
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"⚠️ GSAM2 failed: {e}")
                
                # Synchronize all processes after GSAM2 inference
                # === RGB INPUT FRAMES VISUALIZATION ===
                rgb_images_for_wandb = []
                if cfg.WANDB.USE_WANDB:
                    B, T, C, H, W = X_all_augmented.shape
                    for t in range(min(T, 10)):  # Log up to 6 frames
                        # Convert from tensor to numpy and transpose for WandB
                        img = X_all_augmented[0, t].cpu().numpy()  # (C, H, W)
                        img = np.transpose(img, (1, 2, 0))  # (H, W, C)
                        img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Scale to [0, 255]
                        rgb_images_for_wandb.append(wandb.Image(img, caption=f"Train Frame {t}"))

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
                    
                    # Save depth to disk
                    imageio.imwrite(f"depth_frame_{t}_viridis.png", colored_uint8)
                    
                    # Save corresponding RGB frame
                    if t < X_all_augmented.shape[1]:  # Safety check
                        # Save augmented RGB frame
                        rgb_tensor_aug = X_all_augmented[0, t].cpu()  # Shape: (C, H, W)
                        rgb_img_aug = np.transpose(rgb_tensor_aug.numpy(), (1, 2, 0))  # Shape: (H, W, C)
                        rgb_img_aug = np.clip(rgb_img_aug * 255, 0, 255).astype(np.uint8)
                        imageio.imwrite(f"rgb_frame_{t}_augmented.png", rgb_img_aug)
                        
                        # Save unaugmented RGB frame  
                        rgb_tensor_orig = X_all[0, t].cpu()  # Shape: (C, H, W)
                        rgb_img_orig = np.transpose(rgb_tensor_orig.numpy(), (1, 2, 0))  # Shape: (H, W, C)
                        rgb_img_orig = np.clip(rgb_img_orig * 255, 0, 255).astype(np.uint8)
                        imageio.imwrite(f"rgb_frame_{t}_original.png", rgb_img_orig)
                        
                        print(f"💾 Saved frames: depth_frame_{t}_viridis.png, rgb_frame_{t}_original.png, rgb_frame_{t}_augmented.png")
                    
                    # Prepare for WandB
                    if cfg.WANDB.USE_WANDB:
                        from PIL import Image as PILImage
                        pil_image = PILImage.fromarray(colored_uint8)
                        depth_images_for_wandb.append(wandb.Image(pil_image, caption=f"Depth Frame {t}"))
                
                # === SEGMENTATION VISUALIZATION ===
                segmentation_images_for_wandb = []
                if cfg.OUTPUT.SAVE_SEGMENTATION and 'segmentation' in predictions:
                    pred_segmentation = predictions['segmentation']  # shape [B, T, H, W, 6] from model
                    
                    # Handle tensor vs numpy array and remove batch dimension
                    if isinstance(pred_segmentation, torch.Tensor):
                        pred_segmentation = pred_segmentation.cpu().detach().numpy()
                    
                    # Remove batch dimension if present: [B, T, H, W, 9] -> [T, H, W, 9]
                    if pred_segmentation.ndim == 5 and pred_segmentation.shape[0] == 1:
                        pred_segmentation = pred_segmentation.squeeze(0)
                    
                    for t in range(min(pred_segmentation.shape[0], 8)):  # Limit to 8 frames
                        pred_frame = pred_segmentation[t]  # (H, W, 6)
                        # Check if we have multi-class output (6 channels) or single-class (1 channel)
                        if pred_frame.shape[-1] >= 6:  # Multi-class output
                            # Multi-class segmentation: apply softmax and get class predictions
                            num_classes = pred_frame.shape[-1]
                            pred_logits = torch.from_numpy(pred_frame)  # (H, W, num_classes)
                            pred_probs = torch.softmax(pred_logits, dim=-1)  # (H, W, num_classes)
                            pred_classes = torch.argmax(pred_probs, dim=-1)  # (H, W)
                            pred_classes_np = pred_classes.numpy()
                            
                            # Use categorical colormap for class visualization
                            import matplotlib.pyplot as plt
                            cmap = plt.get_cmap('tab10')
                            # Normalize class values to [0,1] for colormap
                            pred_vis = cmap(pred_classes_np / (num_classes - 1))[:, :, :3]  # (H, W, 3)
                            pred_vis_uint8 = (pred_vis * 255).astype(np.uint8)
                            
                            unique_classes = np.unique(pred_classes_np)
                            # Set class names based on number of classes
                            if num_classes == 6:
                                class_names = ['background', 'vehicle', 'bicycle', 'person', 'road sign', 'traffic light']
                            elif num_classes == 7:
                                class_names = ['road', 'vehicle', 'person', 'traffic light', 'traffic sign', 'sky', 'bg/building']
                            else:
                                class_names = [f'class_{i}' for i in range(num_classes)]
                            present_classes = [class_names[i] for i in unique_classes if i < len(class_names)]
                            print(f"🎨 Pred segmentation frame {t}: multi-class visualization")
                            print(f"   Present classes: {present_classes}")
                            print(f"   Class counts: {[(class_names[i], np.sum(pred_classes_np == i)) for i in unique_classes if i < len(class_names)]}")
                            
                            # Save a color legend for the first frame
                            if t == 0:
                                try:
                                    import matplotlib.pyplot as plt
                                    import matplotlib.patches as patches
                                    
                                    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                                    ax.set_xlim(0, 10)
                                    ax.set_ylim(0, len(class_names))
                                    
                                    cmap = plt.get_cmap('tab10')
                                    for i, class_name in enumerate(class_names):
                                        color = cmap(i / 10.0)[:3]  # RGB values
                                        rect = patches.Rectangle((0, len(class_names) - i - 1), 8, 0.8, 
                                                               linewidth=1, edgecolor='black', facecolor=color)
                                        ax.add_patch(rect)
                                        ax.text(8.5, len(class_names) - i - 0.6, f"{i}: {class_name}", 
                                               fontsize=12, va='center')
                                    
                                    ax.set_title('Multi-Class Segmentation Color Legend', fontsize=14, fontweight='bold')
                                    ax.set_xlabel('Class Colors')
                                    ax.set_yticks([])
                                    ax.set_xticks([])
                                    
                                    plt.tight_layout()
                                    plt.savefig('segmentation_color_legend.png', dpi=150, bbox_inches='tight')
                                    plt.close()
                                    print("🎨 Saved color legend: segmentation_color_legend.png")
                                except Exception as e:
                                    print(f"⚠️ Could not save color legend: {e}")
                            
                        else:
                            # Single-channel output: handle as before
                            if pred_frame.ndim == 3 and pred_frame.shape[-1] == 1:
                                pred_frame = pred_frame.squeeze(-1)  # (H, W)
                            
                            # Apply sigmoid to convert logits to probabilities [0,1]
                            pred_prob = torch.sigmoid(torch.from_numpy(pred_frame))
                            pred_prob_np = pred_prob.numpy()
                            
                            # Check if we have class-aware ground truth to match visualization style
                            if 'segmentation' in pseudo_gt:
                                gt_max_val = pseudo_gt['segmentation'].max() if isinstance(pseudo_gt['segmentation'], np.ndarray) else pseudo_gt['segmentation'].max().item()
                                
                                if gt_max_val > 1.0:  # Class-aware case
                                    # Convert predicted probabilities back to class scale for visualization
                                    pred_class_scale = pred_prob_np * 8.0  # Scale back to [0,8] range
                                    
                                    # Use categorical colormap for consistency with GT
                                    import matplotlib.pyplot as plt
                                    cmap = plt.get_cmap('tab10')
                                    # Normalize to [0,1] for colormap
                                    pred_vis = cmap(pred_class_scale / 10.0)[:, :, :3]  # (H, W, 3)
                                    pred_vis_uint8 = (pred_vis * 255).astype(np.uint8)
                                    print(f"🎨 Pred segmentation frame {t}: class-aware visualization (range: {pred_class_scale.min():.2f}-{pred_class_scale.max():.2f})")
                                else:
                                    # Binary case - use grayscale
                                    pred_vis = np.stack([pred_prob_np] * 3, axis=-1)  # (H, W, 3)
                                    pred_vis_uint8 = (pred_vis * 255).astype(np.uint8)
                                    print(f"🎨 Pred segmentation frame {t}: binary visualization")
                            else:
                                # Default to grayscale if no GT available
                                pred_vis = np.stack([pred_prob_np] * 3, axis=-1)  # (H, W, 3)
                                pred_vis_uint8 = (pred_vis * 255).astype(np.uint8)
                        
                        # Save to disk 
                        filename = f"segmentation_pred_frame_{t}.png"
                        imageio.imwrite(filename, pred_vis_uint8)
                        print(f"💙 Saved predicted segmentation for frame {t} to {filename}")
                        
                        # Prepare for WandB
                        if cfg.WANDB.USE_WANDB:
                            from PIL import Image as PILImage
                            pil_image = PILImage.fromarray(pred_vis_uint8)
                            segmentation_images_for_wandb.append(wandb.Image(pil_image, caption=f"Pred Seg Frame {t}"))
                    
                
                # Log predicted segmentation to WandB
                if cfg.WANDB.USE_WANDB and segmentation_images_for_wandb:
                    wandb.log({"visualizations/segmentation_pred": segmentation_images_for_wandb}, step=global_step)

                # === GT SEGMENTATION VISUALIZATION ===
                # Visualize ground truth segmentation if available (independent of prediction segmentation)
                if cfg.OUTPUT.SAVE_SEGMENTATION and 'segmentation' in pseudo_gt:
                    # Handle both tensor and numpy array cases
                    if isinstance(pseudo_gt['segmentation'], torch.Tensor):
                        gt_segmentation = pseudo_gt['segmentation'].cpu().numpy()
                    else:
                        gt_segmentation = pseudo_gt['segmentation']
                    
                    # Handle shape: could be (B, T, H, W, 1), (T, H, W, 1), or (T, H, W)
                    while gt_segmentation.ndim > 3:
                        # Remove batch dimension if present (size 1)
                        if gt_segmentation.shape[0] == 1:
                            gt_segmentation = gt_segmentation.squeeze(0)
                        # Remove channel dimension if present (size 1)
                        elif gt_segmentation.shape[-1] == 1:
                            gt_segmentation = gt_segmentation.squeeze(-1)
                        else:
                            break
                    
                    gt_segmentation_images_for_wandb = []
                    for t in range(min(gt_segmentation.shape[0], 8)):  # Limit to 8 frames
                        gt_frame = gt_segmentation[t]
                        
                        # Check if we have class-aware masks (values > 1 and discrete)
                        max_val = gt_frame.max()
                        unique_vals = np.unique(gt_frame)
                        
                        if max_val > 1.0 and len(unique_vals) <= 10:  # Class-aware masks
                            # Use categorical colormap for class visualization
                            import matplotlib.pyplot as plt
                            cmap = plt.get_cmap('tab10')  # Distinct colors for up to 10 classes
                            # Normalize class values to [0,1] for colormap
                            gt_normalized = gt_frame / max_val
                            gt_vis = cmap(gt_normalized)[:, :, :3]  # (H, W, 3), drop alpha
                            gt_vis_uint8 = (gt_vis * 255).astype(np.uint8)
                            
                            # Determine number of classes from unique values
                            num_classes_gt = int(max_val) + 1 if max_val < 10 else 6  # Fallback to 6 if unclear
                            
                            # Set class names based on number of classes
                            if num_classes_gt == 6:
                                class_names = ['background', 'vehicle', 'bicycle', 'person', 'road sign', 'traffic light']
                            elif num_classes_gt == 7:
                                class_names = ['road', 'vehicle', 'person', 'traffic light', 'traffic sign', 'sky', 'bg/building']
                            else:
                                class_names = [f'class_{i}' for i in range(num_classes_gt)]
                            
                            present_classes = [class_names[int(i)] for i in unique_vals if int(i) < len(class_names)]
                            print(f"🎨 GT segmentation frame {t}: class-aware visualization")
                            print(f"   Present classes: {present_classes}")
                            print(f"   Class counts: {[(class_names[int(i)], np.sum(gt_frame == i)) for i in unique_vals if int(i) < len(class_names)]}")
                        else:
                            # Binary or continuous masks - use grayscale
                            gt_normalized = gt_frame / 255.0 if gt_frame.max() > 1.0 else gt_frame
                            # Convert to RGB for visualization (grayscale -> RGB)
                            gt_vis = np.stack([gt_normalized] * 3, axis=-1)  # (H, W, 3)
                            gt_vis_uint8 = (gt_vis * 255).astype(np.uint8)
                            print(f"🎨 GT segmentation frame {t}: binary/continuous visualization")
                        
                        # Save to disk
                        filename = f"segmentation_gt_frame_{t}.png"
                        imageio.imwrite(filename, gt_vis_uint8)
                        print(f"💙 Saved ground truth segmentation for frame {t} to {filename}")
                        
                        # Prepare for WandB
                        if cfg.WANDB.USE_WANDB:
                            from PIL import Image as PILImage
                            pil_image = PILImage.fromarray(gt_vis_uint8)
                            gt_segmentation_images_for_wandb.append(wandb.Image(pil_image, caption=f"GT Seg Frame {t}"))
                    
                    # Log GT segmentation to WandB
                    if cfg.WANDB.USE_WANDB and gt_segmentation_images_for_wandb:
                        wandb.log({"visualizations/segmentation_gt": gt_segmentation_images_for_wandb}, step=global_step)

                # === MOTION VISUALIZATION ===
                motion_images_for_wandb = []
                if 'motion' in predictions:
                    pred_motion = predictions['motion']  # shape [B, T, H, W, 1] with binary motion mask logits
                    
                    # Handle tensor vs numpy array and remove batch dimension
                    if isinstance(pred_motion, torch.Tensor):
                        # Apply sigmoid to get probabilities
                        pred_motion = torch.sigmoid(pred_motion).cpu().detach().numpy()
                    else:
                        # If numpy, apply sigmoid
                        pred_motion = 1.0 / (1.0 + np.exp(-pred_motion))
                    
                    # Remove batch dimension if present: [B, T, H, W, 1] -> [T, H, W, 1]
                    if pred_motion.ndim == 5 and pred_motion.shape[0] == 1:
                        pred_motion = pred_motion.squeeze(0)
                    
                    for t in range(min(pred_motion.shape[0], 8)):  # Limit to 8 frames
                        motion_frame = pred_motion[t]  # (H, W, 1) - binary motion probability
                        
                        # Squeeze last dimension to get (H, W)
                        if motion_frame.ndim == 3 and motion_frame.shape[-1] == 1:
                            motion_frame = motion_frame.squeeze(-1)
                        
                        # Motion frame now contains probabilities [0, 1]
                        # Create a heatmap visualization
                        import matplotlib.pyplot as plt
                        import matplotlib.colors as mcolors
                        
                        # Create red-to-blue colormap for motion (blue=static, red=moving)
                        cmap = plt.get_cmap('coolwarm')
                        motion_colored = cmap(motion_frame)
                        motion_rgb = (motion_colored[:, :, :3] * 255).astype(np.uint8)
                        
                        # Save binary motion visualization to disk
                        filename = f"motion_pred_frame_{t}.png"
                        imageio.imwrite(filename, motion_rgb)
                        print(f"🏃 Saved predicted motion mask for frame {t} to {filename}")
                        print(f"   Motion probability range: {motion_frame.min():.3f} - {motion_frame.max():.3f}")
                        print(f"   Pixels with motion > 0.5: {(motion_frame > 0.5).sum()} / {motion_frame.size}")
                        
                        # Prepare for WandB
                        if cfg.WANDB.USE_WANDB:
                            from PIL import Image as PILImage
                            pil_image = PILImage.fromarray(motion_rgb)
                            motion_images_for_wandb.append(wandb.Image(pil_image, caption=f"Motion Mask Frame {t}"))
                    
                    # Also visualize ground truth motion if available
                    if 'motion' in pseudo_gt:
                        # Handle both tensor and numpy array cases
                        if isinstance(pseudo_gt['motion'], torch.Tensor):
                            gt_motion = pseudo_gt['motion'].cpu().numpy()
                        else:
                            gt_motion = pseudo_gt['motion']
                        
                        # Handle shape: could be (B, T, H, W, 3), (T, H, W, 3)
                        while gt_motion.ndim > 4:
                            # Remove batch dimension if present (size 1)
                            if gt_motion.shape[0] == 1:
                                gt_motion = gt_motion.squeeze(0)
                            else:
                                break
                        
                        gt_motion_images_for_wandb = []
                        for t in range(min(gt_motion.shape[0], 8)):  # Limit to 8 frames
                            gt_frame = gt_motion[t]  # (H, W, 1) - already binary
                            
                            # Squeeze last dimension to get (H, W)
                            if gt_frame.ndim == 3 and gt_frame.shape[-1] == 1:
                                gt_frame = gt_frame.squeeze(-1)
                            
                            # GT is already binary (0 or 1)
                            binary_gt_mask = gt_frame
                            
                            # Create same visualization as predictions
                            cmap = plt.get_cmap('coolwarm')
                            gt_motion_colored = cmap(binary_gt_mask)
                            gt_motion_rgb = (gt_motion_colored[:, :, :3] * 255).astype(np.uint8)
                            
                            # Save to disk
                            filename = f"motion_gt_frame_{t}.png"
                            imageio.imwrite(filename, gt_motion_rgb)
                            print(f"🏃 Saved ground truth motion mask for frame {t} to {filename}")
                            print(f"   GT pixels with motion: {binary_gt_mask.sum()} / {binary_gt_mask.size}")
                            
                            # Prepare for WandB
                            if cfg.WANDB.USE_WANDB:
                                from PIL import Image as PILImage
                                pil_image = PILImage.fromarray(gt_motion_rgb)
                                gt_motion_images_for_wandb.append(wandb.Image(pil_image, caption=f"GT Motion Mask Frame {t}"))
                        
                        # Log GT motion to WandB
                        if cfg.WANDB.USE_WANDB and gt_motion_images_for_wandb:
                            wandb.log({"visualizations/motion_gt": gt_motion_images_for_wandb}, step=global_step)
                
                # Log predicted motion to WandB
                if cfg.WANDB.USE_WANDB and motion_images_for_wandb:
                    wandb.log({"visualizations/motion_pred": motion_images_for_wandb}, step=global_step)

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
                        
                        # Prepare for WandB
                        if cfg.WANDB.USE_WANDB:
                            from PIL import Image as PILImage
                            pil_image = PILImage.fromarray(colored_uint8)
                            confidence_images_for_wandb.append(wandb.Image(pil_image, caption=f"Confidence Frame {t}"))
                
                # === NORMAL MAP VISUALIZATION ===
                normal_images_for_wandb = []
                if 'local_points' in predictions:
                    try:
                        # Convert back to torch tensor for normal computation
                        points_tensor = torch.from_numpy(predictions['local_points']).float()  # (T, H, W, 3)
                        
                        for t in range(points_tensor.shape[0]):
                            # Compute normals for this frame
                            frame_points = points_tensor[t:t+1]  # (1, H, W, 3) - add batch dimension
                            normals = NormalLosses.compute_normals_from_grid(frame_points)  # (1, H-2, W-2, 3)
                            normals = normals.squeeze(0).numpy()  # (H-2, W-2, 3)
                            
                            # Convert normals from [-1,1] to [0,1] for visualization
                            # RGB channels represent X, Y, Z components of normal vectors
                            normals_vis = (normals + 1.0) * 0.5  # [-1,1] -> [0,1]
                            normals_vis = np.clip(normals_vis, 0, 1)
                            
                            # Convert to uint8 RGB
                            normal_rgb = (normals_vis * 255).astype(np.uint8)
                            
                            # Pad back to original size (add border pixels that were excluded)
                            H_orig, W_orig = points_tensor.shape[1:3]
                            normal_rgb_padded = np.zeros((H_orig, W_orig, 3), dtype=np.uint8)
                            normal_rgb_padded[1:H_orig-1, 1:W_orig-1] = normal_rgb
                            
                            # Save to disk
                            imageio.imwrite(f"normal_frame_{t}_rgb.png", normal_rgb_padded)
                            
                            # Prepare for WandB
                            if cfg.WANDB.USE_WANDB:
                                from PIL import Image as PILImage
                                pil_image = PILImage.fromarray(normal_rgb_padded)
                                normal_images_for_wandb.append(wandb.Image(pil_image, caption=f"Normal Frame {t}"))
                        
                        print(f"💙 Saved {points_tensor.shape[0]} normal maps")
                    except Exception as e:
                        print(f"❌ Error computing normal maps: {e}")
                
                # === OPTICAL FLOW VISUALIZATION ===
                flow_images_for_wandb = []
                if 'flow' in predictions and cfg.MODEL.USE_FLOW_HEAD:
                    try:
                        pred_flow = predictions['flow']  # shape (T, H, W, 2) - (dx, dy) per pixel
                        
                        # Also get ground truth flow if available
                        gt_flow = None
                        if 'flow' in pseudo_gt:
                            if isinstance(pseudo_gt['flow'], torch.Tensor):
                                gt_flow = pseudo_gt['flow'].cpu().numpy()
                            else:
                                gt_flow = pseudo_gt['flow']
                            
                            # Remove batch dimension if present
                            if gt_flow.ndim == 5 and gt_flow.shape[0] == 1:
                                gt_flow = gt_flow.squeeze(0)
                        
                        # Visualize flow for each frame transition
                        for t in range(min(pred_flow.shape[0] - 1, 6)):  # T-1 transitions, limit to 6
                            # Predicted flow
                            flow = pred_flow[t]  # (H, W, 2) - flow in pixel units
                            
                            # Flow is already in pixel units (no denormalization needed)
                            flow_pixels = flow
                            
                            # Compute flow magnitude and angle
                            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                            angle = np.arctan2(flow[..., 1], flow[..., 0])
                            
                            # HSV visualization (hue=direction, saturation=1, value=magnitude)
                            h, w = flow.shape[:2]
                            hsv = np.zeros((h, w, 3), dtype=np.uint8)
                            hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)  # Hue (angle)
                            hsv[..., 1] = 255  # Saturation (always full)
                            # Flow is now in pixel units, typical values are 0-200 pixels movement
                            # Values > 200 indicate very fast motion
                            # Clip to [0, 200] range for visualization then map to [0, 255]
                            mag_norm = np.clip(magnitude / 200.0, 0, 1)  # Divide by 200 to visualize up to 200 pixels
                            hsv[..., 2] = (mag_norm * 255).astype(np.uint8)  # Value (magnitude)
                            
                            # Convert to RGB
                            import cv2
                            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                            
                            # Save predicted flow
                            filename = f"flow_pred_frame_{t}_to_{t+1}.png"
                            imageio.imwrite(filename, flow_rgb)
                            print(f"🌊 Saved predicted flow for frames {t}→{t+1} to {filename}")
                            print(f"   Flow magnitude range (pixels): {magnitude.min():.3f} - {magnitude.max():.3f}")
                            print(f"   Flow component range (pixels): {flow_pixels[..., 0].min():.1f} - {flow_pixels[..., 0].max():.1f} (x), {flow_pixels[..., 1].min():.1f} - {flow_pixels[..., 1].max():.1f} (y)")
                            
                            # Prepare for WandB
                            if cfg.WANDB.USE_WANDB:
                                from PIL import Image as PILImage
                                pil_image = PILImage.fromarray(flow_rgb)
                                flow_images_for_wandb.append(wandb.Image(pil_image, caption=f"Flow Pred {t}→{t+1}"))
                            
                            # Save ground truth flow if available
                            if gt_flow is not None and t < gt_flow.shape[0]:
                                gt_flow_t = gt_flow[t]  # (H, W, 2) - flow in pixel units
                                
                                # GT flow is already in pixel units (no denormalization needed)
                                gt_flow_pixels = gt_flow_t
                                
                                # Same visualization for GT
                                gt_magnitude = np.sqrt(gt_flow_t[..., 0]**2 + gt_flow_t[..., 1]**2)
                                gt_angle = np.arctan2(gt_flow_t[..., 1], gt_flow_t[..., 0])
                                
                                gt_hsv = np.zeros((h, w, 3), dtype=np.uint8)
                                gt_hsv[..., 0] = ((gt_angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
                                gt_hsv[..., 1] = 255
                                # Use same normalization as predictions for consistent visualization
                                gt_mag_norm = np.clip(gt_magnitude / 200.0, 0, 1)  # Divide by 200 to visualize up to 200 pixels
                                gt_hsv[..., 2] = (gt_mag_norm * 255).astype(np.uint8)
                                
                                gt_flow_rgb = cv2.cvtColor(gt_hsv, cv2.COLOR_HSV2RGB)
                                
                                # Save GT flow
                                gt_filename = f"flow_gt_frame_{t}_to_{t+1}.png"
                                imageio.imwrite(gt_filename, gt_flow_rgb)
                                print(f"🌊 Saved ground truth flow for frames {t}→{t+1} to {gt_filename}")
                                print(f"   GT flow magnitude range (pixels): {gt_magnitude.min():.3f} - {gt_magnitude.max():.3f}")
                                print(f"   GT flow component range (pixels): {gt_flow_pixels[..., 0].min():.1f} - {gt_flow_pixels[..., 0].max():.1f} (x), {gt_flow_pixels[..., 1].min():.1f} - {gt_flow_pixels[..., 1].max():.1f} (y)")
                                
                                if cfg.WANDB.USE_WANDB:
                                    gt_pil_image = PILImage.fromarray(gt_flow_rgb)
                                    flow_images_for_wandb.append(wandb.Image(gt_pil_image, caption=f"Flow GT {t}→{t+1}"))
                        
                        # Log flow to WandB
                        if cfg.WANDB.USE_WANDB and flow_images_for_wandb:
                            wandb.log({"visualizations/flow": flow_images_for_wandb}, step=global_step)
                            
                    except Exception as e:
                        print(f"❌ Error visualizing flow: {e}")
                        import traceback
                        traceback.print_exc()

                # === MOTION DETECTION FROM OPTICAL FLOW ===
                motion_images_for_wandb = []
                if cfg.MOTION_DETECTION.ENABLE and 'flow' in predictions and 'local_points' in predictions:
                    from utils.flow_motion import FlowMotionDetector
                    from utils.dynamic_objects import DynamicObjectTracker
                    from utils.motion_visualization import MotionVisualizer
                    
                    # Initialize motion detection components
                    motion_detector = FlowMotionDetector(motion_threshold=cfg.MOTION_DETECTION.MOTION_THRESHOLD)
                    motion_visualizer = MotionVisualizer(max_motion_display=cfg.MOTION_DETECTION.MAX_MOTION_DISPLAY)
                    
                    # Get flow and depth data
                    pred_flow = predictions['flow']  # (T, H, W, 2)
                    points = predictions['points']  # (T, H, W, 3) - world coordinates
                    depth_maps = points[..., 2]  # (T, H, W) - Z component is depth
                    
                    # Compute 3D motion for each frame transition
                    for t in range(min(pred_flow.shape[0] - 1, 6)):  # T-1 transitions, limit to 6
                        flow_t = pred_flow[t]  # (H, W, 2)
                        depth_current = depth_maps[t]  # (H, W)
                        depth_next = depth_maps[t + 1]  # (H, W)
                        
                        # Use simple intrinsics (can be made configurable)
                        H, W = flow_t.shape[:2]
                        fx = fy = W  # Simple assumption
                        cx, cy = W/2, H/2
                        intrinsics = np.array([fx, fy, cx, cy])
                        
                        # Convert flow to 3D motion
                        motion_3d = motion_detector.flow_to_3d_motion(
                            flow=flow_t,
                            depth_current=depth_current,
                            depth_next=depth_next,
                            intrinsics=intrinsics,
                            denormalize_flow=False
                        )
                        
                        # Detect dynamic pixels
                        dynamic_mask = motion_detector.detect_moving_pixels(motion_3d)
                        
                        # Save motion visualizations
                        if cfg.MOTION_DETECTION.SAVE_MOTION_VISUALIZATIONS:
                            # Save 3D motion field
                            motion_stats = motion_visualizer.save_motion_field(
                                motion_3d, f"motion_field_frame_{t}_to_{t+1}.png"
                            )
                            
                            # Save motion as HSV
                            motion_visualizer.save_motion_hsv(
                                motion_3d, f"motion_hsv_frame_{t}_to_{t+1}.png"
                            )
                            
                            # Save dynamic mask
                            mask_stats = motion_visualizer.save_dynamic_masks(
                                dynamic_mask, f"dynamic_mask_frame_{t}_to_{t+1}.png"
                            )
                            
                            print(f"🏃 Saved motion detection for frames {t}→{t+1}")
                            print(f"   Motion range: {motion_stats['min_motion']:.3f} - {motion_stats['max_motion']:.3f} m")
                            print(f"   Dynamic pixels: {mask_stats['dynamic_pixels']} / {mask_stats['total_pixels']} ({mask_stats['dynamic_ratio']:.1%})")
                        
                        # Prepare WandB images
                        if cfg.WANDB.USE_WANDB:
                            wandb_images = motion_visualizer.create_wandb_images(motion_3d, dynamic_mask, {})
                            
                            motion_images_for_wandb.extend([
                                wandb.Image(wandb_images['motion_magnitude'], caption=f"Motion Magnitude {t}→{t+1}"),
                                wandb.Image(wandb_images['dynamic_masks'], caption=f"Dynamic Mask {t}→{t+1}")
                            ])
                    
                    # Log motion detection to WandB
                    if cfg.WANDB.USE_WANDB and motion_images_for_wandb:
                        wandb.log({"visualizations/motion_detection": motion_images_for_wandb}, step=global_step)
                
                # === CAMERA TRAJECTORY VISUALIZATION ===
                camera_images_for_wandb = []
                if 'camera_poses' in predictions:
                    try:
                        # Get camera poses (T, 4, 4)
                        camera_poses = predictions['camera_poses']  # numpy array (T, 4, 4)
                        
                        # Extract camera positions (translation vectors)
                        camera_positions = camera_poses[:, :3, 3]  # (T, 3)
                        
                        # Create 3D trajectory plot
                        import matplotlib.pyplot as plt
                        from mpl_toolkits.mplot3d import Axes3D
                        
                        fig = plt.figure(figsize=(10, 8))
                        ax = fig.add_subplot(111, projection='3d')
                        
                        # Plot trajectory
                        ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                               'b-', linewidth=2, label='Camera Trajectory')
                        
                        # Mark start and end points
                        ax.scatter(camera_positions[0, 0], camera_positions[0, 1], camera_positions[0, 2], 
                                  c='green', s=100, label='Start', marker='o')
                        ax.scatter(camera_positions[-1, 0], camera_positions[-1, 1], camera_positions[-1, 2], 
                                  c='red', s=100, label='End', marker='s')
                        
                        # Add frame markers
                        for i in range(0, len(camera_positions), max(1, len(camera_positions)//5)):
                            ax.scatter(camera_positions[i, 0], camera_positions[i, 1], camera_positions[i, 2], 
                                     c='orange', s=30, alpha=0.7)
                        
                        # Labels and formatting
                        ax.set_xlabel('X (m)')
                        ax.set_ylabel('Y (m)')
                        ax.set_zlabel('Z (m)')
                        ax.set_title(f'Camera Trajectory - Step {global_step}')
                        ax.legend()
                        
                        # Equal aspect ratio for better visualization
                        max_range = np.max(np.abs(camera_positions)) * 1.1
                        ax.set_xlim([-max_range, max_range])
                        ax.set_ylim([-max_range, max_range])
                        ax.set_zlim([-max_range, max_range])
                        
                        # Save to disk (replaces previous)
                        trajectory_path = "camera_trajectory.png"
                        plt.savefig(trajectory_path, dpi=150, bbox_inches='tight')
                        print(f"💙 Saved camera trajectory to {trajectory_path}")
                        
                        # Prepare for WandB
                        if cfg.WANDB.USE_WANDB:
                            # Save to buffer and load for WandB
                            import io
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            
                            from PIL import Image as PILImage
                            pil_image = PILImage.open(buf)
                            camera_images_for_wandb.append(wandb.Image(pil_image, caption=f"Camera Trajectory Step {global_step}"))
                        
                        plt.close(fig)  # Free memory
                        
                    except Exception as e:
                        print(f"❌ Error creating camera trajectory visualization: {e}")
                
                # === LOG TO WANDB ===
                if cfg.WANDB.USE_WANDB:
                    wandb_log_dict = {}
                    
                    # Log RGB input frames
                    if rgb_images_for_wandb:
                        wandb_log_dict["visualizations/rgb_frames"] = rgb_images_for_wandb
                        print(f"🚀 Logged {len(rgb_images_for_wandb)} RGB input frames to WandB")
                    
                    
                    # Log depth images
                    if depth_images_for_wandb:
                        wandb_log_dict["visualizations/depth_maps"] = depth_images_for_wandb
                        print(f"🚀 Logged {len(depth_images_for_wandb)} depth maps to WandB")
                    
                    # Log confidence images  
                    if confidence_images_for_wandb:
                        wandb_log_dict["visualizations/confidence_maps"] = confidence_images_for_wandb
                        print(f"🚀 Logged {len(confidence_images_for_wandb)} confidence maps to WandB")
                    
                    # Log normal maps
                    if normal_images_for_wandb:
                        wandb_log_dict["visualizations/normal_maps"] = normal_images_for_wandb
                        print(f"🚀 Logged {len(normal_images_for_wandb)} normal maps to WandB")
                    
                    # Log camera trajectory
                    if camera_images_for_wandb:
                        wandb_log_dict["visualizations/camera_trajectory"] = camera_images_for_wandb
                        print(f"🚀 Logged camera trajectory to WandB")
                    
                    # Send all visualizations to WandB
                    if wandb_log_dict:
                        run.log(wandb_log_dict, step=global_step)

            epoch_loss += current_loss
            running_loss += current_loss
            loss_history.append(current_loss)
                
            # Logging
            if global_step % cfg.LOGGING.LOG_FREQ == 0 and accelerator.is_main_process:
                current_lr = scheduler.get_last_lr()[0]
                
                # TensorBoard logging
                writer.add_scalar("Loss/Train", current_loss, global_step)
                writer.add_scalar("Loss/Pi3_Loss", pi3_loss.item(), global_step)
                writer.add_scalar("Loss/Distillation_Loss", current_distillation_loss, global_step)
                writer.add_scalar("Learning_Rate", current_lr, global_step)
                
                # Weights & Biases logging
                if cfg.WANDB.USE_WANDB:
                    log_dict = {
                        "train/total_loss": current_loss,
                        "train/pi3_loss": pi3_loss.item(),
                        "train/distillation_loss": current_distillation_loss,
                        "train/point_map_loss": point_map_loss.item(),
                        "train/camera_pose_loss": camera_pose_loss.item(),
                        "train/conf_loss": conf_loss.item() if torch.is_tensor(conf_loss) else conf_loss,
                        "train/normal_loss": normal_loss.item() if torch.is_tensor(normal_loss) else normal_loss,
                        "train/segmentation_loss": segmentation_loss.item() if torch.is_tensor(segmentation_loss) else segmentation_loss,
                        "train/motion_loss": motion_loss.item() if torch.is_tensor(motion_loss) else motion_loss,
                        "train/flow_loss": flow_loss.item() if torch.is_tensor(flow_loss) else flow_loss,
                        "train/frozen_decoder_supervision_loss": frozen_decoder_loss.item() if torch.is_tensor(frozen_decoder_loss) else frozen_decoder_loss,
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch,
                        "train/best_loss": best_loss,
                        "train/step": global_step
                    }
                    # Add warmup phase indicator if using warmup
                    if warmup_steps > 0:
                        log_dict["train/warmup_phase"] = 1.0 if global_step < warmup_steps else 0.0
                    run.log(log_dict, step=global_step)
                
                postfix_dict = {
                    'loss': f'{current_loss:.6f}',
                    'pi3': f'{pi3_loss.item():.6f}',
                    'lr': f'{current_lr:.2e}',
                    'best': f'{best_loss:.6f}',
                    'val_best': f'{best_val_loss:.6f}' if val_dataloader else 'N/A'
                }
                
                # Add distillation loss to postfix if enabled
                if cfg.MODEL.USE_DISTILLED_VIT:
                    postfix_dict['distill'] = f'{current_distillation_loss:.6f}'
                
                # Add segmentation loss to postfix if segmentation head is enabled
                if cfg.MODEL.USE_SEGMENTATION_HEAD and segmentation_loss != 0.0:
                    seg_loss_val = segmentation_loss.item() if torch.is_tensor(segmentation_loss) else segmentation_loss
                    postfix_dict['seg_loss'] = f'{seg_loss_val:.6f}'
                # Add warmup indicator if we're in warmup phase
                if warmup_steps > 0 and global_step < warmup_steps:
                    postfix_dict['warmup'] = f'{global_step}/{warmup_steps}'
                progress_bar.set_postfix(postfix_dict)
            
            # Validation check
            if (val_dataloader is not None and 
                global_step % cfg.VALIDATION.VAL_FREQ == 0 and 
                global_step > 0 and 
                accelerator.is_main_process):
                
                print(f"\n🔍 Running validation at step {global_step}...")
                val_metrics = run_validation(
                    train_model, frozen_model, val_dataloader, cfg, accelerator, preprocess_image, dtype, global_step, run
                )
                
                val_loss_history.append(val_metrics['val_loss'])
                
                # Log validation metrics
                writer.add_scalar("Loss/Validation", val_metrics['val_loss'], global_step)
                writer.add_scalar("Loss/Val_Point", val_metrics['val_point_loss'], global_step)
                writer.add_scalar("Loss/Val_Camera", val_metrics['val_camera_loss'], global_step)
                writer.add_scalar("Loss/Val_Flow", val_metrics['val_flow_loss'], global_step)
                writer.add_scalar("Loss/Val_Frozen_Decoder", val_metrics['val_frozen_decoder_loss'], global_step)
                
                # Log unweighted validation losses for tracking actual performance
                writer.add_scalar("Loss/Val_Unweighted_L1_Points", val_metrics['val_unweighted_l1_points'], global_step)
                writer.add_scalar("Loss/Val_Unweighted_Pose", val_metrics['val_unweighted_pose_loss'], global_step)
                writer.add_scalar("Loss/Val_Scaled_Corrected_L1_Points", val_metrics['val_scale_corrected_l1_points'], global_step)

                
                if cfg.WANDB.USE_WANDB:
                    run.log({
                        "val/total_loss": val_metrics['val_loss'],
                        "val/scaled_point_map_loss": val_metrics['val_scale_corrected_l1_points'],
                        "val/point_map_loss": val_metrics['val_point_loss'],
                        "val/camera_pose_loss": val_metrics['val_camera_loss'],
                        "val/conf_loss": val_metrics['val_conf_loss'],
                        "val/flow_loss": val_metrics['val_flow_loss'],
                        "val/frozen_decoder_loss": val_metrics['val_frozen_decoder_loss'],
                        # Unweighted validation losses for tracking actual performance
                        "val/unweighted_l1_points": val_metrics['val_unweighted_l1_points'],
                        "val/unweighted_pose_loss": val_metrics['val_unweighted_pose_loss'],
                        "val/step": global_step
                    }, step=global_step)
                
                print(f"📊 Validation Results:")
                print(f"   Total Loss (weighted): {val_metrics['val_loss']:.6f}")
                print(f"   Point Loss (weighted): {val_metrics['val_point_loss']:.6f}")
                print(f"   Camera Loss (weighted): {val_metrics['val_camera_loss']:.6f}")
                print(f"   Conf Loss (weighted): {val_metrics['val_conf_loss']:.6f}")
                print(f"   L1 Points (unweighted): {val_metrics['val_unweighted_l1_points']:.6f}")
                print(f"   Pose Loss (unweighted): {val_metrics['val_unweighted_pose_loss']:.6f}")
                
                # Early stopping check
                if True or val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    steps_without_improvement = 0
                    
                    # Save best validation model
                    best_val_checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': accelerator.unwrap_model(train_model).state_dict(),
                        'config': cfg
                    }
                    
                    # Add distilled ViT state dict if enabled
                    if distilled_vit is not None and cfg.MODEL.USE_DISTILLED_VIT:
                        best_val_checkpoint['distilled_vit_state_dict'] = accelerator.unwrap_model(distilled_vit).state_dict()
                        print(f"📝 Including distilled ViT state dict in validation checkpoint")
                    
                    best_val_model_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, 'best_val_model.pt')
                    print(f"📝 Saving best validation model locally to: {best_val_model_path}")
                    torch.save(best_val_checkpoint, best_val_model_path)
                    print(f"💾 New best validation model saved! Val Loss: {best_val_loss:.6f}")
                    
                    # Upload best validation model to S3
                    upload_success = False
                    if cfg.OUTPUT.UPLOAD_TO_S3:
                        try:
                            s3_filename = f"{actual_run_name}_best_val_model.pt" if actual_run_name else "best_val_model.pt"
                            s3_path = f"s3://{cfg.OUTPUT.S3_BUCKET}/{cfg.OUTPUT.S3_PREFIX}/{s3_filename}"
                            save_state_dict_to_s3(best_val_checkpoint, s3_path)
                            upload_success = True
                        except Exception as e:
                            print(f"❌ Failed to upload best val model to S3: {e}")
                            upload_success = False
                    
                    if cfg.WANDB.USE_WANDB:
                        run.log({
                            "val/best_model_saved": True,
                            "val/new_best_loss": best_val_loss,
                            "val/s3_upload_success": upload_success
                        }, step=global_step)
                else:
                    steps_without_improvement += 1
                    print(f"⚠️  No validation improvement. Steps without improvement: {steps_without_improvement}/{cfg.VALIDATION.EARLY_STOPPING_PATIENCE}")
                
                # Early stopping
                if cfg.VALIDATION.EARLY_STOPPING_PATIENCE > 0 and steps_without_improvement >= cfg.VALIDATION.EARLY_STOPPING_PATIENCE:
                    print(f"🛑 Early stopping triggered after {steps_without_improvement} validation checks without improvement.")
                    if cfg.WANDB.USE_WANDB:
                        run.log({
                            "training/early_stopped": True,
                            "training/final_step": global_step,
                            "training/final_val_loss": val_metrics['val_loss']
                        }, step=global_step)
                    # break
                
            if (val_dataloader is not None and 
                global_step % cfg.VALIDATION.VAL_FREQ == 0 and 
                global_step > 0):
                accelerator.wait_for_everyone()
            
            if global_step % cfg.LOGGING.SAVE_FREQ == 0 and global_step != 0 and accelerator.is_main_process:
                # Calculate average loss over the last save_freq steps
                recent_loss = running_loss / cfg.LOGGING.SAVE_FREQ
                
                # Check if this is the best loss so far
                if True or recent_loss < best_loss:
                    best_loss = recent_loss
                    
                    # Save the best model
                    checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': accelerator.unwrap_model(train_model).state_dict(),
                        'config': cfg
                    }
                    
                    # Add distilled ViT state dict if enabled
                    if distilled_vit is not None and cfg.MODEL.USE_DISTILLED_VIT:
                        checkpoint['distilled_vit_state_dict'] = accelerator.unwrap_model(distilled_vit).state_dict()
                        print(f"📝 Including distilled ViT state dict in checkpoint")
                        
                        # Also save standalone distilled ViT checkpoint
                        distilled_vit_checkpoint = {
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': accelerator.unwrap_model(distilled_vit).state_dict(),
                            'config': cfg.MODEL.DISTILLED_VIT,
                            'teacher_model_name': cfg.MODEL.ENCODER_NAME
                        }
                        distilled_vit_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, 'best_distilled_vit.pt')
                        torch.save(distilled_vit_checkpoint, distilled_vit_path)
                        print(f"📝 Standalone distilled ViT saved to: {distilled_vit_path}")
                    
                    best_model_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, 'best_model.pt')
                    print(f"📝 Saving best training model locally to: {best_model_path}")
                    torch.save(checkpoint, best_model_path)
                    print(f"💾 New best model saved! Loss: {best_loss:.6f} at step {global_step}")
                    
                    # Upload best training model to S3
                    upload_success = False
                    if cfg.OUTPUT.UPLOAD_TO_S3:
                        try:
                            s3_filename = f"{actual_run_name}_best_model.pt" if actual_run_name else "best_model.pt"
                            s3_path = f"s3://{cfg.OUTPUT.S3_BUCKET}/{cfg.OUTPUT.S3_PREFIX}/{s3_filename}"
                            save_state_dict_to_s3(checkpoint, s3_path)
                            upload_success = True
                            
                            # Also upload standalone distilled ViT if enabled
                            if distilled_vit is not None and cfg.MODEL.USE_DISTILLED_VIT:
                                distilled_s3_filename = f"{actual_run_name}_best_distilled_vit.pt" if actual_run_name else "best_distilled_vit.pt"
                                distilled_s3_path = f"s3://{cfg.OUTPUT.S3_BUCKET}/{cfg.OUTPUT.S3_PREFIX}/{distilled_s3_filename}"
                                save_state_dict_to_s3(distilled_vit_checkpoint, distilled_s3_path)
                                print(f"☁️ Distilled ViT uploaded to: {distilled_s3_path}")
                                
                        except Exception as e:
                            print(f"❌ Failed to upload best model to S3: {e}")
                            upload_success = False
                    
                    # Log best model to wandb
                    if cfg.WANDB.USE_WANDB:
                        run.log({
                            "train/best_model_saved": True,
                            "train/new_best_loss": best_loss,
                            "train/s3_upload_success": upload_success
                        }, step=global_step)
                
                # Always save a recent checkpoint
                recent_checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': accelerator.unwrap_model(train_model).state_dict(),
                    'config': cfg
                }
                
                recent_model_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, f'checkpoint_step_{global_step}.pt')
                torch.save(recent_checkpoint, recent_model_path)
                
                # Keep only the last 3 checkpoints to save disk space
                checkpoint_files = sorted([f for f in os.listdir(cfg.OUTPUT.CHECKPOINT_DIR) if f.startswith('checkpoint_step_')])
                if len(checkpoint_files) > 3:
                    for old_checkpoint in checkpoint_files[:-3]:
                        os.remove(os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, old_checkpoint))
                
                # Reset running loss
                running_loss = 0.0
            
            # Save distilled ViT every 1000 steps
            if global_step % 1000 == 0 and global_step != 0 and accelerator.is_main_process and distilled_vit is not None and cfg.MODEL.USE_DISTILLED_VIT:
                distilled_vit_checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': accelerator.unwrap_model(distilled_vit).state_dict(),
                    'config': cfg.MODEL.DISTILLED_VIT,
                    'teacher_model_name': cfg.MODEL.ENCODER_NAME
                }
                distilled_vit_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, f'distilled_vit_step_{global_step}.pt')
                torch.save(distilled_vit_checkpoint, distilled_vit_path)
                print(f"💾 Distilled ViT saved at step {global_step}: {distilled_vit_path}")
            
            # Synchronize all processes after model saving operations
            if global_step % cfg.LOGGING.SAVE_FREQ == 0 and global_step != 0:
                accelerator.wait_for_everyone()
            
            global_step += 1
            torch.cuda.empty_cache()

        # Break out of epoch loop if early stopping was triggered
        if (cfg.VALIDATION.EARLY_STOPPING_PATIENCE > 0 and 
            steps_without_improvement >= cfg.VALIDATION.EARLY_STOPPING_PATIENCE):
            # break
            pass

        # End of epoch logging
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        if accelerator.is_main_process:
            print(f"📊 Epoch {epoch+1} completed - Avg Loss: {avg_epoch_loss:.6f}")
            if val_dataloader is not None:
                print(f"   Best Val Loss so far: {best_val_loss:.6f}")
                print(f"   Steps without improvement: {steps_without_improvement}")
            
            # Log epoch metrics
            if cfg.WANDB.USE_WANDB:
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
                run.log(epoch_log, step=global_step)
        
    # Training complete - cleanup
    writer.close()
    
    if cfg.WANDB.USE_WANDB:
        run.finish()
        print("📊 Weights & Biases run finished!")
    
    # Save final pseudo_gt data if we have any
    if accelerator.is_main_process and len(pseudo_gt_storage['step']) > 0:
        print(f"💾 Saving final pseudo_gt data with {len(pseudo_gt_storage['step'])} steps...")
        torch.save(pseudo_gt_storage, pseudo_gt_save_path)
        print(f"✅ Pseudo GT data saved to: {pseudo_gt_save_path}")
    
    # Final summary
    if accelerator.is_main_process:
        print("🎉 Training complete!")
        print(f"📊 Best training loss achieved: {best_loss:.6f}")
        if val_dataloader is not None:
            print(f"📊 Best validation loss achieved: {best_val_loss:.6f}")
            print(f"💾 Best validation model saved at: {os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, 'best_val_model.pt')}")
            if steps_without_improvement >= cfg.VALIDATION.EARLY_STOPPING_PATIENCE:
                print(f"🛑 Training stopped early due to no validation improvement for {steps_without_improvement} checks")
        print(f"💾 Best training model saved at: {os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, 'best_model.pt')}")
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
            'early_stopped': steps_without_improvement >= cfg.VALIDATION.EARLY_STOPPING_PATIENCE,
            'training_config': cfg
        }
        summary_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)  # default=str handles non-serializable objects
        print(f"📋 Training summary saved at: {summary_path}")
        
        # Upload final training summary to S3
        if cfg.OUTPUT.UPLOAD_TO_S3:
            upload_file_to_s3(
                summary_path,
                s3_bucket=cfg.OUTPUT.S3_BUCKET,
                s3_prefix=cfg.OUTPUT.S3_PREFIX,
                wandb_run_name=actual_run_name
            )


def main():
    """Main function with argument parsing for configuration."""
    parser = argparse.ArgumentParser(description="Pi3 Cluster Training with YACS Configuration")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file (optional, defaults to config.yaml)"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--use_accelerate", 
        action='store_true',
        default=True,
        help="Use accelerate launcher (default: True)"
    )


    args = parser.parse_args()
    
    # Load and update configuration
    cfg = get_cfg_defaults()
    cfg = update_config(cfg, args)
    
    print("==> Final configuration:")
    print(cfg)
    train_model()

if __name__ == "__main__":
    main()