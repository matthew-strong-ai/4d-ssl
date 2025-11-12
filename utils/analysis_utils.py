#!/usr/bin/env python3
"""
Analysis utilities for object tracking and dynamics.

This module contains functions for analyzing object movement, dynamics,
and other scene understanding tasks during training.
"""

import numpy as np
import torch


def analyze_object_dynamics(gsam2_results, pred_tracks, pred_visibility, point_maps, composite_masks=None, verbose=True, motion_threshold=0.1, smooth_sigma=2.0):
    """
    Analyze object dynamics using GSAM2 masks, CoTracker2 tracks, and 3D point maps.
    
    Args:
        gsam2_results: Results from GSAM2 process_frames
        pred_tracks: CoTracker2 tracks [1, T, num_points, 2] where last dim is (x, y)
        pred_visibility: Track visibility [1, T, num_points] - bool if point is visible
        point_maps: 3D point maps [T, H, W, 3]
        composite_masks: List of composite masks for each frame [T, H, W]
        verbose: Whether to print progress messages and object analysis results
        motion_threshold: Minimum 3D motion magnitude (in meters) to consider an object as dynamic
        smooth_sigma: Gaussian smoothing sigma for per-object motion smoothing (default: 2.0)
    
    Returns:
        tuple: (dynamic_analysis, motion_maps, dynamic_masks)
            - dynamic_analysis: dict with analysis results for each object
            - motion_maps: list of 3D motion maps for each frame transition [T-1, H, W, 3]
            - dynamic_masks: list of binary masks indicating moving pixels [T, H, W] (0=static, 1=dynamic)
    """

    _, T, num_points, _ = pred_tracks.shape
    H, W = point_maps.shape[1], point_maps.shape[2]
    dynamic_analysis = {}
    motion_maps = []
    dynamic_masks = []
    
    # Initialize dynamic masks for each frame
    for t in range(T):
        dynamic_masks.append(np.zeros((H, W), dtype=np.uint8))
    
    # Compute 3D motion maps for each frame transition
    for t in range(T - 1):
        motion_map = compute_3d_motion_field(pred_tracks, pred_visibility, point_maps, t, H, W, composite_masks, smooth_sigma=smooth_sigma)
        motion_maps.append(motion_map)
    
    # Get first frame masks and find which tracks belong to each object
    for obj_id, mask in gsam2_results['masks'][0].items():
        # Handle different mask dimensions
        mask_2d = mask[0]  # Assuming mask shape is [1, H, W] or [H, W]

        coords = np.where(mask_2d)
        if len(coords) == 2:
            y_coords, x_coords = coords
        elif len(coords) == 3:
            # 3D mask - take first channel or flatten
            _, y_coords, x_coords = coords
        else:
            continue
            
        if len(y_coords) == 0:
            continue
            
        # Find tracks that start within this mask
        object_tracks = []
        for point_idx in range(num_points):
            # Get track position in first frame
            x, y = pred_tracks[0, 0, point_idx, :]  # [2] -> (x, y)
            x, y = int(x), int(y)
            
            # Check if this track starts within the mask
            if 0 <= x < mask_2d.shape[1] and 0 <= y < mask_2d.shape[0] and mask_2d[y, x]:
                object_tracks.append(point_idx)
        
        if len(object_tracks) == 0:
            continue
        
        # Analyze movement for each track in this object
        track_movements = []
        
        for track_idx in object_tracks:
            # Extract 2D trajectory for this track: [T, 2]
            trajectory_2d = pred_tracks[0, :, track_idx, :]  # [T, 2] -> (x, y) positions
            
            # Convert to 3D using point maps
            trajectory_3d = []
            valid_points = 0
            
            for t in range(T):
                # Check if point is visible at this frame
                is_visible = pred_visibility[0, t, track_idx]
                if not is_visible:
                    continue
                    
                x, y = trajectory_2d[t]  # (x, y)
                x, y = int(x), int(y)
                
                # Get 3D point from point map (if within bounds)
                if 0 <= x < point_maps.shape[2] and 0 <= y < point_maps.shape[1]:
                    point_3d = point_maps[t, y, x, :]  # [3] -> (X, Y, Z)
                    trajectory_3d.append(point_3d)
                    valid_points += 1
            
            if valid_points >= 2:  # Need at least 2 points to compute movement
                trajectory_3d = np.array(trajectory_3d)  # [valid_points, 3]
                
                # Compute total distance traveled by this point
                distances = np.linalg.norm(np.diff(trajectory_3d, axis=0), axis=1)
                total_distance = np.sum(distances)  # Total path length
                
                track_movements.append(total_distance)
        
        if len(track_movements) > 0:
            # Use median total distance as measure of object dynamics
            median_movement = np.median(track_movements)
            mean_movement = np.mean(track_movements) 
            
            # Threshold for dynamic vs static (can be tuned)
            is_dynamic = median_movement > motion_threshold
            
            dynamic_analysis[obj_id] = {
                'median_total_movement': median_movement,
                'mean_total_movement': mean_movement,
                'num_valid_tracks': len(track_movements),
                'is_dynamic': is_dynamic,
                'track_movements': track_movements
            }
            
            # If object is dynamic, mark all its pixels as dynamic in all frames
            if is_dynamic:
                for t in range(T):
                    # Get object mask for this frame
                    if t < len(gsam2_results['masks']) and obj_id in gsam2_results['masks'][t]:
                        obj_mask = gsam2_results['masks'][t][obj_id]
                        if obj_mask.ndim == 3:
                            obj_mask = obj_mask[0]  # Remove first dimension if present
                        # Mark all pixels in this object's mask as dynamic
                        dynamic_masks[t][obj_mask > 0] = 1
            
            # Use matching icons from visualization
            icon = "▶️" if is_dynamic else "⏸️"  # Arrow for dynamic, pause/stop for static
            status = "DYNAMIC" if is_dynamic else "STATIC"
            if verbose:
                print(f"🎯 Object {obj_id}: {icon} {status} "
                      f"(median movement: {median_movement:.3f}m, {len(track_movements)} tracks)")
    
    return dynamic_analysis, motion_maps, dynamic_masks


def compute_3d_motion_field(pred_tracks, pred_visibility, point_maps, t, H, W, composite_masks=None, smooth_sigma=2.0):
    """
    Compute 3D motion field for frame transition t -> t+1 using CoTracker points.
    Motion is only computed for segmented object regions with per-object smoothing.
    
    Args:
        pred_tracks: CoTracker2 tracks [1, T, num_points, 2]
        pred_visibility: Track visibility [1, T, num_points]
        point_maps: 3D point maps [T, H, W, 3]
        t: Current frame index
        H, W: Height and width of the motion field
        composite_masks: List of composite masks for each frame [T, H, W]
        smooth_sigma: Gaussian smoothing sigma for per-object smoothing (default: 2.0)
        
    Returns:
        motion_field: 3D motion field [H, W, 3] with displacement vectors (zero outside objects)
    """
    from scipy.spatial.distance import cdist
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    
    _, T, num_points, _ = pred_tracks.shape
    motion_field = np.zeros((H, W, 3), dtype=np.float32)
    
    # Get valid tracks for both frames t and t+1
    valid_points_t = []
    valid_points_t1 = []
    motion_vectors_3d = []
    
    for point_idx in range(num_points):
        # Check if point is visible in both frames
        vis_t = pred_visibility[0, t, point_idx]
        vis_t1 = pred_visibility[0, t + 1, point_idx] if t + 1 < T else False
        
        if not (vis_t and vis_t1):
            continue
            
        # Get 2D positions
        x_t, y_t = pred_tracks[0, t, point_idx, :]
        x_t1, y_t1 = pred_tracks[0, t + 1, point_idx, :]
        
        # Convert to integer coordinates for indexing
        x_t, y_t = int(x_t), int(y_t)
        x_t1, y_t1 = int(x_t1), int(y_t1)
        
        # Check bounds for both frames
        if (0 <= x_t < W and 0 <= y_t < H and 
            0 <= x_t1 < W and 0 <= y_t1 < H):
            
            # Get 3D points from point maps
            point_3d_t = point_maps[t, y_t, x_t, :]
            point_3d_t1 = point_maps[t + 1, y_t1, x_t1, :]
            
            # Compute 3D motion vector
            motion_3d = point_3d_t1 - point_3d_t
            
            valid_points_t.append([x_t, y_t])
            motion_vectors_3d.append(motion_3d)
    
    if len(valid_points_t) == 0:
        return motion_field
    
    valid_points_t = np.array(valid_points_t)
    motion_vectors_3d = np.array(motion_vectors_3d)
    
    # Create grid coordinates for interpolation
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    # Interpolate motion for each dimension (X, Y, Z)
    for dim in range(3):
        if len(valid_points_t) >= 3:  # Need at least 3 points for proper interpolation
            # Use griddata for smooth interpolation
            interpolated_motion = griddata(
                valid_points_t, motion_vectors_3d[:, dim], 
                grid_points, method='linear', fill_value=0.0
            )
        else:
            # For very few points, use nearest neighbor
            interpolated_motion = griddata(
                valid_points_t, motion_vectors_3d[:, dim], 
                grid_points, method='nearest', fill_value=0.0
            )
        
        motion_field[:, :, dim] = interpolated_motion.reshape(H, W)
    
    # Apply per-object smoothing and masking
    if composite_masks is not None and len(composite_masks) > t:
        # Create smoothed motion field per object
        smoothed_motion_field = np.zeros_like(motion_field)
        
        # Get unique object IDs (excluding background which is 0)
        object_ids = np.unique(composite_masks[t])
        object_ids = object_ids[object_ids > 0]
        
        for obj_id in object_ids:
            # Get mask for this specific object
            obj_mask = (composite_masks[t] == obj_id)
            
            if np.sum(obj_mask) < 10:  # Skip very small objects
                continue
            
            # Create dilated mask for smoothing (to avoid edge artifacts)
            from scipy.ndimage import binary_dilation
            dilated_mask = binary_dilation(obj_mask, iterations=int(smooth_sigma * 2))
            
            # Smooth motion within this object's region
            for dim in range(3):
                # Extract motion for this object (with dilated region)
                obj_motion = motion_field[:, :, dim] * dilated_mask
                
                # Apply Gaussian smoothing
                if smooth_sigma > 0:
                    smoothed_obj_motion = gaussian_filter(obj_motion, sigma=smooth_sigma)
                else:
                    smoothed_obj_motion = obj_motion
                
                # Apply back only to the original object mask (not dilated)
                smoothed_motion_field[:, :, dim] += smoothed_obj_motion * obj_mask
        
        motion_field = smoothed_motion_field
    else:
        # If no masks provided, just apply smoothing to the whole field
        if smooth_sigma > 0:
            for dim in range(3):
                motion_field[:, :, dim] = gaussian_filter(motion_field[:, :, dim], sigma=smooth_sigma)
    
    return motion_field


def compute_optical_flow_magnitude(point_maps):
    """
    Compute optical flow magnitude from sequential 3D point maps.
    
    Args:
        point_maps: 3D point maps [T, H, W, 3]
        
    Returns:
        flow_magnitudes: Array of flow magnitudes [T-1, H, W]
    """
    T, H, W, _ = point_maps.shape
    flow_magnitudes = []
    
    for t in range(T - 1):
        # Compute 3D displacement between consecutive frames
        displacement = point_maps[t + 1] - point_maps[t]  # [H, W, 3]
        
        # Compute magnitude of 3D displacement
        magnitude = np.linalg.norm(displacement, axis=-1)  # [H, W]
        flow_magnitudes.append(magnitude)
    
    return np.array(flow_magnitudes)  # [T-1, H, W]


def analyze_scene_motion(point_maps, threshold=0.1):
    """
    Analyze overall scene motion patterns.
    
    Args:
        point_maps: 3D point maps [T, H, W, 3]
        threshold: Motion threshold for considering a pixel as moving
        
    Returns:
        dict: Scene motion analysis results
    """
    flow_magnitudes = compute_optical_flow_magnitude(point_maps)
    
    # Compute statistics
    mean_motion = np.mean(flow_magnitudes)
    std_motion = np.std(flow_magnitudes) 
    max_motion = np.max(flow_magnitudes)
    
    # Count moving pixels
    moving_pixels = flow_magnitudes > threshold
    motion_density = np.mean(moving_pixels)  # Fraction of pixels in motion
    
    # Temporal motion pattern
    temporal_motion = np.mean(flow_magnitudes, axis=(1, 2))  # [T-1]
    
    analysis = {
        'mean_motion': mean_motion,
        'std_motion': std_motion,
        'max_motion': max_motion,
        'motion_density': motion_density,
        'temporal_motion': temporal_motion,
        'is_dynamic_scene': motion_density > 0.1  # Scene is dynamic if >10% pixels moving
    }
    
    return analysis


def compute_depth_consistency(point_maps):
    """
    Compute temporal consistency of depth estimates.
    
    Args:
        point_maps: 3D point maps [T, H, W, 3]
        
    Returns:
        dict: Depth consistency metrics
    """
    depths = point_maps[..., 2]  # [T, H, W] - Z component
    T, H, W = depths.shape
    
    # Compute frame-to-frame depth differences
    depth_diffs = []
    for t in range(T - 1):
        diff = np.abs(depths[t + 1] - depths[t])
        depth_diffs.append(diff)
    
    depth_diffs = np.array(depth_diffs)  # [T-1, H, W]
    
    # Statistics
    mean_depth_diff = np.mean(depth_diffs)
    std_depth_diff = np.std(depth_diffs)
    max_depth_diff = np.max(depth_diffs)
    
    # Stability metric (lower is more stable)
    stability_score = mean_depth_diff / (np.mean(depths) + 1e-8)
    
    return {
        'mean_depth_difference': mean_depth_diff,
        'std_depth_difference': std_depth_diff,
        'max_depth_difference': max_depth_diff,
        'stability_score': stability_score,
        'is_stable': stability_score < 0.05  # Threshold for stable depth
    }


def track_object_trajectories(pred_tracks, pred_visibility, object_masks):
    """
    Track object trajectories over time using CoTracker results.
    
    Args:
        pred_tracks: CoTracker2 tracks [1, T, num_points, 2]
        pred_visibility: Track visibility [1, T, num_points]
        object_masks: Object masks for each frame {frame_idx: {obj_id: mask}}
        
    Returns:
        dict: Object trajectory information
    """
    _, T, num_points, _ = pred_tracks.shape
    object_trajectories = {}
    
    # For each object in the first frame, find associated tracks
    for obj_id, mask in object_masks[0].items():
        mask_2d = mask[0] if mask.ndim == 3 else mask
        
        # Find tracks that start within this object
        object_tracks = []
        coords = np.where(mask_2d)
        if len(coords) == 2:
            y_coords, x_coords = coords
        else:
            continue
            
        if len(y_coords) == 0:
            continue
        
        for point_idx in range(num_points):
            x, y = pred_tracks[0, 0, point_idx, :]
            x, y = int(x), int(y)
            
            if 0 <= x < mask_2d.shape[1] and 0 <= y < mask_2d.shape[0] and mask_2d[y, x]:
                object_tracks.append(point_idx)
        
        if len(object_tracks) == 0:
            continue
        
        # Extract trajectories for this object
        trajectories = []
        for track_idx in object_tracks:
            trajectory = []
            for t in range(T):
                if pred_visibility[0, t, track_idx]:
                    x, y = pred_tracks[0, t, track_idx, :]
                    trajectory.append([float(x), float(y)])
                else:
                    trajectory.append(None)  # Invisible point
            trajectories.append(trajectory)
        
        object_trajectories[obj_id] = {
            'track_indices': object_tracks,
            'trajectories': trajectories,
            'num_tracks': len(object_tracks)
        }
    
    return object_trajectories