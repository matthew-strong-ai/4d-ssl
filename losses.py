"""
File for losses for Autonomy SSL.

"""
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

import math

from alignment import align_points_scale
from geometry import depth_edge, se3_inverse


def normalize_pred_gt(pred, gt):
    """
    Normalize predicted and ground truth points and camera poses by the average distance of points from origin.
    """
    local_points = pred['local_points']
    camera_poses = pred['camera_poses']
    B, N, H, W, _ = local_points.shape
    masks = torch.ones(B, N, H, W, dtype=torch.bool, device=local_points.device)

    # normalize predict points
    all_pts = local_points.clone()
    all_pts[~masks] = 0
    all_pts = all_pts.reshape(B, N, -1, 3)
    all_dis = all_pts.norm(dim=-1)
    norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
    local_points  = local_points / norm_factor[..., None, None, None, None]

    if 'global_points' in pred and pred['global_points'] is not None:
        pred['global_points'] /= norm_factor[..., None, None, None, None]

    camera_poses_normalized = camera_poses.clone()
    camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

    pred['local_points'] = local_points
    pred['camera_poses'] = camera_poses_normalized

    # normalize ground truth with its own norm factor
    if 'local_points' in gt:
        gt_local_points = gt['local_points']
        gt_all_pts = gt_local_points.clone()
        gt_all_pts[~masks] = 0
        gt_all_pts = gt_all_pts.reshape(B, N, -1, 3)
        gt_all_dis = gt_all_pts.norm(dim=-1)
        gt_norm_factor = gt_all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        gt['local_points'] = gt_local_points / gt_norm_factor[..., None, None, None, None]

        gt_camera_poses = gt['camera_poses'].clone()
        gt_camera_poses[..., :3, 3] /= gt_norm_factor.view(B, 1, 1)
        gt['camera_poses'] = gt_camera_poses

    return pred, gt


def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))

def rot_ang_loss(R, Rgt, eps=1e-6, reduction='mean'):
    """
    Args:
        R: estimated rotation matrix [B, 3, 3]
        Rgt: ground-truth rotation matrix [B, 3, 3]
        reduction: 'mean' or 'none'
    Returns:  
        R_err: rotation angular error 
    """
    residual = torch.matmul(R.transpose(1, 2), Rgt)
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
    if reduction == 'mean':
        return R_err.mean()         # [0, 3.14]
    else:
        return R_err               # [B] shape



def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)


class Pi3Losses:
    """ Pi3 losses for point cloud and camera loss """


    def get_sky_mask(segformer, img):
        with torch.no_grad():
            output = segformer.inference_(img)
            output = output == 2
        return output



    @staticmethod
    def pi3_loss(predictions, gt, epsilon=0.1, m_frames=3, future_frame_weight=2.0, gradient_weight=0.1, 
                 normal_loss_weight=0.0, detection_targets=None, detection_loss_weight=0.0, segformer=None, images=None):
        """
        Pi3 loss between prediction and gt dict.
        Assumes predictions and gt have same shapes.
        
        Args:
            predictions: Dict containing predicted points, camera poses, etc.
            gt: Dict containing ground truth points, camera poses, etc.
            epsilon: Threshold for confidence loss
            m_frames: Number of current/input frames (first m frames)
            future_frame_weight: Weight multiplier for future frame supervision
            gradient_weight: Weight for gradient comparison term in confidence loss
            normal_loss_weight: Weight for normal loss (0.0 disables it)
            detection_targets: Detection targets from GroundingDINO (optional)
            detection_loss_weight: Weight for detection loss (0.0 disables it)
        """
        sky_mask = None
        if False and segformer is not None and images is not None:
            # combine B and N dimensions for segmentation
            B, N, C, H, W = images.shape
            images_reshaped = images.view(B * N, C, H, W)
            sky_mask = Pi3Losses.get_sky_mask(segformer, images_reshaped).reshape(B, N, H, W)  # [B, N, H, W]


        # according to the authors, the loss is performed on the *local* point maps
        # predictions, gt = normalize_pred_gt(predictions, gt)

        pred_point_maps = predictions["local_points"]
        gt_point_maps = gt["local_points"]

        point_map_loss, scale_factor, aligned_local_pts = PointCloudLosses.official_pi3_point_loss(
            pred_point_maps, gt_point_maps, m_frames=m_frames, future_frame_weight=future_frame_weight, sky_mask=sky_mask
        )

        pred_camera_poses = predictions["camera_poses"]
        gt_camera_poses = gt["camera_poses"]
        camera_pose_loss = CameraPoseLosses.official_pi3_camera_pose_loss(pred_camera_poses, gt_camera_poses, scale_factor, m_frames=m_frames, future_frame_weight=future_frame_weight)

        confidence_loss = 0.0
        if "conf" in predictions:
            pred_conf_maps = predictions["conf"]
            # apply sky mask to 
            confidence_loss = ConfidenceLosses.confidence_loss(pred_conf_maps, pred_point_maps, gt_point_maps, epsilon, gradient_weight,
                                                               sky_mask=sky_mask, m_frames=m_frames, future_frame_weight=future_frame_weight)

        # compute normal loss if weight > 0
        normal_loss = 0.0
        if normal_loss_weight > 0:
            pred_points = aligned_local_pts
            gt_points = gt["local_points"]
            # Create mask of all 1s for normal loss
            B, N, H, W, _ = pred_points.shape
            mask = torch.ones(B, N, H, W, dtype=torch.bool, device=pred_points.device)
            normal_loss = NormalLosses.pi3_official_normal_loss(pred_points, gt_points, mask)

        # compute segmentation loss if segmentation targets are provided
        segmentation_loss_val = 0.0
        if "segmentation" in predictions and "segmentation" in gt:
            pred_segmentation = predictions["segmentation"]  # [B, T, H, W, 4] for multi-class
            gt_segmentation = gt["segmentation"]  # Should be [B, T, H, W, 1] or broadcastable
            segmentation_loss_val = SegmentationLosses.segmentation_bce_loss(pred_segmentation, gt_segmentation, m_frames=m_frames, future_frame_weight=future_frame_weight)
            
        # compute motion loss if motion targets are provided
        motion_loss_val = 0.0
        if "motion" in predictions and "motion" in gt:
            pred_motion = predictions["motion"]  # [B, T, H, W, 3] with 3D motion vectors
            gt_motion = gt["motion"]  # Should be [B, T, H, W, 3] or broadcastable
            # crop up to T of 5
            pred_motion = pred_motion[:, :5]
            motion_loss_val = MotionLosses.motion_smooth_l1_loss(pred_motion, gt_motion)


        # compute frozen decoder supervision loss if features are available
        frozen_decoder_loss_val = 0.0
        if "features" in gt and "all_decoder_features" in predictions:
            frozen_decoder_loss_val = FrozenDecoderSupervision.decoder_feature_loss(predictions, gt, m_frames=m_frames)

        # compute total loss
        return point_map_loss, camera_pose_loss, confidence_loss, normal_loss, segmentation_loss_val, motion_loss_val, frozen_decoder_loss_val

    @staticmethod
    def pi3_loss_with_confidence_weighting(predictions, gt, epsilon=0.1, m_frames=3, future_frame_weight=2.0, 
                                         gamma=1.0, alpha=0.1, use_conf_weighted_points=True, gradient_weight=0.1, normal_loss_weight=0.0, detection_targets=None, detection_loss_weight=0.0, segformer=None, images=None):
        """
        Pi3 loss with optional confidence-weighted point loss.
        
        Args:
            predictions: Dict containing predicted points, camera poses, conf, etc.
            gt: Dict containing ground truth points, camera poses, etc.
            epsilon: Threshold for confidence loss
            m_frames: Number of current/input frames (first m frames)
            future_frame_weight: Weight multiplier for future frame supervision
            gamma: Weight for confidence-weighted reconstruction loss
            alpha: Weight for confidence regularization term
            use_conf_weighted_points: If True, use confidence-weighted point loss instead of scale-invariant loss
            gradient_weight: Weight for gradient comparison term in confidence loss
            normal_loss_weight: Weight for normal loss (0.0 disables it)
            detection_targets: Detection targets from GroundingDINO (optional)
            detection_loss_weight: Weight for detection loss (0.0 disables it)
            segformer: Segformer model for sky detection (optional)
            images: Input images for sky detection (optional)
        """
        
        # Generate sky mask if segformer and images are provided
        sky_mask = None
        if False and segformer is not None and images is not None:
            B, N, C, H, W = images.shape
            images_reshaped = images.view(B * N, C, H, W)
            sky_mask = Pi3Losses.get_sky_mask(segformer, images_reshaped).reshape(B, N, H, W)  # [B, N, H, W]

        pred_point_maps = predictions["local_points"]
        gt_point_maps = gt["local_points"]

        # Use official pi3 point loss which supports sky masking
        point_map_loss, scale_factor, aligned_local_pts = PointCloudLosses.official_pi3_point_loss(
            pred_point_maps, gt_point_maps, m_frames=m_frames, future_frame_weight=future_frame_weight, sky_mask=sky_mask
        )
        
        # Apply confidence weighting if requested and confidence is available
        if use_conf_weighted_points and "conf" in predictions:
            pred_conf_maps = predictions["conf"]
            # Apply confidence weighting to the already computed point loss
            # This is a simplified approach - for full confidence weighting, we'd need to modify the loss computation
            conf_weight = pred_conf_maps.mean()  # Simple confidence weighting
            point_map_loss = point_map_loss * (gamma * conf_weight + alpha * (1 - conf_weight))

        # compute pi3 camera pose loss on all frames using the computed scale factor
        pred_camera_poses = predictions["camera_poses"]
        gt_camera_poses = gt["camera_poses"]
        camera_pose_loss = CameraPoseLosses.official_pi3_camera_pose_loss(pred_camera_poses, gt_camera_poses, scale_factor, m_frames=m_frames, future_frame_weight=future_frame_weight)

        # compute confidence loss if confidence maps are provided and not using conf-weighted points
        confidence_loss = 0.0
        if "conf" in predictions and not use_conf_weighted_points:
            pred_conf_maps = predictions["conf"]
            confidence_loss = ConfidenceLosses.confidence_loss(pred_conf_maps, pred_point_maps, gt_point_maps, epsilon, gradient_weight,
                                                              sky_mask=sky_mask, m_frames=m_frames, future_frame_weight=future_frame_weight)

        # compute normal loss if weight > 0
        normal_loss = 0.0
        if normal_loss_weight > 0 and "points" in predictions:
            pred_points = predictions["points"]
            gt_points = gt["points"]
            # Use sky mask if available, otherwise use all pixels
            if sky_mask is not None:
                # Resize sky mask to match point dimensions if needed
                B, N, H, W, _ = pred_points.shape
                if sky_mask.shape[-2:] != (H, W):
                    mask = torch.nn.functional.interpolate(
                        sky_mask.float().unsqueeze(1), size=(H, W), mode='nearest'
                    ).squeeze(1).bool()
                    mask = ~mask  # Invert to get non-sky pixels
                else:
                    mask = ~sky_mask  # Non-sky pixels
            else:
                B, N, H, W, _ = pred_points.shape
                mask = torch.ones(B, N, H, W, dtype=torch.bool, device=pred_points.device)
            normal_loss = NormalLosses.pi3_official_normal_loss(pred_points, gt_points, mask)

        # compute detection loss if weight > 0 and targets provided
        detection_loss_val = 0.0
        if detection_loss_weight > 0 and "detections" in predictions:
            from detection_utils import detection_loss
            pred_detections = predictions["detections"]
            
            # Get detection architecture from predictions (if available)
            detection_architecture = getattr(predictions, 'detection_architecture', 'dense')
            if isinstance(pred_detections, dict):
                detection_architecture = 'detr'  # DETR outputs are dicts
            
            detection_loss_val, _, _ = detection_loss(
                pred_detections, 
                target_detections=detection_targets,
                detection_architecture=detection_architecture
            )

        # compute segmentation loss if segmentation targets are provided
        segmentation_loss_val = 0.0
        if "segmentation" in predictions and "segmentation" in gt:
            pred_segmentation = predictions["segmentation"]  # [B, T, H, W, 4] for multi-class
            gt_segmentation = gt["segmentation"]  # Should be [B, T, H, W, 1] or broadcastable
            segmentation_loss_val = SegmentationLosses.segmentation_bce_loss(pred_segmentation, gt_segmentation, m_frames=m_frames, future_frame_weight=future_frame_weight)

        # compute motion loss if motion targets are provided
        motion_loss_val = 0.0
        if "motion" in predictions and "motion" in gt:
            pred_motion = predictions["motion"]  # [B, T, H, W, 3] with 3D motion vectors
            gt_motion = gt["motion"]  # Should be [B, T, H, W, 3] or broadcastable
            motion_loss_val = MotionLosses.motion_smooth_l1_loss(pred_motion, gt_motion)

        # compute frozen decoder supervision loss if features are available
        frozen_decoder_loss_val = 0.0
        if "features" in gt and "all_decoder_features" in predictions:
            frozen_decoder_loss_val = FrozenDecoderSupervision.decoder_feature_loss(predictions, gt, m_frames=m_frames)

        return point_map_loss, camera_pose_loss, confidence_loss, normal_loss, detection_loss_val, segmentation_loss_val, motion_loss_val, frozen_decoder_loss_val


class ConfidenceLosses:
    """ confidence losses """
    
    @staticmethod
    def compute_depth_gradients(depth_maps):
        """
        Compute spatial gradients of depth maps.
        
        Args:
            depth_maps: [B, H, W] or [B, T, H, W] depth values (z-component of points)
            
        Returns:
            grad_x, grad_y: Gradients in x and y directions
        """
        if depth_maps.dim() == 3:
            # [B, H, W] case
            grad_x = depth_maps[:, :, 1:] - depth_maps[:, :, :-1]  # [B, H, W-1]
            grad_y = depth_maps[:, 1:, :] - depth_maps[:, :-1, :]  # [B, H-1, W]
        elif depth_maps.dim() == 4:
            # [B, T, H, W] case
            grad_x = depth_maps[:, :, :, 1:] - depth_maps[:, :, :, :-1]  # [B, T, H, W-1]
            grad_y = depth_maps[:, :, 1:, :] - depth_maps[:, :, :-1, :]  # [B, T, H-1, W]
        else:
            raise ValueError(f"Expected 3D or 4D depth maps, got {depth_maps.dim()}D")
            
        return grad_x, grad_y
    
    @staticmethod
    def confidence_loss(pred_conf_maps, pred_points, gt_points, epsilon=0.1, gradient_weight=0.1,
                        sky_mask=None, m_frames=3, future_frame_weight=1.0):
        """
        Binary Cross-Entropy (BCE) confidence loss with scale-invariant L1 error and gradient term.
        Includes gradient comparison term: ||Σ D_i ⊙ (∇D̂_i − ∇D_i)|| weighted by confidence.
        
        Args:
            pred_conf_maps: [B, N, H, W, 1] or [B, N, H, W] or [B, H, W] or [B, N] predicted confidence maps
            pred_points: [B, N, H, W, 3] or [B, H, W, 3] or [B, N, 3] predicted 3D points
            gt_points: [B, N, H, W, 3] or [B, H, W, 3] or [B, N, 3] ground truth 3D points
            epsilon: threshold for L1 reconstruction error
            gradient_weight: weight for the gradient comparison term
            sky_mask: optional sky mask to force confidence to 0 in sky regions
            m_frames: number of current frames (first m frames)
            future_frame_weight: weight multiplier for future frame losses
        
        Returns:
            BCE loss between predicted confidence and ground truth targets plus gradient term
        """
        # Squeeze last dimension if it's 1 (handle [B, H, W, 1] case)
        if pred_conf_maps.dim() > 3 and pred_conf_maps.shape[-1] == 1:
            pred_conf_maps = pred_conf_maps.squeeze(-1)  # [B, H, W, 1] -> [B, H, W]
        
        # Compute optimal scale factor for scale-invariant error
        scale_factor = PointCloudLosses.optimal_scale_factor(pred_points, gt_points)
        
        # Apply scale factor to predicted points
        scaled_pred_points = scale_factor * pred_points
        
        # Compute L1 reconstruction error per point with scale factor applied
        l1_error = torch.norm(scaled_pred_points - gt_points, p=1, dim=-1)  # [B, H, W] or [B, N]
        
        # Create ground truth confidence targets
        # Target is 1 if L1 error < epsilon, 0 otherwise
        gt_confidence = (l1_error < epsilon).float()
        
        # Apply BCE loss with future frame weighting
        # Check if we have temporal dimension (N frames)
        has_temporal = pred_conf_maps.dim() == 4 and pred_points.shape[1] > 1
        
        if has_temporal and future_frame_weight != 1.0:
            # Create frame weights
            N = pred_conf_maps.shape[1]
            frame_weights = torch.ones(N, device=pred_conf_maps.device, dtype=pred_conf_maps.dtype)
            frame_weights[m_frames:] = future_frame_weight
            
            # Compute per-element BCE loss
            bce_per_element = F.binary_cross_entropy_with_logits(pred_conf_maps, gt_confidence, reduction='none')
            
            # Apply frame weights: reshape weights to [1, N, 1, 1] for broadcasting
            frame_weights = frame_weights.view(1, N, 1, 1)
            weighted_bce = bce_per_element * frame_weights
            bce_loss = weighted_bce.mean()
        else:
            bce_loss = F.binary_cross_entropy_with_logits(pred_conf_maps, gt_confidence)

        # sky mask loss if provided.
        sky_mask_loss = 0
        if sky_mask is not None:
            sky_mask_loss = F.binary_cross_entropy_with_logits(pred_conf_maps[sky_mask], torch.zeros_like(pred_conf_maps[sky_mask]))

        # Add gradient comparison term: ||Σ D_i ⊙ (∇D̂_i − ∇D_i)||
        gradient_loss = 0.0
        if gradient_weight > 0.0 and pred_points.dim() >= 3:
            # Extract depth maps (z-component)
            pred_depth = scaled_pred_points[..., 2]  # [B, H, W] or [B, T, H, W]
            gt_depth = gt_points[..., 2]  # [B, H, W] or [B, T, H, W]
            
            # Compute gradients
            pred_grad_x, pred_grad_y = ConfidenceLosses.compute_depth_gradients(pred_depth)
            gt_grad_x, gt_grad_y = ConfidenceLosses.compute_depth_gradients(gt_depth)
            
            # Compute gradient differences
            grad_diff_x = pred_grad_x - gt_grad_x  # [B, H, W-1] or [B, T, H, W-1]
            grad_diff_y = pred_grad_y - gt_grad_y  # [B, H-1, W] or [B, T, H-1, W]
            
            # Get confidence weights for gradient locations
            if pred_depth.dim() == 3:
                # [B, H, W] case
                conf_x = torch.sigmoid(pred_conf_maps[:, :, :-1])  # [B, H, W-1]
                conf_y = torch.sigmoid(pred_conf_maps[:, :-1, :])  # [B, H-1, W]
            else:
                # [B, T, H, W] case  
                conf_x = torch.sigmoid(pred_conf_maps[:, :, :, :-1])  # [B, T, H, W-1]
                conf_y = torch.sigmoid(pred_conf_maps[:, :, :-1, :])  # [B, T, H-1, W]
            
            # Apply confidence weighting: D_i ⊙ (∇D̂_i − ∇D_i)
            weighted_grad_diff_x = conf_x * grad_diff_x
            weighted_grad_diff_y = conf_y * grad_diff_y
            
            # Compute L2 norm: ||Σ D_i ⊙ (∇D̂_i − ∇D_i)||
            # Apply future frame weighting if temporal dimension exists
            if pred_depth.dim() == 4 and future_frame_weight != 1.0:
                N = pred_depth.shape[1]
                frame_weights = torch.ones(N, device=pred_depth.device, dtype=pred_depth.dtype)
                frame_weights[m_frames:] = future_frame_weight
                frame_weights = frame_weights.view(1, N, 1, 1)
                
                # Weight gradient losses
                # Compute norm first to get [B, N] shape, then apply frame weights
                weighted_grad_x_norm = torch.norm(weighted_grad_diff_x, p=2, dim=(-2, -1))  # [B, N]
                weighted_grad_y_norm = torch.norm(weighted_grad_diff_y, p=2, dim=(-2, -1))  # [B, N]
                
                # Apply frame weights (broadcast from [1, N, 1, 1] to [B, N])
                frame_weights_2d = frame_weights.squeeze(-1).squeeze(-1)  # [1, N]
                weighted_grad_x_loss = weighted_grad_x_norm * frame_weights_2d
                weighted_grad_y_loss = weighted_grad_y_norm * frame_weights_2d
                
                gradient_loss = weighted_grad_x_loss.mean() + weighted_grad_y_loss.mean()
            else:
                gradient_loss = torch.norm(weighted_grad_diff_x, p=2).mean() + torch.norm(weighted_grad_diff_y, p=2).mean()
        
        return sky_mask_loss + bce_loss + gradient_weight * gradient_loss


class NormalLosses:
    """ Normal vector losses for surface smoothness """

    @staticmethod
    def pi3_official_normal_loss(points, gt_points, mask):
        """
        official Pi3 loss for normals.
        """
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    @staticmethod
    def compute_normals_from_grid(points):
        """
        Compute normal vectors from 3D point maps using adjacent neighbors on image grid.
        
        Args:
            points: [B, H, W, 3] predicted 3D points in image grid format
            
        Returns:
            normals: [B, H-2, W-2, 3] normal vectors (excluding border pixels)
        """
        B, H, W, _ = points.shape
        
        # Get adjacent neighbors (excluding borders to avoid out-of-bounds)
        # For each point (i,j), use neighbors (i-1,j), (i+1,j), (i,j-1), (i,j+1)
        center = points[:, 1:H-1, 1:W-1, :]  # [B, H-2, W-2, 3]
        left = points[:, 1:H-1, 0:W-2, :]    # [B, H-2, W-2, 3]
        right = points[:, 1:H-1, 2:W, :]     # [B, H-2, W-2, 3]
        up = points[:, 0:H-2, 1:W-1, :]      # [B, H-2, W-2, 3]
        down = points[:, 2:H, 1:W-1, :]      # [B, H-2, W-2, 3]
        
        # Compute vectors to adjacent neighbors
        vec_horizontal = right - left        # [B, H-2, W-2, 3]
        vec_vertical = down - up            # [B, H-2, W-2, 3]
        
        # Compute normal via cross product
        normals = torch.cross(vec_horizontal, vec_vertical, dim=-1)  # [B, H-2, W-2, 3]
        
        # Normalize to unit vectors
        normals_norm = torch.norm(normals, dim=-1, keepdim=True)  # [B, H-2, W-2, 1]
        normals = normals / (normals_norm + 1e-8)  # Avoid division by zero
        
        return normals
    
    @staticmethod
    def normal_loss(pred_points, gt_points):
        """
        Normal loss that minimizes angle between predicted and ground truth normals.
        
        Args:
            pred_points: [B, H, W, 3] predicted 3D points
            gt_points: [B, H, W, 3] ground truth 3D points
            
        Returns:
            Angular loss between predicted and ground truth normals
        """
        # Compute normals from both predicted and ground truth points
        pred_normals = NormalLosses.compute_normals_from_grid(pred_points)  # [B, H-2, W-2, 3]
        gt_normals = NormalLosses.compute_normals_from_grid(gt_points)      # [B, H-2, W-2, 3]
        
        # Compute cosine similarity (dot product of unit normals)
        cos_sim = torch.sum(pred_normals * gt_normals, dim=-1)  # [B, H-2, W-2]
        
        # Clamp to avoid numerical issues with acos
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        
        # Angular loss: minimize angle between normals
        # We use 1 - cos_sim as a proxy for angular distance (equivalent to 1 - cosine similarity)
        # This avoids computing acos which can be numerically unstable
        angular_loss = (1 - cos_sim).mean()
        
        return angular_loss
    
    @staticmethod
    def normal_loss_with_angles(pred_points, gt_points):
        """
        Normal loss using actual angles between normals (more direct but less stable).
        
        Args:
            pred_points: [B, H, W, 3] predicted 3D points
            gt_points: [B, H, W, 3] ground truth 3D points
            
        Returns:
            Mean angular error in radians between predicted and ground truth normals
        """
        # Compute normals from both predicted and ground truth points
        pred_normals = NormalLosses.compute_normals_from_grid(pred_points)  # [B, H-2, W-2, 3]
        gt_normals = NormalLosses.compute_normals_from_grid(gt_points)      # [B, H-2, W-2, 3]
        
        # Compute cosine similarity
        cos_sim = torch.sum(pred_normals * gt_normals, dim=-1)  # [B, H-2, W-2]
        
        # Clamp to valid range for acos
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        
        # Compute angles in radians
        angles = torch.acos(cos_sim)  # [B, H-2, W-2]
        
        # Return mean angular error
        return angles.mean()

class PointCloudLosses:
    """Collection of point cloud loss functions"""

    local_align_res = 4096

    @staticmethod
    def optimal_scale_factor(pred_points, gt_points):
        """
        Compute GLOBAL optimal scale factor for scale-invariant point cloud loss.
        Following Pi3 paper's scale-invariant formulation.
        
        Returns a single global scale factor that applies to the entire group of images,
        regardless of batch or temporal dimensions.

        Supports common shapes:
        - [B, T, H, W, 3]: returns global scalar → shape [1] (broadcastable)
        - [B, H, W, 3]: returns global scalar → shape [1] (broadcastable) 
        - [B, N, 3]: returns global scalar → shape [1] (broadcastable)
        - [T, H, W, 3]: returns global scalar → shape [1] (broadcastable)
        - [H, W, 3] or [N, 3]: returns global scalar → shape [1] (broadcastable)
        """
        eps = 1e-6  # Increased for better numerical stability
        assert pred_points.shape[-1] == 3 and gt_points.shape[-1] == 3, "Last dim must be 3 (x,y,z)"

        # Compute global dot product and squared norm across ALL dimensions
        # This gives us a single scale factor for the entire group of images
        dot_per_point = (pred_points * gt_points).sum(dim=-1)  # Remove last dim (xyz)
        pred_sq_per_point = (pred_points * pred_points).sum(dim=-1)  # Remove last dim (xyz)
        
        # Sum across ALL spatial, temporal, and batch dimensions to get global values
        global_numerator = dot_per_point.sum()  # Sum over all remaining dims -> scalar
        global_denominator = pred_sq_per_point.sum()  # Sum over all remaining dims -> scalar
        
        # Compute global scale factor
        global_denominator = torch.clamp(global_denominator, min=eps)
        global_scale = global_numerator / global_denominator
        
        # Clamp scale factor to reasonable range to prevent extreme values
        global_scale = torch.clamp(global_scale, min=0.1, max=10.0)
        
        # Return as a scalar tensor that can broadcast to any shape
        return global_scale.view(1)


    @staticmethod
    def pi3_si_invariant_point_map_loss(pred_points, gt_points, m_frames=3, future_frame_weight=2.0):
        """
        Point cloud scale-invariant L1 loss with stronger supervision on future frames.
        
        Args:
            pred_points: [B, T, H, W, 3] predicted 3D points
            gt_points: [B, T, H, W, 3] ground truth 3D points  
            m_frames: number of current/input frames (first m frames)
            future_frame_weight: multiplier for future frame losses
        """
        scale_factor = PointCloudLosses.optimal_scale_factor(pred_points, gt_points)
        return PointCloudLosses.pi3_si_invariant_point_map_loss_with_scale(
            pred_points, gt_points, scale_factor, m_frames, future_frame_weight
        )
    
    @staticmethod
    def official_pi3_point_loss(local_pred_points, local_gt_points, local_align_res=4096, m_frames=3, future_frame_weight=1.0, sky_mask=None):
        """
        Official Pi3 point loss with future frame weighting.
        
        Args:
            local_pred_points: (B, N, H, W, 3) predicted local points
            local_gt_points: (B, N, H, W, 3) ground truth local points  
            local_align_res: Resolution for alignment computation
            m_frames: Number of current frames (remaining are future frames)
            future_frame_weight: Weight multiplier for future frame losses
        """

        final_loss = 0.0

        B, N, H, W, _ = local_pred_points.shape

        weights_ = local_gt_points[..., 2]
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, None, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)

        # Create valid masks, excluding sky pixels if sky_mask is provided
        if sky_mask is not None:
            # Resize sky_mask to match point map dimensions if needed
            if sky_mask.shape[-2:] != (H, W):
                sky_mask = torch.nn.functional.interpolate(
                    sky_mask.float().unsqueeze(1), size=(H, W), mode='nearest'
                ).squeeze(1).bool()
            # Valid pixels are non-sky pixels (invert sky mask)
            valid_masks = ~sky_mask  # [B, N, H, W]
        else:
            # Use all pixels if no sky mask provided
            valid_masks = torch.ones((B, N, H, W), device=local_pred_points.device, dtype=torch.bool)

        with torch.no_grad():
            xyz_pred_local = PointCloudLosses.prepare_ROE(local_pred_points.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=local_align_res).contiguous()
            xyz_gt_local = PointCloudLosses.prepare_ROE(local_gt_points.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=local_align_res).contiguous()
            xyz_weights_local = PointCloudLosses.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=local_align_res).contiguous()[:, :, 0]

            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * local_pred_points

        criteria_local = nn.L1Loss(reduction='none')
        local_pts_loss = criteria_local(aligned_local_pts[valid_masks].float(), local_gt_points[valid_masks].float()) # * weights_[valid_masks].float()[..., None]

        # Apply future frame weighting
        # Shape: local_pts_loss is (B*N*H*W*valid_pixels, 3)
        # We need to create frame weights and apply them
        if future_frame_weight != 1.0:
            # Create frame weights: (N,) -> current frames = 1.0, future frames = future_frame_weight
            frame_weights = torch.ones(N, device=local_pred_points.device, dtype=local_pred_points.dtype)
            frame_weights[m_frames:] = future_frame_weight
            
            # Expand weights to match loss tensor shape
            # valid_masks shape: (B, N, H, W) -> need to map to flattened indices
            frame_indices = torch.arange(N, device=local_pred_points.device).view(1, N, 1, 1).expand(B, N, H, W)
            frame_indices_flat = frame_indices[valid_masks]  # Get frame indices for valid pixels
            
            # Apply frame weights to loss
            pixel_frame_weights = frame_weights[frame_indices_flat].unsqueeze(-1)  # (valid_pixels, 1)
            local_pts_loss = local_pts_loss * pixel_frame_weights

        final_loss += local_pts_loss.mean()

        return final_loss, S_opt_local, aligned_local_pts

    @staticmethod
    def prepare_ROE(pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
                # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)

            output.append(valid_pts)

        return torch.stack(output, dim=0)



    @staticmethod
    def pi3_si_invariant_point_map_loss_with_scale(pred_points, gt_points, scale_factor, m_frames=3, future_frame_weight=2.0):
        """
        Point cloud scale-invariant L1 loss using precomputed scale factor.
        
        Args:
            pred_points: [B, T, H, W, 3] predicted 3D points
            gt_points: [B, T, H, W, 3] ground truth 3D points
            scale_factor: precomputed scale factor (scalar tensor)
            m_frames: number of current/input frames (first m frames)
            future_frame_weight: multiplier for future frame losses
        """
        error = (scale_factor * pred_points) - gt_points
        weighted_error = torch.norm(error, p=1, dim=-1)  # [B, T, H, W]
        
        # Create frame weights: current frames get weight 1.0, future frames get future_frame_weight
        B, T = weighted_error.shape[:2]
        frame_weights = torch.ones(T, device=weighted_error.device)
        frame_weights[m_frames:] = future_frame_weight  # Emphasize future frames
        
        # Apply frame weights: [B, T, H, W] * [T] -> [B, T, H, W]
        frame_weights = frame_weights.view(1, T, 1, 1)  # Broadcast to [1, T, 1, 1]
        weighted_error = weighted_error * frame_weights
        
        # Compute mean loss across all dimensions
        loss = weighted_error.mean()
        return loss


    @staticmethod
    def pi3_point_map_loss(pred_points, gt_points):
        """
        Point cloud loss
        """
        return F.mse_loss(pred_points, gt_points)

    @staticmethod
    def pi3_confidence_weighted_point_loss(pred_points, gt_points, pred_conf, gamma=1.0, alpha=0.1, gradient_weight=0.1, eps=1e-6):
        """
        Confidence-weighted Pi3 point loss with regularization and gradient error.
        
        Loss formula: gamma * loss * conf - alpha * log(conf + eps) + gradient_weight * grad_error * conf
        where loss is the L2 distance between predicted and ground truth points,
        and grad_error is the L2 distance between depth gradients.
        
        Args:
            pred_points: [B, T, H, W, 3] or [B, H, W, 3] predicted 3D points
            gt_points: [B, T, H, W, 3] or [B, H, W, 3] ground truth 3D points
            pred_conf: [B, T, H, W, 1] or [B, H, W, 1] predicted confidence maps
            gamma: weight for the confidence-weighted reconstruction loss
            alpha: weight for the confidence regularization term
            gradient_weight: weight for the confidence-weighted gradient error
            eps: small constant to prevent log(0)
            
        Returns:
            Confidence-weighted loss combining reconstruction, regularization, and gradient terms
        """
        # Squeeze confidence if last dimension is 1
        if pred_conf.dim() > 3 and pred_conf.shape[-1] == 1:
            pred_conf = pred_conf.squeeze(-1)  # [B, T, H, W, 1] -> [B, T, H, W]
        
        # Ensure confidence is in [0, 1] range using sigmoid if needed
        pred_conf = torch.sigmoid(pred_conf)
        
        # Clamp confidence to prevent extreme values that cause NaNs
        pred_conf = torch.clamp(pred_conf, min=eps, max=1.0 - eps)
        
        # Compute L1 distance per point with clamping to prevent extreme values
        point_diff = pred_points - gt_points
        l1_error = torch.norm(point_diff, p=1, dim=-1)  # [B, T, H, W] or [B, H, W]
        
        # Clamp L2 error to prevent extreme values
        l1_error = torch.clamp(l1_error, max=100.0)  # Reasonable upper bound

        
        # Confidence-weighted reconstruction loss: gamma * loss * conf
        weighted_reconstruction_loss = gamma * l1_error * pred_conf
        
        # Confidence regularization term: -alpha * log(conf + eps)
        # This encourages the model to be confident (high conf values)
        log_conf = torch.log(pred_conf + eps)
        # Clamp log values to prevent extreme negative values
        log_conf = torch.clamp(log_conf, min=-10.0)  # Prevent extreme negative values
        confidence_regularization = -alpha * log_conf
        
        # Confidence-weighted gradient error term
        gradient_loss = 0.0
        if gradient_weight > 0.0 and pred_points.dim() >= 3:
            # Extract depth maps (z-component)
            pred_depth = pred_points[..., 2]  # [B, T, H, W] or [B, H, W]
            gt_depth = gt_points[..., 2]  # [B, T, H, W] or [B, H, W]
            
            # Compute gradients using the helper function from ConfidenceLosses
            pred_grad_x, pred_grad_y = ConfidenceLosses.compute_depth_gradients(pred_depth)
            gt_grad_x, gt_grad_y = ConfidenceLosses.compute_depth_gradients(gt_depth)
            
            # Compute gradient differences
            grad_diff_x = pred_grad_x - gt_grad_x  # [B, H, W-1] or [B, T, H, W-1]
            grad_diff_y = pred_grad_y - gt_grad_y  # [B, H-1, W] or [B, T, H-1, W]
            
            # Get confidence weights for gradient locations
            if pred_depth.dim() == 3:
                # [B, H, W] case
                conf_x = pred_conf[:, :, :-1]  # [B, H, W-1]
                conf_y = pred_conf[:, :-1, :]  # [B, H-1, W]
            else:
                # [B, T, H, W] case
                conf_x = pred_conf[:, :, :, :-1]  # [B, T, H, W-1]
                conf_y = pred_conf[:, :, :-1, :]  # [B, T, H-1, W]
            
            # Compute L1 error of gradients (element-wise)
            grad_error_x = torch.abs(grad_diff_x)  # [B, H, W-1] or [B, T, H, W-1]
            grad_error_y = torch.abs(grad_diff_y)  # [B, H-1, W] or [B, T, H-1, W]
            
            # Apply confidence weighting: gradient_weight * grad_error * conf
            weighted_grad_error_x = gradient_weight * grad_error_x * conf_x
            weighted_grad_error_y = gradient_weight * grad_error_y * conf_y
            
            # Combine gradient errors
            gradient_loss = weighted_grad_error_x.mean() + weighted_grad_error_y.mean()
        
        # Combine all terms - compute means of first two terms before adding
        reconstruction_mean = weighted_reconstruction_loss.mean()
        regularization_mean = confidence_regularization.mean()
        total_loss = reconstruction_mean + regularization_mean + gradient_loss
        
        # Check for NaNs before returning
        if torch.isnan(total_loss).any():
            print("❌ NaN detected in confidence-weighted loss, returning fallback L2 loss")
            return torch.nn.functional.mse_loss(pred_points, gt_points)
        
        # Return mean loss with additional clamping for safety
        mean_loss = total_loss.mean()
        return torch.clamp(mean_loss, max=1000.0)  # Reasonable upper bound for loss
    
    @staticmethod
    def chamfer_distance(pred_points, gt_points, bidirectional=True):
        """
        Chamfer Distance between two point clouds
        Args:
            pred_points: [B, N, 3] predicted points
            gt_points: [B, M, 3] ground truth points
        """
        # pred_points: [B, N, 3], gt_points: [B, M, 3]
        B, N, _ = pred_points.shape
        _, M, _ = gt_points.shape
        
        # Expand dimensions for pairwise distance computation
        pred_expanded = pred_points.unsqueeze(2)  # [B, N, 1, 3]
        gt_expanded = gt_points.unsqueeze(1)      # [B, 1, M, 3]
        
        # Compute pairwise distances
        distances = torch.norm(pred_expanded - gt_expanded, dim=3)  # [B, N, M]
        
        # Forward distance: for each predicted point, find closest GT point
        forward_dist = torch.min(distances, dim=2)[0]  # [B, N]
        forward_loss = forward_dist.mean()
        
        if bidirectional:
            # Backward distance: for each GT point, find closest predicted point
            backward_dist = torch.min(distances, dim=1)[0]  # [B, M]
            backward_loss = backward_dist.mean()
            return forward_loss + backward_loss
        
        return forward_loss
    
    @staticmethod
    def earth_movers_distance(pred_points, gt_points):
        """
        Approximation of Earth Mover's Distance using Hungarian algorithm
        Note: This is a simplified version, for production use scipy.optimize.linear_sum_assignment
        """
        # Simple approximation using L2 distance
        B, N, _ = pred_points.shape
        _, M, _ = gt_points.shape
        
        # Ensure same number of points by sampling
        if N > M:
            indices = torch.randperm(N, device=pred_points.device)[:M]
            pred_points = pred_points[:, indices]
        elif M > N:
            indices = torch.randperm(M, device=gt_points.device)[:N]
            gt_points = gt_points[:, indices]
        
        # Compute optimal assignment (simplified)
        distances = torch.norm(pred_points - gt_points, dim=2)
        return distances.mean()
    
    @staticmethod
    def normal_consistency_loss(pred_points, gt_points, k=8):
        """
        Loss based on normal vector consistency
        Args:
            pred_points: [B, N, 3] predicted points
            gt_points: [B, N, 3] ground truth points (same number)
            k: number of neighbors for normal computation
        """
        def compute_normals(points, k=8):
            B, N, _ = points.shape
            # Simple normal computation using k-nearest neighbors
            # This is a simplified version - in practice, use KDTree or more efficient methods
            
            normals = torch.zeros_like(points)
            for b in range(B):
                pts = points[b]  # [N, 3]
                for i in range(N):
                    # Find k nearest neighbors
                    dists = torch.norm(pts - pts[i:i+1], dim=1)
                    _, indices = torch.topk(dists, k+1, largest=False)
                    neighbors = pts[indices[1:]]  # Exclude the point itself
                    
                    # Compute normal using PCA (simplified)
                    centered = neighbors - neighbors.mean(0)
                    cov = torch.mm(centered.T, centered)
                    _, _, V = torch.svd(cov)
                    normals[b, i] = V[:, -1]  # Normal is last eigenvector
            
            return normals
        
        pred_normals = compute_normals(pred_points, k)
        gt_normals = compute_normals(gt_points, k)
        
        # Cosine similarity loss
        cos_sim = F.cosine_similarity(pred_normals, gt_normals, dim=2)
        return (1 - cos_sim).mean()


class CameraPoseLosses:
    """Collection of camera pose loss functions"""


    @staticmethod
    def official_pi3_camera_pose_loss(pred_pose, gt_pose, scale, m_frames=3, future_frame_weight=1.0):
        """
        Camera pose loss with future frame weighting.
        
        Args:
            pred_pose: Predicted camera poses (B, N, 4, 4)
            gt_pose: Ground truth camera poses (B, N, 4, 4)
            scale: Scale factor for alignment
            m_frames: Number of current frames (first m frames)
            future_frame_weight: Weight multiplier for future frame losses
        """
        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        pred_pose_align[..., :3, 3] *=  scale.view(B, 1, 1)

        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)

        pred_w2c_exp = pred_w2c.unsqueeze(2)
        pred_pose_exp = pred_pose_align.unsqueeze(1)
        
        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)
        
        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
        
        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        # Apply future frame weighting if needed
        if future_frame_weight != 1.0:
            # Create frame pair weights for relative poses
            frame_weights = torch.ones(N, N, device=pred_pose.device, dtype=pred_pose.dtype)
            # Weight pairs involving future frames
            for i in range(N):
                for j in range(N):
                    if i >= m_frames or j >= m_frames:  # If either frame is a future frame
                        frame_weights[i, j] = future_frame_weight
            
            # Apply mask and flatten weights to match loss computation
            frame_weights_masked = frame_weights[mask].view(-1)  # Shape: (N*(N-1),)
            # Repeat for each batch
            frame_weights_expanded = frame_weights_masked.repeat(B)  # Shape: (B*N*(N-1),)
            
            trans_loss = F.huber_loss(t_pred, t_gt, reduction='none', delta=0.1)
            trans_loss = (trans_loss.mean(dim=-1) * frame_weights_expanded).mean()
            
            rot_loss_per_pair = rot_ang_loss(
                R_pred.reshape(-1, 3, 3), 
                R_gt.reshape(-1, 3, 3),
                reduction='none'
            )
            rot_loss = (rot_loss_per_pair * frame_weights_expanded).mean()
        else:
            trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)
            rot_loss = rot_ang_loss(
                R_pred.reshape(-1, 3, 3), 
                R_gt.reshape(-1, 3, 3)
            )

        alpha = 1

        total_loss = alpha * trans_loss + rot_loss
        return total_loss

    @staticmethod
    def pi3_camera_pose_loss(pred_poses, gt_poses, lambda_trans=1.0, scale_factor=None):
        """
        Pose loss - computes relative pose loss across all batch samples with optional scale correction
        Args:
            pred_poses: [B, N, 4, 4] predicted poses
            gt_poses: [B, N, 4, 4] ground truth poses
            scale_factor: optional precomputed scale factor (defaults to 1.0 if None)
        """
        # Use precomputed scale or default to 1.0
        if scale_factor is None:
            scale_factor = torch.tensor(1.0, device=pred_poses.device)
        
        return CameraPoseLosses.pi3_camera_pose_loss_with_scale(pred_poses, gt_poses, scale_factor, lambda_trans)
    
    @staticmethod
    def pi3_camera_pose_loss_with_scale(pred_poses, gt_poses, scale_factor, lambda_trans=1.0):
        """
        Pose loss using precomputed scale factor from point clouds
        Args:
            pred_poses: [B, N, 4, 4] predicted poses
            gt_poses: [B, N, 4, 4] ground truth poses
            scale_factor: precomputed scale factor (scalar tensor)
        """
        B, N = pred_poses.shape[:2]
        if N < 2:
            return pred_poses.new_tensor(0.0)
        
        # Apply precomputed scale factor to predicted translations
        pred_translations = pred_poses[:, :, :3, 3]  # [B, N, 3]
        
        # Apply scale to predicted poses (only translation part)
        scaled_pred_poses = pred_poses.clone()
        scaled_pred_poses[:, :, :3, 3] = scale_factor * pred_translations
        
        # Vectorized computation of all relative poses using scaled predictions
        pred_i = scaled_pred_poses.unsqueeze(2)  # [B, N, 1, 4, 4]
        pred_j = scaled_pred_poses.unsqueeze(1)  # [B, 1, N, 4, 4]
        gt_i = gt_poses.unsqueeze(2)             # [B, N, 1, 4, 4]
        gt_j = gt_poses.unsqueeze(1)             # [B, 1, N, 4, 4]
        
        # Compute all relative poses at once: inv(pose_i) @ pose_j for all i,j pairs
        pred_rel_poses = torch.linalg.inv(pred_i) @ pred_j  # [B, N, N, 4, 4]
        gt_rel_poses = torch.linalg.inv(gt_i) @ gt_j        # [B, N, N, 4, 4]
        
        # Extract rotations and translations
        pred_R = pred_rel_poses[:, :, :, :3, :3]  # [B, N, N, 3, 3]
        pred_t = pred_rel_poses[:, :, :, :3, 3]   # [B, N, N, 3]
        gt_R = gt_rel_poses[:, :, :, :3, :3]      # [B, N, N, 3, 3]
        gt_t = gt_rel_poses[:, :, :, :3, 3]       # [B, N, N, 3]
        
        # Use upper-triangular mask to avoid double-counting (i<j), exclude diagonal
        pair_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=pred_poses.device), diagonal=1)
        pair_mask = pair_mask.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
        
        # Apply mask to get valid pairs flattened
        pred_R_valid = pred_R[pair_mask]  # [B*N*(N-1)/2, 3, 3]
        pred_t_valid = pred_t[pair_mask]  # [B*N*(N-1)/2, 3]
        gt_R_valid = gt_R[pair_mask]      # [B*N*(N-1)/2, 3, 3]
        gt_t_valid = gt_t[pair_mask]      # [B*N*(N-1)/2, 3]
        
        # Compute losses on all valid pairs (means provide proper normalization)
        rot_loss = CameraPoseLosses.rotation_loss(pred_R_valid, gt_R_valid, loss_type='geodesic')
        trans_loss = CameraPoseLosses.translation_loss(pred_t_valid, gt_t_valid, loss_type='huber')
        
        return rot_loss + lambda_trans * trans_loss
    
    @staticmethod
    def translation_loss(pred_trans, gt_trans, loss_type='l2'):
        """Translation loss between camera poses"""
        if loss_type == 'l1':
            return F.l1_loss(pred_trans, gt_trans)
        elif loss_type == 'l2':
            return F.mse_loss(pred_trans, gt_trans)
        elif loss_type == 'huber':
            return F.huber_loss(pred_trans, gt_trans)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    @staticmethod
    def rotation_loss(pred_rot, gt_rot, loss_type='geodesic'):
        """
        Rotation loss between camera poses
        Args:
            pred_rot: [B, 3, 3] or [B, 4] (rotation matrices or quaternions)
            gt_rot: [B, 3, 3] or [B, 4] (rotation matrices or quaternions)
        """
        if loss_type == 'frobenius':
            # Frobenius norm of rotation matrix difference
            return F.mse_loss(pred_rot, gt_rot)
        
        elif loss_type == 'geodesic':
            # Geodesic distance on SO(3)
            if pred_rot.shape[-1] == 4:  # Quaternions
                # Convert to rotation matrices
                pred_rot = CameraPoseLosses.quaternion_to_matrix(pred_rot)
                gt_rot = CameraPoseLosses.quaternion_to_matrix(gt_rot)
            
            # Compute relative rotation: R_rel = R_gt^T @ R_pred
            rel_rot = torch.bmm(gt_rot.transpose(-2, -1), pred_rot)
            
            # Geodesic distance = arccos((trace(R) - 1) / 2)
            trace = rel_rot.diagonal(dim1=-2, dim2=-1).sum(-1)
            cos_angle = (trace - 1) / 2
            cos_angle = torch.clamp(cos_angle, -1 + 1e-6, 1 - 1e-6)
            angles = torch.acos(cos_angle)
            
            return angles.mean()
        
        elif loss_type == 'chordal':
            # Chordal distance
            return F.mse_loss(pred_rot, gt_rot)
        
        else:
            raise ValueError(f"Unknown rotation loss type: {loss_type}")
    
    @staticmethod
    def pose_composition_loss(pred_poses, gt_poses):
        """
        Loss based on pose composition
        Args:
            pred_poses: [B, T, 4, 4] predicted poses
            gt_poses: [B, T, 4, 4] ground truth poses
        """
        B, T = pred_poses.shape[:2]
        
        # Compute relative poses between consecutive frames
        pred_rel_poses = []
        gt_rel_poses = []
        
        for t in range(T - 1):
            # Relative pose = inv(pose_t) @ pose_{t+1}
            pred_rel = torch.bmm(
                CameraPoseLosses.invert_pose(pred_poses[:, t]),
                pred_poses[:, t + 1]
            )
            gt_rel = torch.bmm(
                CameraPoseLosses.invert_pose(gt_poses[:, t]),
                gt_poses[:, t + 1]
            )
            
            pred_rel_poses.append(pred_rel)
            gt_rel_poses.append(gt_rel)
        
        pred_rel_poses = torch.stack(pred_rel_poses, dim=1)  # [B, T-1, 4, 4]
        gt_rel_poses = torch.stack(gt_rel_poses, dim=1)
        
        # Compute losses on relative poses
        trans_loss = CameraPoseLosses.translation_loss(
            pred_rel_poses[:, :, :3, 3], 
            gt_rel_poses[:, :, :3, 3]
        )
        rot_loss = CameraPoseLosses.rotation_loss(
            pred_rel_poses[:, :, :3, :3], 
            gt_rel_poses[:, :, :3, :3]
        )
        
        return trans_loss + rot_loss
    
    @staticmethod
    def reprojection_loss(pred_poses, gt_poses, points_3d, intrinsics):
        """
        Reprojection loss using 3D points
        Args:
            pred_poses: [B, 4, 4] predicted camera poses
            gt_poses: [B, 4, 4] ground truth camera poses
            points_3d: [B, N, 3] 3D points in world coordinates
            intrinsics: [B, 3, 3] camera intrinsics
        """
        # Project 3D points using predicted poses
        pred_points_2d = CameraPoseLosses.project_points(points_3d, pred_poses, intrinsics)
        
        # Project 3D points using ground truth poses
        gt_points_2d = CameraPoseLosses.project_points(points_3d, gt_poses, intrinsics)
        
        # L2 loss in image space
        return F.mse_loss(pred_points_2d, gt_points_2d)
    
    @staticmethod
    def invert_pose(poses):
        """Invert 4x4 pose matrices"""
        B = poses.shape[0]
        inv_poses = torch.zeros_like(poses)
        
        R = poses[:, :3, :3]
        t = poses[:, :3, 3:4]
        
        R_inv = R.transpose(-2, -1)
        t_inv = -torch.bmm(R_inv, t)
        
        inv_poses[:, :3, :3] = R_inv
        inv_poses[:, :3, 3:4] = t_inv
        inv_poses[:, 3, 3] = 1
        
        return inv_poses
    
    @staticmethod
    def project_points(points_3d, poses, intrinsics):
        """Project 3D points to 2D using camera poses and intrinsics"""
        B, N, _ = points_3d.shape
        
        # Convert to homogeneous coordinates
        points_3d_hom = torch.cat([points_3d, torch.ones(B, N, 1, device=points_3d.device)], dim=2)
        
        # Transform to camera coordinates
        # Note: pose is camera-to-world, so we need world-to-camera
        inv_poses = CameraPoseLosses.invert_pose(poses)
        points_cam = torch.bmm(points_3d_hom, inv_poses.transpose(-2, -1))[:, :, :3]
        
        # Project to image plane
        points_2d_hom = torch.bmm(points_cam, intrinsics.transpose(-2, -1))
        points_2d = points_2d_hom[:, :, :2] / (points_2d_hom[:, :, 2:3] + 1e-6)
        
        return points_2d
    
    @staticmethod
    def quaternion_to_matrix(quaternions):
        """Convert quaternions to rotation matrices"""
        # quaternions: [B, 4] in (w, x, y, z) format
        w, x, y, z = quaternions.unbind(-1)
        
        # Normalize quaternions
        norm = torch.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix
        rotation_matrix = torch.stack([
            torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
            torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], dim=-1),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], dim=-1)
        ], dim=-2)
        
        return rotation_matrix

    @staticmethod
    def sequential_camera_loss(pred_poses, gt_poses, lambda_trans=1.0, lambda_rot=1.0, velocity_weight=0.5):
        """
        Sequential camera loss that enforces smooth motion and temporal consistency.
        
        Args:
            pred_poses: [B, N, 4, 4] predicted camera poses
            gt_poses: [B, N, 4, 4] ground truth camera poses
            lambda_trans: weight for translation loss
            lambda_rot: weight for rotation loss
            velocity_weight: weight for velocity consistency loss
            
        Returns:
            Combined sequential loss
        """
        B, N = pred_poses.shape[:2]
        if N < 2:
            return pred_poses.new_tensor(0.0)
        
        # Sequential pose loss (frame-to-frame)
        sequential_loss = 0.0
        velocity_loss = 0.0
        
        for t in range(N - 1):
            # Frame-to-frame pose loss
            pred_R_t = pred_poses[:, t, :3, :3]
            pred_t_t = pred_poses[:, t, :3, 3]
            pred_R_t1 = pred_poses[:, t+1, :3, :3]
            pred_t_t1 = pred_poses[:, t+1, :3, 3]
            
            gt_R_t = gt_poses[:, t, :3, :3]
            gt_t_t = gt_poses[:, t, :3, 3]
            gt_R_t1 = gt_poses[:, t+1, :3, :3]
            gt_t_t1 = gt_poses[:, t+1, :3, 3]
            
            # Relative transformation between consecutive frames
            pred_rel_R = torch.bmm(pred_R_t.transpose(-2, -1), pred_R_t1)
            pred_rel_t = torch.bmm(pred_R_t.transpose(-2, -1), (pred_t_t1 - pred_t_t).unsqueeze(-1)).squeeze(-1)
            
            gt_rel_R = torch.bmm(gt_R_t.transpose(-2, -1), gt_R_t1)
            gt_rel_t = torch.bmm(gt_R_t.transpose(-2, -1), (gt_t_t1 - gt_t_t).unsqueeze(-1)).squeeze(-1)
            
            # Rotation and translation losses
            rot_loss = CameraPoseLosses.rotation_loss(pred_rel_R, gt_rel_R, loss_type='geodesic')
            trans_loss = CameraPoseLosses.translation_loss(pred_rel_t, gt_rel_t, loss_type='huber')
            
            sequential_loss += lambda_rot * rot_loss + lambda_trans * trans_loss
            
            # Velocity consistency loss (if we have at least 3 frames)
            if t < N - 2:
                # Current velocity: t -> t+1
                pred_vel_curr = pred_t_t1 - pred_t_t
                gt_vel_curr = gt_t_t1 - gt_t_t
                
                # Next velocity: t+1 -> t+2
                pred_t_t2 = pred_poses[:, t+2, :3, 3]
                gt_t_t2 = gt_poses[:, t+2, :3, 3]
                pred_vel_next = pred_t_t2 - pred_t_t1
                gt_vel_next = gt_t_t2 - gt_t_t1
                
                # Velocity acceleration (change in velocity)
                pred_accel = pred_vel_next - pred_vel_curr
                gt_accel = gt_vel_next - gt_vel_curr
                
                velocity_loss += F.mse_loss(pred_accel, gt_accel)
        
        # Normalize by number of sequential pairs
        sequential_loss = sequential_loss / (N - 1)
        if N > 2:
            velocity_loss = velocity_loss / (N - 2)
        
        return sequential_loss + velocity_weight * velocity_loss

    @staticmethod
    def epipolar_camera_loss(pred_poses, gt_poses, points_3d=None, lambda_epipolar=1.0):
        """
        Epipolar geometry-based camera loss using fundamental matrix constraints.
        
        Args:
            pred_poses: [B, N, 4, 4] predicted camera poses
            gt_poses: [B, N, 4, 4] ground truth camera poses
            points_3d: [B, M, 3] optional 3D points for reprojection
            lambda_epipolar: weight for epipolar constraint loss
            
        Returns:
            Epipolar geometry loss
        """
        B, N = pred_poses.shape[:2]
        if N < 2:
            return pred_poses.new_tensor(0.0)
        
        device = pred_poses.device
        
        # Use identity intrinsics as approximation (can be made configurable)
        K = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        
        total_loss = 0.0
        num_pairs = 0
        
        # Compute epipolar loss for all frame pairs
        for i in range(N):
            for j in range(i + 1, N):
                # Extract poses
                pred_pose_i, pred_pose_j = pred_poses[:, i], pred_poses[:, j]
                gt_pose_i, gt_pose_j = gt_poses[:, i], gt_poses[:, j]
                
                # Compute relative poses
                pred_rel_pose = torch.bmm(torch.linalg.inv(pred_pose_i), pred_pose_j)
                gt_rel_pose = torch.bmm(torch.linalg.inv(gt_pose_i), gt_pose_j)
                
                # Extract relative rotation and translation
                pred_R_rel = pred_rel_pose[:, :3, :3]
                pred_t_rel = pred_rel_pose[:, :3, 3]
                gt_R_rel = gt_rel_pose[:, :3, :3]
                gt_t_rel = gt_rel_pose[:, :3, 3]
                
                # Compute essential matrices
                pred_E = CameraPoseLosses.skew_symmetric(pred_t_rel) @ pred_R_rel
                gt_E = CameraPoseLosses.skew_symmetric(gt_t_rel) @ gt_R_rel
                
                # Essential matrix loss (Frobenius norm)
                essential_loss = F.mse_loss(pred_E, gt_E)
                
                total_loss += lambda_epipolar * essential_loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else pred_poses.new_tensor(0.0)

    @staticmethod
    def trajectory_smoothness_loss(pred_poses, gt_poses, lambda_smooth=1.0):
        """
        Trajectory smoothness loss that penalizes jerky camera motion.
        
        Args:
            pred_poses: [B, N, 4, 4] predicted camera poses
            gt_poses: [B, N, 4, 4] ground truth camera poses
            lambda_smooth: weight for smoothness penalty
            
        Returns:
            Trajectory smoothness loss
        """
        B, N = pred_poses.shape[:2]
        if N < 3:
            return pred_poses.new_tensor(0.0)
        
        # Extract translation trajectories
        pred_traj = pred_poses[:, :, :3, 3]  # [B, N, 3]
        gt_traj = gt_poses[:, :, :3, 3]      # [B, N, 3]
        
        # Compute second derivatives (acceleration) as smoothness measure
        pred_vel = pred_traj[:, 1:] - pred_traj[:, :-1]  # [B, N-1, 3]
        gt_vel = gt_traj[:, 1:] - gt_traj[:, :-1]        # [B, N-1, 3]
        
        pred_accel = pred_vel[:, 1:] - pred_vel[:, :-1]  # [B, N-2, 3]
        gt_accel = gt_vel[:, 1:] - gt_vel[:, :-1]        # [B, N-2, 3]
        
        # L2 loss on acceleration (smoothness)
        smoothness_loss = F.mse_loss(pred_accel, gt_accel)
        
        # Also penalize high acceleration magnitudes
        pred_accel_mag = torch.norm(pred_accel, dim=-1)  # [B, N-2]
        gt_accel_mag = torch.norm(gt_accel, dim=-1)      # [B, N-2]
        
        magnitude_loss = F.mse_loss(pred_accel_mag, gt_accel_mag)
        
        return lambda_smooth * (smoothness_loss + 0.5 * magnitude_loss)

    @staticmethod
    def scale_invariant_camera_loss(pred_poses, gt_poses, lambda_scale=1.0):
        """
        Scale-invariant camera loss that is robust to global scale differences.
        
        Args:
            pred_poses: [B, N, 4, 4] predicted camera poses
            gt_poses: [B, N, 4, 4] ground truth camera poses
            lambda_scale: weight for scale-invariant loss
            
        Returns:
            Scale-invariant camera loss
        """
        B, N = pred_poses.shape[:2]
        if N < 2:
            return pred_poses.new_tensor(0.0)
        
        # Extract translations
        pred_t = pred_poses[:, :, :3, 3]  # [B, N, 3]
        gt_t = gt_poses[:, :, :3, 3]      # [B, N, 3]
        
        # Center trajectories (remove global translation)
        pred_t_centered = pred_t - pred_t.mean(dim=1, keepdim=True)
        gt_t_centered = gt_t - gt_t.mean(dim=1, keepdim=True)
        
        # Compute optimal scale factor using least squares
        # Solve: scale * pred_t_centered ≈ gt_t_centered
        numerator = torch.sum(pred_t_centered * gt_t_centered, dim=[1, 2])  # [B]
        denominator = torch.sum(pred_t_centered * pred_t_centered, dim=[1, 2])  # [B]
        
        # Avoid division by zero
        scale_factor = numerator / (denominator + 1e-8)  # [B]
        scale_factor = scale_factor.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        
        # Apply scale factor and compute loss
        pred_t_scaled = scale_factor * pred_t_centered
        translation_loss = F.mse_loss(pred_t_scaled, gt_t_centered)
        
        # Rotation loss (scale-invariant by nature)
        pred_R = pred_poses[:, :, :3, :3]  # [B, N, 3, 3]
        gt_R = gt_poses[:, :, :3, :3]      # [B, N, 3, 3]
        
        rotation_loss = 0.0
        for t in range(N):
            rotation_loss += CameraPoseLosses.rotation_loss(
                pred_R[:, t], gt_R[:, t], loss_type='geodesic'
            )
        rotation_loss = rotation_loss / N
        
        return lambda_scale * (translation_loss + rotation_loss)

    @staticmethod
    def photometric_camera_loss(pred_poses, gt_poses, images, intrinsics=None, lambda_photo=1.0):
        """
        Photometric camera loss using image warping and photometric consistency.
        
        Args:
            pred_poses: [B, N, 4, 4] predicted camera poses
            gt_poses: [B, N, 4, 4] ground truth camera poses
            images: [B, N, C, H, W] input images
            intrinsics: [B, 3, 3] camera intrinsics (optional)
            lambda_photo: weight for photometric loss
            
        Returns:
            Photometric consistency loss
        """
        B, N, C, H, W = images.shape
        device = images.device
        
        if N < 2:
            return pred_poses.new_tensor(0.0)
        
        # Use identity intrinsics if not provided
        if intrinsics is None:
            intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
            # Adjust for image dimensions
            intrinsics[:, 0, 0] = W / 2  # fx
            intrinsics[:, 1, 1] = H / 2  # fy
            intrinsics[:, 0, 2] = W / 2  # cx
            intrinsics[:, 1, 2] = H / 2  # cy
        
        total_loss = 0.0
        num_pairs = 0
        
        # Generate depth maps (assuming unit depth as approximation)
        depth_map = torch.ones(B, H, W, device=device)
        
        # Compare photometric consistency for consecutive frames
        for t in range(N - 1):
            img_curr = images[:, t]      # [B, C, H, W]
            img_next = images[:, t + 1]  # [B, C, H, W]
            
            # Predicted and ground truth relative poses
            pred_pose_rel = torch.bmm(
                torch.linalg.inv(pred_poses[:, t]), 
                pred_poses[:, t + 1]
            )
            gt_pose_rel = torch.bmm(
                torch.linalg.inv(gt_poses[:, t]), 
                gt_poses[:, t + 1]
            )
            
            # Warp current image to next frame using predicted poses
            pred_warped = CameraPoseLosses.warp_image(
                img_curr, depth_map, pred_pose_rel, intrinsics
            )
            
            # Warp current image to next frame using ground truth poses
            gt_warped = CameraPoseLosses.warp_image(
                img_curr, depth_map, gt_pose_rel, intrinsics
            )
            
            # Photometric loss (L1 difference between warped and target)
            pred_photo_loss = F.l1_loss(pred_warped, img_next, reduction='mean')
            gt_photo_loss = F.l1_loss(gt_warped, img_next, reduction='mean')
            
            # Penalize deviation from ground truth photometric error
            photo_loss = F.mse_loss(pred_photo_loss, gt_photo_loss)
            
            total_loss += lambda_photo * photo_loss
            num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else pred_poses.new_tensor(0.0)

    @staticmethod
    def skew_symmetric(vec):
        """
        Create skew-symmetric matrix from 3D vector.
        
        Args:
            vec: [B, 3] 3D vectors
            
        Returns:
            [B, 3, 3] skew-symmetric matrices
        """
        B = vec.shape[0]
        device = vec.device
        
        skew = torch.zeros(B, 3, 3, device=device)
        skew[:, 0, 1] = -vec[:, 2]
        skew[:, 0, 2] = vec[:, 1]
        skew[:, 1, 0] = vec[:, 2]
        skew[:, 1, 2] = -vec[:, 0]
        skew[:, 2, 0] = -vec[:, 1]
        skew[:, 2, 1] = vec[:, 0]
        
        return skew

    @staticmethod
    def warp_image(image, depth, pose, intrinsics):
        """
        Warp image using depth and camera pose (simplified implementation).
        
        Args:
            image: [B, C, H, W] input image
            depth: [B, H, W] depth map
            pose: [B, 4, 4] camera pose transformation
            intrinsics: [B, 3, 3] camera intrinsics
            
        Returns:
            [B, C, H, W] warped image
        """
        B, C, H, W = image.shape
        device = image.device
        
        # Create pixel grid
        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Convert to homogeneous coordinates
        ones = torch.ones_like(x)
        pixel_coords = torch.stack([x, y, ones], dim=-1)  # [H, W, 3]
        pixel_coords = pixel_coords.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 3]
        
        # Unproject to 3D
        inv_K = torch.linalg.inv(intrinsics)  # [B, 3, 3]
        points_3d = torch.einsum('bij,bhwj->bhwi', inv_K, pixel_coords)  # [B, H, W, 3]
        points_3d = points_3d * depth.unsqueeze(-1)  # [B, H, W, 3]
        
        # Transform to new camera frame
        points_3d_hom = torch.cat([
            points_3d, 
            torch.ones(B, H, W, 1, device=device)
        ], dim=-1)  # [B, H, W, 4]
        
        transformed_points = torch.einsum('bij,bhwj->bhwi', pose, points_3d_hom)  # [B, H, W, 4]
        transformed_points = transformed_points[..., :3]  # [B, H, W, 3]
        
        # Project back to 2D
        projected = torch.einsum('bij,bhwj->bhwi', intrinsics, transformed_points)  # [B, H, W, 3]
        projected_2d = projected[..., :2] / (projected[..., 2:3] + 1e-8)  # [B, H, W, 2]
        
        # Normalize to [-1, 1] for grid_sample
        projected_2d[..., 0] = 2.0 * projected_2d[..., 0] / (W - 1) - 1.0
        projected_2d[..., 1] = 2.0 * projected_2d[..., 1] / (H - 1) - 1.0
        
        # Sample from original image
        warped = F.grid_sample(
            image, projected_2d, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )
        
        return warped


class CombinedLoss(nn.Module):
    """Combined loss function for multi-task learning"""
    
    def __init__(self, 
                 depth_weight=1.0, 
                 pointcloud_weight=1.0, 
                 pose_weight=1.0,
                 depth_loss_type='l1',
                 pose_rot_loss_type='geodesic'):
        super().__init__()
        
        self.depth_weight = depth_weight
        self.pointcloud_weight = pointcloud_weight
        self.pose_weight = pose_weight
        self.depth_loss_type = depth_loss_type
        self.pose_rot_loss_type = pose_rot_loss_type
        
        self.depth_losses = DepthLosses()
        self.pointcloud_losses = PointCloudLosses()
        self.pose_losses = CameraPoseLosses()
    
    def forward(self, predictions, targets):
        """
        Compute combined loss
        Args:
            predictions: dict with keys 'depth', 'points_3d', 'poses'
            targets: dict with corresponding ground truth values
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Depth loss
        if 'depth' in predictions and 'depth' in targets:
            mask = targets.get('depth_mask', None)
            
            if self.depth_loss_type == 'l1':
                depth_loss = self.depth_losses.l1_loss(predictions['depth'], targets['depth'], mask)
            elif self.depth_loss_type == 'l2':
                depth_loss = self.depth_losses.l2_loss(predictions['depth'], targets['depth'], mask)
            elif self.depth_loss_type == 'berhu':
                depth_loss = self.depth_losses.berhu_loss(predictions['depth'], targets['depth'], mask)
            elif self.depth_loss_type == 'scale_invariant':
                depth_loss = self.depth_losses.scale_invariant_loss(predictions['depth'], targets['depth'], mask)
            else:
                raise ValueError(f"Unknown depth loss type: {self.depth_loss_type}")
            
            total_loss += self.depth_weight * depth_loss
            loss_dict['depth_loss'] = depth_loss
        
        # Point cloud loss
        if 'points_3d' in predictions and 'points_3d' in targets:
            pc_loss = self.pointcloud_losses.chamfer_distance(predictions['points_3d'], targets['points_3d'])
            total_loss += self.pointcloud_weight * pc_loss
            loss_dict['pointcloud_loss'] = pc_loss
        
        # Pose loss
        if 'poses' in predictions and 'poses' in targets:
            # Split pose into rotation and translation
            pred_rot = predictions['poses'][:, :3, :3]
            pred_trans = predictions['poses'][:, :3, 3]
            gt_rot = targets['poses'][:, :3, :3]
            gt_trans = targets['poses'][:, :3, 3]
            
            trans_loss = self.pose_losses.translation_loss(pred_trans, gt_trans)
            rot_loss = self.pose_losses.rotation_loss(pred_rot, gt_rot, self.pose_rot_loss_type)
            
            pose_loss = trans_loss + rot_loss
            total_loss += self.pose_weight * pose_loss
            loss_dict['pose_loss'] = pose_loss
            loss_dict['translation_loss'] = trans_loss
            loss_dict['rotation_loss'] = rot_loss
        
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict


class ObjectDetectionLosses:
    """Collection of object detection loss functions for DETR/Grounding DINO style models"""

    @staticmethod
    def hungarian_matcher(pred_logits, pred_boxes, gt_labels, gt_boxes, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        """
        Hungarian matcher for bipartite matching between predictions and ground truth.
        
        Args:
            pred_logits: [B, num_queries, num_classes] - predicted class logits
            pred_boxes: [B, num_queries, 4] - predicted boxes (cx, cy, w, h) normalized [0,1]
            gt_labels: list of [num_gt_i] - ground truth labels for each image
            gt_boxes: list of [num_gt_i, 4] - ground truth boxes for each image
            
        Returns:
            list of (pred_indices, gt_indices) tuples for each image
        """
        B, num_queries = pred_logits.shape[:2]
        
        # Convert logits to probabilities
        pred_probs = F.softmax(pred_logits, dim=-1)  # [B, num_queries, num_classes]
        
        indices = []
        for b in range(B):
            # Get ground truth for this image
            tgt_labels = gt_labels[b]  # [num_gt]
            tgt_boxes = gt_boxes[b]    # [num_gt, 4]
            
            if len(tgt_labels) == 0:
                # No ground truth objects
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue
            
            # Classification cost: negative log probability of correct class
            cost_class_mat = -pred_probs[b, :, tgt_labels]  # [num_queries, num_gt]
            
            # L1 cost between predicted and ground truth boxes
            cost_bbox_mat = torch.cdist(pred_boxes[b], tgt_boxes, p=1)  # [num_queries, num_gt]
            
            # Generalized IoU cost
            cost_giou_mat = -ObjectDetectionLosses.generalized_box_iou(pred_boxes[b], tgt_boxes)  # [num_queries, num_gt]
            
            # Combined cost matrix
            cost_matrix = cost_class * cost_class_mat + cost_bbox * cost_bbox_mat + cost_giou * cost_giou_mat
            cost_matrix = cost_matrix.detach().cpu().numpy()
            
            # Hungarian algorithm
            pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
            
            indices.append((torch.tensor(pred_indices, dtype=torch.long), torch.tensor(gt_indices, dtype=torch.long)))
        
        return indices

    @staticmethod
    def generalized_box_iou(boxes1, boxes2):
        """
        Generalized IoU between two sets of boxes.
        
        Args:
            boxes1: [N, 4] boxes in (cx, cy, w, h) format
            boxes2: [M, 4] boxes in (cx, cy, w, h) format
            
        Returns:
            [N, M] generalized IoU matrix
        """
        # Convert to (x1, y1, x2, y2) format
        def box_cxcywh_to_xyxy(boxes):
            cx, cy, w, h = boxes.unbind(-1)
            return torch.stack([cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h], dim=-1)
        
        boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)  # [N, 4]
        boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)  # [M, 4]
        
        # Compute intersection
        lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])  # [N, M, 2]
        rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])  # [N, M, 2]
        
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
        
        # Compute areas
        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])  # [N]
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])  # [M]
        union = area1[:, None] + area2[None, :] - intersection  # [N, M]
        
        # IoU
        iou = intersection / union
        
        # Compute enclosing box
        lt_enclosing = torch.min(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])  # [N, M, 2]
        rb_enclosing = torch.max(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])  # [N, M, 2]
        wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)  # [N, M, 2]
        area_enclosing = wh_enclosing[:, :, 0] * wh_enclosing[:, :, 1]  # [N, M]
        
        # Generalized IoU
        giou = iou - (area_enclosing - union) / area_enclosing
        
        return giou

    @staticmethod
    def detr_loss(pred_logits, pred_boxes, targets, num_classes, loss_weights=None):
        """
        DETR-style detection loss with Hungarian matching.
        
        Args:
            pred_logits: [B, num_queries, num_classes] or [num_layers, B, num_queries, num_classes] (with aux loss)
            pred_boxes: [B, num_queries, 4] or [num_layers, B, num_queries, 4] (with aux loss)
            targets: list of dicts with 'labels' [num_gt] and 'boxes' [num_gt, 4] for each image
            num_classes: number of object classes
            loss_weights: dict with loss component weights
            
        Returns:
            dict with loss components
        """
        if loss_weights is None:
            loss_weights = {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
        
        # Handle auxiliary losses
        if pred_logits.dim() == 4:  # [num_layers, B, num_queries, num_classes]
            num_layers = pred_logits.shape[0]
            aux_loss = True
        else:
            num_layers = 1
            aux_loss = False
            pred_logits = pred_logits.unsqueeze(0)
            pred_boxes = pred_boxes.unsqueeze(0)
        
        total_losses = {}
        
        for layer_idx in range(num_layers):
            layer_pred_logits = pred_logits[layer_idx]  # [B, num_queries, num_classes]
            layer_pred_boxes = pred_boxes[layer_idx]    # [B, num_queries, 4]
            
            # Extract ground truth
            gt_labels = [t['labels'] for t in targets]
            gt_boxes = [t['boxes'] for t in targets]
            
            # Hungarian matching
            indices = ObjectDetectionLosses.hungarian_matcher(
                layer_pred_logits, layer_pred_boxes, gt_labels, gt_boxes
            )
            
            # Compute losses
            losses = ObjectDetectionLosses.compute_loss_components(
                layer_pred_logits, layer_pred_boxes, gt_labels, gt_boxes, indices, num_classes
            )
            
            # Weight and accumulate losses
            layer_prefix = f"layer_{layer_idx}_" if aux_loss else ""
            for loss_name, loss_value in losses.items():
                weighted_loss = loss_weights.get(loss_name, 1.0) * loss_value
                total_losses[f"{layer_prefix}{loss_name}"] = weighted_loss
        
        # Final loss is sum of all components
        final_loss = sum(total_losses.values())
        total_losses['total_loss'] = final_loss
        
        return total_losses

    @staticmethod
    def compute_loss_components(pred_logits, pred_boxes, gt_labels, gt_boxes, indices, num_classes):
        """
        Compute individual loss components after Hungarian matching.
        
        Args:
            pred_logits: [B, num_queries, num_classes]
            pred_boxes: [B, num_queries, 4]
            gt_labels: list of ground truth labels for each image
            gt_boxes: list of ground truth boxes for each image
            indices: list of (pred_indices, gt_indices) tuples from Hungarian matching
            num_classes: number of object classes
            
        Returns:
            dict with loss components
        """
        B, num_queries = pred_logits.shape[:2]
        device = pred_logits.device
        
        # Prepare matched predictions and targets
        pred_indices_all = []
        gt_indices_all = []
        batch_indices = []
        
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                pred_indices_all.append(pred_idx)
                gt_indices_all.append(gt_idx)
                batch_indices.extend([b] * len(pred_idx))
        
        if len(pred_indices_all) == 0:
            # No matches found
            return {
                'loss_ce': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_bbox': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_giou': torch.tensor(0.0, device=device, requires_grad=True),
            }
        
        pred_indices_all = torch.cat(pred_indices_all)
        gt_indices_all = torch.cat(gt_indices_all)
        batch_indices = torch.tensor(batch_indices, device=device)
        
        # Classification loss
        target_classes = torch.full((B, num_queries), num_classes, dtype=torch.long, device=device)  # background class
        
        # Set matched targets
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = gt_labels[b][gt_idx]
        
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes)
        
        # Bounding box losses (only on matched predictions)
        if len(pred_indices_all) > 0:
            # Get matched predictions
            matched_pred_boxes = pred_boxes[batch_indices, pred_indices_all]
            
            # Get matched ground truth
            matched_gt_boxes = torch.cat([gt_boxes[b][gt_idx] for b, (_, gt_idx) in enumerate(indices) if len(gt_idx) > 0])
            
            # L1 loss
            loss_bbox = F.l1_loss(matched_pred_boxes, matched_gt_boxes, reduction='mean')
            
            # GIoU loss
            giou = ObjectDetectionLosses.generalized_box_iou(matched_pred_boxes, matched_gt_boxes)
            loss_giou = 1 - torch.diag(giou)  # Take diagonal for matched pairs
            loss_giou = loss_giou.mean()
        else:
            loss_bbox = torch.tensor(0.0, device=device, requires_grad=True)
            loss_giou = torch.tensor(0.0, device=device, requires_grad=True)
        
        return {
            'loss_ce': loss_ce,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
        }

    @staticmethod
    def focal_loss(pred_logits, targets, alpha=0.25, gamma=2.0):
        """
        Focal loss for addressing class imbalance in object detection.
        
        Args:
            pred_logits: [B, num_queries, num_classes] predicted logits
            targets: [B, num_queries] target class indices
            alpha: weighting factor for rare class
            gamma: focusing parameter
        """
        ce_loss = F.cross_entropy(pred_logits.transpose(1, 2), targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()


class SegmentationLosses:
    """ Segmentation losses for composite masks """
    
    @staticmethod
    def segmentation_bce_loss(pred_segmentation, gt_segmentation, m_frames=3, future_frame_weight=1.0):
        """
        Loss for segmentation masks (supports both binary and class-aware masks).
        
        Args:
            pred_segmentation: Predicted segmentation [B, T, H, W, 1]
            gt_segmentation: Ground truth segmentation [B, T, H, W, 1] or [T, H, W, 1]
                           Can be binary [0,1] or class-aware [0,1,2,3]
            m_frames: Number of current frames (first m frames)
            future_frame_weight: Weight multiplier for future frame losses
            
        Returns:
            Loss tensor
        """
        # Handle shape differences - gt might be [T, H, W, 1], pred is [B, T, H, W, 1]
        if gt_segmentation.dim() == 4 and pred_segmentation.dim() == 5:
            # Expand gt to match pred shape [B, T, H, W, 1]
            gt_segmentation = gt_segmentation.unsqueeze(0).expand_as(pred_segmentation)
        
        # Convert to float
        pred_segmentation = pred_segmentation.float()
        gt_segmentation = gt_segmentation.float()

        # Check if we have class-aware masks (values > 1)
        # Avoid .item() call which breaks DDP - just check tensor directly
        max_gt_value = gt_segmentation.max()
        
        if True:
            # Class-aware masks: use Cross Entropy loss for multi-class segmentation
            # GT shape: [B, T, H, W, 1] with class values [0,1,2,3]
            # Pred shape: [B, T, H, W, 4] with logits for each class
            
            # Remove channel dimension from GT and convert to long for CE loss
            gt_classes = gt_segmentation.squeeze(-1).long()  # [B, T, H, W]
            
            # Reshape for cross entropy: GT [B*T*H*W], Pred [B*T*H*W, 4]
            B, T, H, W = gt_classes.shape
            gt_flat = gt_classes.reshape(-1)  # [B*T*H*W]
            pred_flat = pred_segmentation.reshape(-1, pred_segmentation.size(-1))  # [B*T*H*W, 4]
            
            # Dynamically determine number of classes from predictions
            num_classes = pred_segmentation.shape[-1]  # Last dimension is number of classes
            class_weights = torch.ones(num_classes, device=gt_segmentation.device)

            # Class weighting based on number of classes
            if num_classes == 6:
                # GSAM2 classes: background, vehicle, bicycle, person, road sign, traffic light
                class_weights[0] = 0.2   # Low weight for background
                class_weights[1:] = 1.2  # Same weight for all object classes
                class_weights[3] = 1.6   # Person
                class_weights[4] = 1.8   # Road sign
                class_weights[5] = 1.8   # Traffic light
            elif num_classes == 7:
                # Cityscapes classes: road, vehicle, person, traffic light, traffic sign, sky, background
                class_weights[0] = 0.5   # Road (common but important)
                class_weights[1] = 1.2   # Vehicle
                class_weights[2] = 1.6   # Person
                class_weights[3] = 1.8   # Traffic light (small, important)
                class_weights[4] = 1.8   # Traffic sign (small, important)
                class_weights[5] = 0.3   # Sky (common, less important)
                class_weights[6] = 0.2   # Building/grass/background
            else:
                # Default: balance all classes except first (assumed background)
                class_weights[0] = 0.2
                class_weights[1:] = 1.2
            
            # Strategy 2: Option to use Focal Loss instead of weighted CE
            use_focal_loss = False  # Set to True to use focal loss for severe imbalance
            
            if use_focal_loss:
                # Focal Loss implementation - automatically handles class imbalance
                alpha = 0.25  # Weight for rare classes
                gamma = 2.0   # Focus on hard examples
                
                # Get class probabilities
                pred_probs = F.softmax(pred_flat, dim=1)  # [B*T*H*W, 4]
                
                # Gather probabilities for true classes
                true_class_probs = pred_probs.gather(1, gt_flat.unsqueeze(1)).squeeze(1)  # [B*T*H*W]
                
                # Compute focal loss
                focal_weight = alpha * (1 - true_class_probs) ** gamma
                ce_loss_raw = F.cross_entropy(pred_flat, gt_flat, weight=class_weights, reduction='none')
                focal_loss = focal_weight * ce_loss_raw
                
                # Apply future frame weighting if needed
                if future_frame_weight != 1.0 and T > m_frames:
                    # Create frame weights and reshape to match flattened tensor
                    frame_weights = torch.ones(T, device=pred_segmentation.device, dtype=pred_segmentation.dtype)
                    frame_weights[m_frames:] = future_frame_weight
                    # Expand to [B, T, H, W] then flatten
                    frame_weights = frame_weights.reshape(1, T, 1, 1).expand(B, T, H, W).reshape(-1)
                    focal_loss = focal_loss * frame_weights
                
                return focal_loss.mean()
            else:
                # Standard weighted cross entropy with actual ground truth
                if future_frame_weight != 1.0 and T > m_frames:
                    # Apply future frame weighting
                    ce_loss_raw = F.cross_entropy(pred_flat, gt_flat, weight=class_weights, reduction='none')
                    
                    # Create frame weights and reshape to match flattened tensor
                    frame_weights = torch.ones(T, device=pred_segmentation.device, dtype=pred_segmentation.dtype)
                    frame_weights[m_frames:] = future_frame_weight
                    # Expand to [B, T, H, W] then flatten
                    frame_weights = frame_weights.reshape(1, T, 1, 1).expand(B, T, H, W).reshape(-1)
                    ce_loss = (ce_loss_raw * frame_weights).mean()
                else:
                    ce_loss = F.cross_entropy(pred_flat, gt_flat, weight=class_weights, reduction='mean')
                return ce_loss
        else:
            # Binary masks: use BCE loss as before
            # Normalize gt_segmentation to [0, 1] if it's in [0, 255]
            if gt_segmentation.max() > 1.0:
                gt_segmentation = gt_segmentation / 255.0
            
            # Use binary_cross_entropy_with_logits which is autocast-safe (combines sigmoid + BCE)
            if future_frame_weight != 1.0 and pred_segmentation.shape[1] > m_frames:
                # Apply future frame weighting
                bce_loss_raw = F.binary_cross_entropy_with_logits(pred_segmentation, gt_segmentation, reduction='none')
                
                # Create frame weights
                B, T = pred_segmentation.shape[:2]
                frame_weights = torch.ones(T, device=pred_segmentation.device, dtype=pred_segmentation.dtype)
                frame_weights[m_frames:] = future_frame_weight
                frame_weights = frame_weights.reshape(1, T, 1, 1, 1).expand_as(bce_loss_raw)
                
                bce_loss = (bce_loss_raw * frame_weights).mean()
            else:
                bce_loss = F.binary_cross_entropy_with_logits(pred_segmentation, gt_segmentation, reduction='mean')
            return bce_loss
    
    @staticmethod
    def segmentation_focal_loss(pred_segmentation, gt_segmentation, alpha=1.0, gamma=2.0):
        """
        Focal loss for segmentation masks to handle class imbalance.
        
        Args:
            pred_segmentation: Predicted segmentation [B, T, H, W, 1]
            gt_segmentation: Ground truth segmentation [B, T, H, W, 1] or [T, H, W, 1]
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            
        Returns:
            Focal loss tensor
        """
        # Handle shape differences
        if gt_segmentation.dim() == 4 and pred_segmentation.dim() == 5:
            gt_segmentation = gt_segmentation.unsqueeze(0).expand_as(pred_segmentation)
        
        # Convert to float and normalize
        pred_segmentation = pred_segmentation.float()
        gt_segmentation = gt_segmentation.float()
        
        if gt_segmentation.max() > 1.0:
            gt_segmentation = gt_segmentation / 255.0
        
        # Compute focal loss using binary_cross_entropy_with_logits for autocast safety
        ce_loss = F.binary_cross_entropy_with_logits(pred_segmentation, gt_segmentation, reduction='none')
        
        # Apply sigmoid to get probabilities for focal weight calculation
        pred_probs = torch.sigmoid(pred_segmentation)
        p_t = pred_probs * gt_segmentation + (1 - pred_probs) * (1 - gt_segmentation)
        focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
        
        return focal_loss.mean()


class MotionLosses:
    """Motion losses for training motion head"""
    
    @staticmethod
    def motion_mse_loss(pred_motion, gt_motion):
        """
        MSE loss for motion prediction.
        
        Args:
            pred_motion: Predicted motion [B, T, H, W, 3] with 3D motion vectors
            gt_motion: Ground truth motion [B, T, H, W, 3] or [T, H, W, 3]
            
        Returns:
            MSE loss tensor
        """
        # Handle shape differences - gt might be [T, H, W, 3], pred is [B, T, H, W, 3]
        if gt_motion.dim() == 4 and pred_motion.dim() == 5:
            # Expand gt to match pred shape [B, T, H, W, 3]
            gt_motion = gt_motion.unsqueeze(0).expand_as(pred_motion)
        
        # Convert to float
        pred_motion = pred_motion.float()
        gt_motion = gt_motion.float()
        
        # Use MSE loss for motion vector regression
        mse_loss = F.mse_loss(pred_motion, gt_motion, reduction='mean')
        
        return mse_loss
    
    @staticmethod
    def motion_l1_loss(pred_motion, gt_motion):
        """
        L1 loss for motion prediction (more robust to outliers).
        
        Args:
            pred_motion: Predicted motion [B, T, H, W, 3] with 3D motion vectors
            gt_motion: Ground truth motion [B, T, H, W, 3] or [T, H, W, 3]
            
        Returns:
            L1 loss tensor
        """
        # Handle shape differences
        if gt_motion.dim() == 4 and pred_motion.dim() == 5:
            gt_motion = gt_motion.unsqueeze(0).expand_as(pred_motion)
        
        # Convert to float
        pred_motion = pred_motion.float()
        gt_motion = gt_motion.float()
        
        # Use L1 loss for motion vector regression
        l1_loss = F.l1_loss(pred_motion, gt_motion, reduction='mean')
        
        return l1_loss
    
    @staticmethod
    def motion_smooth_l1_loss(pred_motion, gt_motion, beta=1.0):
        """
        Smooth L1 loss for motion prediction (combines L1 and L2 benefits).
        
        Args:
            pred_motion: Predicted motion [B, T, H, W, 3] with 3D motion vectors
            gt_motion: Ground truth motion [B, T, H, W, 3] or [T, H, W, 3]
            beta: Threshold for switching between L1 and L2 loss
            
        Returns:
            Smooth L1 loss tensor
        """
        # Handle shape differences
        if gt_motion.dim() == 4 and pred_motion.dim() == 5:
            gt_motion = gt_motion.unsqueeze(0).expand_as(pred_motion)
        
        # Convert to float
        pred_motion = pred_motion.float()
        gt_motion = gt_motion.float()

        # visualization for debugging
        # import matplotlib.pyplot as plt
        # plt.subplot(1,2,1)
        # plt.imshow(pred_motion[0,0,:,:,0].detach().cpu().numpy())
        # plt.subplot(1,2,2)
        # plt.imshow(gt_motion[0,0,:,:,0].cpu().numpy())
        # plt.show()
        
        # Use smooth L1 loss for motion vector regression
        smooth_l1_loss = F.smooth_l1_loss(pred_motion, gt_motion, beta=beta, reduction='mean')
        
        return smooth_l1_loss
    
    @staticmethod
    def motion_cosine_similarity_loss(pred_motion, gt_motion):
        """
        Cosine similarity loss for motion direction prediction.
        Focuses on motion direction rather than magnitude.
        
        Args:
            pred_motion: Predicted motion [B, T, H, W, 3] with 3D motion vectors
            gt_motion: Ground truth motion [B, T, H, W, 3] or [T, H, W, 3]
            
        Returns:
            Cosine similarity loss tensor (1 - cosine_similarity)
        """
        # Handle shape differences
        if gt_motion.dim() == 4 and pred_motion.dim() == 5:
            gt_motion = gt_motion.unsqueeze(0).expand_as(pred_motion)
        
        # Convert to float
        pred_motion = pred_motion.float()
        gt_motion = gt_motion.float()
        
        # Flatten spatial dimensions for cosine similarity computation
        pred_flat = pred_motion.view(-1, 3)  # [B*T*H*W, 3]
        gt_flat = gt_motion.view(-1, 3)      # [B*T*H*W, 3]
        
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(pred_flat, gt_flat, dim=1)  # [B*T*H*W]
        
        # Convert to loss (1 - similarity)
        cosine_loss = 1.0 - cosine_sim.mean()
        
        return cosine_loss
    
    @staticmethod
    def motion_magnitude_loss(pred_motion, gt_motion):
        """
        Motion magnitude loss - focuses on the magnitude of motion vectors.
        
        Args:
            pred_motion: Predicted motion [B, T, H, W, 3] with 3D motion vectors
            gt_motion: Ground truth motion [B, T, H, W, 3] or [T, H, W, 3]
            
        Returns:
            Magnitude loss tensor
        """
        # Handle shape differences
        if gt_motion.dim() == 4 and pred_motion.dim() == 5:
            gt_motion = gt_motion.unsqueeze(0).expand_as(pred_motion)
        
        # Convert to float
        pred_motion = pred_motion.float()
        gt_motion = gt_motion.float()
        
        # Compute magnitudes
        pred_magnitude = torch.norm(pred_motion, dim=-1)  # [B, T, H, W]
        gt_magnitude = torch.norm(gt_motion, dim=-1)      # [B, T, H, W]
        
        # Use L1 loss on magnitudes
        magnitude_loss = F.l1_loss(pred_magnitude, gt_magnitude, reduction='mean')
        
        return magnitude_loss


class FrozenDecoderSupervision:
    """Supervision losses using frozen model decoder features."""
    
    @staticmethod
    def decoder_feature_loss(predictions, gt, m_frames=3):
        """
        Supervision loss between frozen model decoder features and autoregressive model features.
        
        Args:
            predictions: Dict containing autoregressive decoder features
            gt: Dict containing frozen model decoder features  
            m_frames: Number of current frames
            
        Returns:
            Loss tensor
        """
        if 'features' not in gt or 'all_decoder_features' not in predictions:
            return torch.tensor(0.0, device=predictions['points'].device)
        
        frozen_features = gt['features']                           # [B*M, S, D] - teacher (current frames only)
        all_features = predictions['all_decoder_features']        # [B*(M+N), S, D] - student (current + future)
        
        # Extract current frames from autoregressive features to match frozen features
        # all_features contains [current_frames, future_frames], we want [current_frames]
        BM = frozen_features.shape[0]  # B * M (current frames)
        current_features = all_features[:BM]  # [B*M, S, D] - student current frames
        
        # Ensure shapes match
        if frozen_features.shape != current_features.shape:
            print(f"⚠️  Feature shape mismatch: frozen {frozen_features.shape} vs current {current_features.shape}")
            return torch.tensor(0.0, device=predictions['points'].device)
        
        # L2 loss between frozen (teacher) and current (student) decoder features
        feature_loss = F.mse_loss(current_features, frozen_features)
        
        return feature_loss


# Example usage
if __name__ == "__main__":
    # Test depth losses
    batch_size, height, width = 2, 64, 64
    pred_depth = torch.randn(batch_size, 1, height, width) * 10 + 5
    gt_depth = torch.randn(batch_size, 1, height, width) * 10 + 5
    mask = torch.rand(batch_size, 1, height, width) > 0.2
    
    depth_losses = DepthLosses()
    l1_loss = depth_losses.l1_loss(pred_depth, gt_depth, mask)
    berhu_loss = depth_losses.berhu_loss(pred_depth, gt_depth, mask)
    scale_inv_loss = depth_losses.scale_invariant_loss(pred_depth, gt_depth, mask)
    
    print(f"L1 Loss: {l1_loss.item():.4f}")
    print(f"BerHu Loss: {berhu_loss.item():.4f}")
    print(f"Scale Invariant Loss: {scale_inv_loss.item():.4f}")
    
    # Test point cloud losses
    pred_points = torch.randn(batch_size, 1000, 3)
    gt_points = torch.randn(batch_size, 1000, 3)
    
    pc_losses = PointCloudLosses()
    chamfer_dist = pc_losses.chamfer_distance(pred_points, gt_points)
    print(f"Chamfer Distance: {chamfer_dist.item():.4f}")
    
    # Test pose losses
    pred_poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    gt_poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    # Add some noise
    pred_poses[:, :3, 3] += torch.randn(batch_size, 3) * 0.1
    
    pose_losses = CameraPoseLosses()
    trans_loss = pose_losses.translation_loss(pred_poses[:, :3, 3], gt_poses[:, :3, 3])
    rot_loss = pose_losses.rotation_loss(pred_poses[:, :3, :3], gt_poses[:, :3, :3])
    
    print(f"Translation Loss: {trans_loss.item():.4f}")
    print(f"Rotation Loss: {rot_loss.item():.4f}")
    
    # Test object detection losses
    B, num_queries, num_classes = 2, 100, 80
    pred_logits = torch.randn(B, num_queries, num_classes)
    pred_boxes = torch.rand(B, num_queries, 4)  # normalized [0,1]
    
    # Mock targets
    targets = [
        {'labels': torch.tensor([1, 5, 10]), 'boxes': torch.rand(3, 4)},
        {'labels': torch.tensor([2, 8]), 'boxes': torch.rand(2, 4)},
    ]
    
    detection_losses = ObjectDetectionLosses()
    loss_dict = detection_losses.detr_loss(pred_logits, pred_boxes, targets, num_classes)
    
    print("\nObject Detection Losses:")
    for k, v in loss_dict.items():
        if torch.is_tensor(v):
            print(f"  {k}: {v.item():.4f}")
        else:
            print(f"  {k}: {v}")