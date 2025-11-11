"""
PPGeo loss functions for self-supervised depth and pose learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import math


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images."""
    
    def __init__(self):
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        
        self.refl = nn.ReflectionPad2d(1)
        
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
    
    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud."""
    
    def __init__(self, batch_size, height, width):
        super().__init__()
        
        self.batch_size = batch_size
        self.height = height
        self.width = width
        
        meshgrid = torch.meshgrid(torch.arange(self.width), torch.arange(self.height), indexing='xy')
        self.id_coords = torch.stack(meshgrid, dim=0).float()
        self.ones = torch.ones(self.batch_size, 1, self.height * self.width)
        
        self.pix_coords = torch.unsqueeze(torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = torch.cat([self.pix_coords, self.ones], 1)
    
    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords.to(depth.device))
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones.to(depth.device)], 1)
        
        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T."""
    
    def __init__(self, batch_size, height, width, eps=1e-7):
        super().__init__()
        
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
    
    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        
        cam_points = torch.matmul(P, points)
        
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        
        return pix_coords


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix."""
    R = rot_from_axisangle(axisangle)
    t = translation.clone()
    
    if invert:
        R = R.transpose(1, 2)
        t *= -1
    
    T = get_translation_matrix(t)
    
    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)
    
    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix."""
    T = torch.zeros(translation_vector.shape[0], 4, 4, device=translation_vector.device, dtype=translation_vector.dtype)
    t = translation_vector.contiguous().view(-1, 3, 1)
    
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t
    
    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix."""
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)
    
    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca
    
    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)
    
    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    
    rot = torch.zeros((vec.shape[0], 4, 4), device=vec.device, dtype=vec.dtype)
    
    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1
    
    return rot


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction."""
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image."""
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
    
    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
    
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    
    return grad_disp_x.mean() + grad_disp_y.mean()


class PPGeoLoss(nn.Module):
    """PPGeo photometric and smoothness losses."""
    
    def __init__(self, scales: List[int] = [0, 1, 2, 3], frame_ids: List[int] = [-1, 0, 1]):
        super().__init__()
        
        self.scales = scales
        self.frame_ids = frame_ids
        self.ssim = SSIM()
        
        # Initialize projection layers for each scale
        self.backproject_depth = {}
        self.project_3d = {}
        
    def init_projection_layers(self, batch_size, height, width, device):
        """Initialize projection layers for the given dimensions."""
        for scale in self.scales:
            h = height // (2 ** scale)
            w = width // (2 ** scale)
            
            self.backproject_depth[scale] = BackprojectDepth(batch_size, h, w)
            self.project_3d[scale] = Project3D(batch_size, h, w)
            
            # Move to device
            self.backproject_depth[scale].pix_coords = self.backproject_depth[scale].pix_coords.to(device)
            self.backproject_depth[scale].ones = self.backproject_depth[scale].ones.to(device)
    
    def compute_reprojection_loss(self, pred, target):
        """Compute reprojection loss between predicted and target images."""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        
        return reprojection_loss
    
    def build_camera_matrix(self, intrinsics_pred, height, width):
        """Build camera intrinsic matrix from predictions."""
        B = intrinsics_pred.shape[0]
        
        # Extract predicted intrinsics (normalized)
        fx_norm, fy_norm, cx_norm, cy_norm = intrinsics_pred[:, 0], intrinsics_pred[:, 1], intrinsics_pred[:, 2], intrinsics_pred[:, 3]
        
        # Convert to actual pixel coordinates
        fx = fx_norm * width
        fy = fy_norm * height
        cx = cx_norm * width
        cy = cy_norm * height
        
        # Build camera matrix
        K = torch.zeros(B, 4, 4, device=intrinsics_pred.device, dtype=intrinsics_pred.dtype)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1
        K[:, 3, 3] = 1
        
        return K
    
    def forward(self, outputs: Dict[str, torch.Tensor], inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute PPGeo losses."""
        images = inputs["images"]  # [B, 3, 3, H, W]
        B, num_frames, C, H, W = images.shape
        device = images.device
        
        # Initialize projection layers if needed
        if not hasattr(self, '_initialized') or not self._initialized:
            self.init_projection_layers(B, H, W, device)
            self._initialized = True
        
        # Extract frame images
        img_prev = images[:, 0]  # [B, 3, H, W]
        img_curr = images[:, 1]  # [B, 3, H, W]
        img_next = images[:, 2]  # [B, 3, H, W]
        
        # Build camera matrix from predicted intrinsics
        K = self.build_camera_matrix(outputs[("intrinsics", 0)], H, W)
        inv_K = torch.linalg.pinv(K)
        
        # Build transformation matrices
        T_prev = transformation_from_parameters(
            outputs[("axisangle", 0, -1)], outputs[("translation", 0, -1)], invert=True)
        T_next = transformation_from_parameters(
            outputs[("axisangle", 0, 1)], outputs[("translation", 0, 1)], invert=False)
        
        total_loss = 0
        losses = {}
        
        for scale in self.scales:
            loss = 0
            reprojection_losses = []
            
            # Get disparity and convert to depth
            disp = outputs[("disp", scale)]
            _, depth = disp_to_depth(disp, 0.1, 100.0)
            
            # Resize depth to full resolution for projection
            if scale > 0:
                depth_full = F.interpolate(depth, [H, W], mode="bilinear", align_corners=False)
            else:
                depth_full = depth
            
            # Project current frame to adjacent frames
            for frame_id, T in [(-1, T_prev), (1, T_next)]:
                # Backproject depth
                cam_points = self.backproject_depth[0](depth_full, inv_K)
                
                # Project to target frame
                pix_coords = self.project_3d[0](cam_points, K, T)
                
                # Sample target frame
                if frame_id == -1:
                    target_img = img_prev
                else:
                    target_img = img_next
                
                # Sample the target image at projected coordinates
                projected_img = F.grid_sample(
                    target_img, pix_coords, padding_mode="border", align_corners=True)
                
                # Compute reprojection loss
                reprojection_loss = self.compute_reprojection_loss(projected_img, img_curr)
                reprojection_losses.append(reprojection_loss)
            
            # Combine reprojection losses
            reprojection_losses = torch.cat(reprojection_losses, 1)
            
            # Compute identity reprojection (for masking)
            identity_losses = []
            for frame_id in [-1, 1]:
                if frame_id == -1:
                    identity_loss = self.compute_reprojection_loss(img_prev, img_curr)
                else:
                    identity_loss = self.compute_reprojection_loss(img_next, img_curr)
                identity_losses.append(identity_loss)
            
            identity_losses = torch.cat(identity_losses, 1)
            
            # Add noise to identity loss to avoid degeneracy
            identity_losses += torch.randn(identity_losses.shape, device=device) * 0.00001
            
            # Combine with reprojection losses
            combined = torch.cat((identity_losses, reprojection_losses), dim=1)
            
            # Take minimum loss (auto-masking)
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, _ = torch.min(combined, dim=1)
            
            loss += to_optimise.mean()
            
            # Smoothness loss
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, img_curr)
            
            loss += 1e-3 * smooth_loss / (2 ** scale)
            
            total_loss += loss
            losses[f"loss_scale_{scale}"] = loss
        
        total_loss /= len(self.scales)
        
        losses.update({
            "total_loss": total_loss,
            "reprojection_loss": reprojection_losses.mean(),
            "smoothness_loss": smooth_loss
        })
        
        return losses