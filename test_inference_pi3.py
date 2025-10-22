import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from PIL import Image
import json
from pathlib import Path

import torchvision.transforms as T
from tqdm import tqdm

# Add Pi3 to path
import sys
sys.path.append("/home/matthew_strong/Desktop/autonomy-wild/Pi3")

from pi3.models.pi3 import Pi3, AutonomyPi3
from pi3.utils.basic import load_images_as_tensor
from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

# Import dataset for loading test data
from SpaTrackerV2.ssl_image_dataset import SequenceLearningDataset
from SpaTrackerV2.multi_folder_consecutive_images_dataset import get_default_transforms

# Import losses for comparison
from losses import Pi3Losses


class Pi3InferenceVisualizer:
    """
    Inference and visualization pipeline for Pi3 models.
    """
    
    def __init__(self, checkpoint_path, m=3, n=3, device='cuda'):
        """
        Initialize the inference visualizer.
        
        Args:
            checkpoint_path (str): Path to trained model checkpoint
            m (int): Number of input frames
            n (int): Number of target frames  
            device (str): Device to run inference on
        """
        self.checkpoint_path = checkpoint_path
        self.m = m
        self.n = n
        self.device = device
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Load frozen model for comparison
        self.frozen_model = Pi3.from_pretrained("yyfz233/Pi3")
        self.frozen_model = self.frozen_model.to(device)
        self.frozen_model.requires_grad_(False)
        self.frozen_model.eval()
        
        print(f"✅ Models loaded successfully!")
        print(f"📊 Input frames: {m}, Target frames: {n}")
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        print(f"📂 Loading checkpoint from: {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Initialize model architecture
        model = AutonomyPi3(full_N=self.m + self.n, extra_tokens=self.n)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📈 Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"🎯 Best loss: {checkpoint.get('best_loss', 'unknown')}")
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        return model
    
    def run_inference(self, video_tensor):
        """
        Run inference on video tensor.
        
        Args:
            video_tensor: Tensor of shape (1, T, C, H, W)
            
        Returns:
            dict: Inference results with predictions and ground truth
        """
        with torch.no_grad():
            # Move video tensor to correct device
            video_tensor = video_tensor.to(self.device)
            
            # Get input frames (first m frames)
            input_frames = video_tensor[:, :self.m]  # (1, m, C, H, W)
            
            # Run trained model inference
            predictions = self.model(input_frames)
            
            # Run frozen model for ground truth comparison
            gt_full = self.frozen_model(video_tensor)  # Use all frames for GT
            
            # Extract relevant parts for comparison
            gt_current_future = {
                'points': gt_full['points'][:, :self.m + self.n],  # Current + future frames
                'local_points': gt_full['local_points'][:, :self.m + self.n] if 'local_points' in gt_full else None,  # 2D image space points
                'camera_poses': gt_full['camera_poses'][:, :self.m + self.n],
                'conf': gt_full['conf'][:, :self.m + self.n] if 'conf' in gt_full else None  # Confidence maps
            }
            
            return {
                'predictions': predictions,
                'ground_truth': gt_current_future,
                'input_frames': input_frames,
                'all_frames': video_tensor
            }
    
    def visualize_point_clouds(self, results, save_dir=None, frame_idx=0):
        """
        Visualize predicted vs ground truth point clouds.
        
        Args:
            results: Results from run_inference
            save_dir: Directory to save visualizations
            frame_idx: Which frame to visualize
        """
        pred_points = results['predictions']['points'][0, frame_idx].cpu().numpy()  # (H, W, 3)
        gt_points = results['ground_truth']['points'][0, frame_idx].cpu().numpy()   # (H, W, 3)
        
        H, W, _ = pred_points.shape
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 8))
        
        # Original image
        ax1 = fig.add_subplot(141)
        input_img = results['input_frames'][0, min(frame_idx, self.m-1)].cpu()
        input_img = input_img.permute(1, 2, 0).numpy()
        input_img = np.clip(input_img, 0, 1)
        ax1.imshow(input_img)
        ax1.set_title(f'Input Frame {min(frame_idx, self.m-1)}')
        ax1.axis('off')
        
        # Predicted depth map
        ax2 = fig.add_subplot(142)
        pred_depth = pred_points[:, :, 2]  # Z component
        im2 = ax2.imshow(pred_depth, cmap='viridis')
        ax2.set_title(f'Predicted Depth (Frame {frame_idx})')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Ground truth depth map
        ax3 = fig.add_subplot(143)
        gt_depth = gt_points[:, :, 2]  # Z component
        im3 = ax3.imshow(gt_depth, cmap='viridis')
        ax3.set_title(f'Ground Truth Depth (Frame {frame_idx})')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # Depth difference
        ax4 = fig.add_subplot(144)
        depth_diff = np.abs(pred_depth - gt_depth)
        im4 = ax4.imshow(depth_diff, cmap='hot')
        ax4.set_title(f'Depth Error (Frame {frame_idx})')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'point_clouds_frame_{frame_idx}.png'), 
                       dpi=150, bbox_inches='tight')
        
        plt.show()
        
        # 3D point cloud visualization
        self._visualize_3d_points(pred_points, gt_points, frame_idx, save_dir)
    
    def _visualize_3d_points(self, pred_points, gt_points, frame_idx, save_dir=None):
        """Visualize 3D point clouds."""
        fig = plt.figure(figsize=(15, 6))
        
        # Sample points for visualization (too many points slow down rendering)
        H, W = pred_points.shape[:2]
        step = max(1, H // 50)  # Sample every 'step' pixels
        
        y_coords, x_coords = np.mgrid[0:H:step, 0:W:step]
        
        # Extract sampled points
        pred_sampled = pred_points[::step, ::step].reshape(-1, 3)
        gt_sampled = gt_points[::step, ::step].reshape(-1, 3)
        
        # Remove invalid points (depth <= 0)
        valid_pred = pred_sampled[:, 2] > 0
        valid_gt = gt_sampled[:, 2] > 0
        
        pred_valid = pred_sampled[valid_pred]
        gt_valid = gt_sampled[valid_gt]
        
        # Predicted 3D points
        ax1 = fig.add_subplot(121, projection='3d')
        if len(pred_valid) > 0:
            ax1.scatter(pred_valid[:, 0], pred_valid[:, 1], pred_valid[:, 2], 
                       c=pred_valid[:, 2], cmap='viridis', s=1, alpha=0.6)
        ax1.set_title(f'Predicted 3D Points (Frame {frame_idx})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z (Depth)')
        
        # Ground truth 3D points
        ax2 = fig.add_subplot(122, projection='3d')
        if len(gt_valid) > 0:
            ax2.scatter(gt_valid[:, 0], gt_valid[:, 1], gt_valid[:, 2], 
                       c=gt_valid[:, 2], cmap='viridis', s=1, alpha=0.6)
        ax2.set_title(f'Ground Truth 3D Points (Frame {frame_idx})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z (Depth)')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'3d_points_frame_{frame_idx}.png'), 
                       dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def visualize_local_points(self, results, save_dir=None, frame_idx=0):
        """
        Visualize predicted vs ground truth local points (2D image space).
        
        Args:
            results: Results from run_inference
            save_dir: Directory to save visualizations
            frame_idx: Which frame to visualize
        """
        # Check if local_points are available
        if 'local_points' not in results['predictions']:
            print("Warning: No local_points found in predictions")
            return
            
        if results['ground_truth']['local_points'] is None:
            print("Warning: No local_points found in ground truth")
            return
            
        pred_local = results['predictions']['local_points'][0, frame_idx].cpu().numpy()  # (H, W, 3)
        gt_local = results['ground_truth']['local_points'][0, frame_idx].cpu().numpy()   # (H, W, 3)
        
        H, W, _ = pred_local.shape
        
        # Get original image for overlay
        original_image = results['all_frames'][0, frame_idx].cpu().permute(1, 2, 0).numpy()
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_image = original_image * std + mean
        original_image = np.clip(original_image, 0, 1)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Original image
        ax1 = fig.add_subplot(231)
        ax1.imshow(original_image)
        ax1.set_title(f'Original Image (Frame {frame_idx})')
        ax1.axis('off')
        
        # Predicted local depth (Z component in image space)
        ax2 = fig.add_subplot(232)
        pred_depth_local = pred_local[:, :, 2]  # Z component in image space
        im2 = ax2.imshow(pred_depth_local, cmap='viridis')
        ax2.set_title(f'Predicted Local Depth (Frame {frame_idx})')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)
        
        # Ground truth local depth
        ax3 = fig.add_subplot(233)
        gt_depth_local = gt_local[:, :, 2]  # Z component in image space
        im3 = ax3.imshow(gt_depth_local, cmap='viridis')
        ax3.set_title(f'Ground Truth Local Depth (Frame {frame_idx})')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3)
        
        # Depth difference
        ax4 = fig.add_subplot(234)
        depth_diff = np.abs(pred_depth_local - gt_depth_local)
        im4 = ax4.imshow(depth_diff, cmap='hot')
        ax4.set_title(f'Local Depth Difference (Frame {frame_idx})')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4)
        
        # X-coordinate comparison (horizontal displacement)
        ax5 = fig.add_subplot(235)
        pred_x = pred_local[:, :, 0]
        gt_x = gt_local[:, :, 0]
        x_diff = np.abs(pred_x - gt_x)
        im5 = ax5.imshow(x_diff, cmap='plasma')
        ax5.set_title(f'X-coordinate Difference (Frame {frame_idx})')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5)
        
        # Y-coordinate comparison (vertical displacement)
        ax6 = fig.add_subplot(236)
        pred_y = pred_local[:, :, 1]
        gt_y = gt_local[:, :, 1]
        y_diff = np.abs(pred_y - gt_y)
        im6 = ax6.imshow(y_diff, cmap='plasma')
        ax6.set_title(f'Y-coordinate Difference (Frame {frame_idx})')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'local_points_frame_{frame_idx}.png'), 
                       dpi=150, bbox_inches='tight')
        
        plt.show()
        
        # Print statistics
        print(f"\n📊 Local Points Statistics (Frame {frame_idx}):")
        print(f"   Local Depth - Mean Error: {np.mean(depth_diff):.4f}, Std: {np.std(depth_diff):.4f}")
        print(f"   X-coordinate - Mean Error: {np.mean(x_diff):.4f}, Std: {np.std(x_diff):.4f}")  
        print(f"   Y-coordinate - Mean Error: {np.mean(y_diff):.4f}, Std: {np.std(y_diff):.4f}")
    
    def visualize_camera_poses(self, results, save_dir=None):
        """
        Visualize predicted vs ground truth camera poses.
        
        Args:
            results: Results from run_inference
            save_dir: Directory to save visualizations
        """
        # Handle predictions and ground truth extraction
        if 'camera_poses' in results['predictions']:
            pred_poses = results['predictions']['camera_poses'][0].cpu().numpy()  # (N+M, 4, 4)
        else:
            print("Warning: No camera_poses found in predictions")
            return {'mean_position_error': 0, 'mean_rotation_error': 0, 'current_position_error': 0, 'future_position_error': 0}
            
        if 'camera_poses' in results['ground_truth']:
            gt_poses = results['ground_truth']['camera_poses'][0].cpu().numpy()   # (N+M, 4, 4)
        else:
            print("Warning: No camera_poses found in ground truth")
            return {'mean_position_error': 0, 'mean_rotation_error': 0, 'current_position_error': 0, 'future_position_error': 0}
        
        # Extract positions and orientations
        pred_positions = pred_poses[:, :3, 3]  # (N+M, 3)
        gt_positions = gt_poses[:, :3, 3]      # (N+M, 3)
        
        pred_rotations = pred_poses[:, :3, :3]  # (N+M, 3, 3)
        gt_rotations = gt_poses[:, :3, :3]      # (N+M, 3, 3)
        
        try:
            # Create visualization
            fig = plt.figure(figsize=(20, 12))
            
            # 3D trajectory plot
            ax1 = fig.add_subplot(231, projection='3d')
        
            # Plot trajectories
            ax1.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], 
                    'r-o', label='Predicted', markersize=4, linewidth=2)
            ax1.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
                    'b-s', label='Ground Truth', markersize=4, linewidth=2)
            
            # Mark current vs future frames
            current_end = self.m
            ax1.plot(pred_positions[:current_end, 0], pred_positions[:current_end, 1], 
                    pred_positions[:current_end, 2], 'ro', markersize=8, label='Current (Pred)')
            ax1.plot(pred_positions[current_end:, 0], pred_positions[current_end:, 1], 
                    pred_positions[current_end:, 2], 'r^', markersize=8, label='Future (Pred)')
            
            ax1.set_title('Camera Trajectories')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y') 
            ax1.set_zlabel('Z')
            ax1.legend()
        
            # Position error over time
            ax2 = fig.add_subplot(232)
            position_errors = np.linalg.norm(pred_positions - gt_positions, axis=1)
            frames = np.arange(len(position_errors))
            
            ax2.plot(frames, position_errors, 'g-o', linewidth=2, markersize=4)
            ax2.axvline(x=current_end-0.5, color='r', linestyle='--', alpha=0.7, label='Current|Future')
            ax2.set_title('Position Error Over Time')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Position Error (L2 norm)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Rotation error over time (geodesic distance)
            ax3 = fig.add_subplot(233)
            rotation_errors = []
            for i in range(len(pred_rotations)):
                # Compute geodesic distance between rotation matrices
                rel_rot = pred_rotations[i] @ gt_rotations[i].T
                trace = np.trace(rel_rot)
                # Clamp trace to valid range for arccos
                trace = np.clip(trace, -1, 3)  # trace(R) ∈ [-1, 3] for rotation matrices
                angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                rotation_errors.append(np.degrees(angle))
            
            ax3.plot(frames, rotation_errors, 'm-o', linewidth=2, markersize=4)
            ax3.axvline(x=current_end-0.5, color='r', linestyle='--', alpha=0.7, label='Current|Future')
            ax3.set_title('Rotation Error Over Time')
            ax3.set_xlabel('Frame')
            ax3.set_ylabel('Rotation Error (degrees)')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # X, Y, Z position components
            ax4 = fig.add_subplot(234)
            ax4.plot(frames, pred_positions[:, 0], 'r-', label='Pred X', linewidth=2)
            ax4.plot(frames, gt_positions[:, 0], 'r--', label='GT X', linewidth=2)
            ax4.plot(frames, pred_positions[:, 1], 'g-', label='Pred Y', linewidth=2)
            ax4.plot(frames, gt_positions[:, 1], 'g--', label='GT Y', linewidth=2)
            ax4.plot(frames, pred_positions[:, 2], 'b-', label='Pred Z', linewidth=2)
            ax4.plot(frames, gt_positions[:, 2], 'b--', label='GT Z', linewidth=2)
            ax4.axvline(x=current_end-0.5, color='k', linestyle='--', alpha=0.7)
            ax4.set_title('Position Components')
            ax4.set_xlabel('Frame')
            ax4.set_ylabel('Position')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Top-down view (X-Y plane)
            ax5 = fig.add_subplot(235)
            ax5.plot(pred_positions[:, 0], pred_positions[:, 1], 'r-o', 
                    label='Predicted', markersize=4, linewidth=2)
            ax5.plot(gt_positions[:, 0], gt_positions[:, 1], 'b-s', 
                    label='Ground Truth', markersize=4, linewidth=2)
            
            # Mark start and end
            ax5.plot(pred_positions[0, 0], pred_positions[0, 1], 'go', markersize=10, label='Start')
            ax5.plot(pred_positions[-1, 0], pred_positions[-1, 1], 'ro', markersize=10, label='End')
            
            ax5.set_title('Top-Down View (X-Y)')
            ax5.set_xlabel('X')
            ax5.set_ylabel('Y')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.axis('equal')
            
            # Summary statistics
            ax6 = fig.add_subplot(236)
            ax6.axis('off')
            
            # Calculate statistics
            mean_pos_error = np.mean(position_errors)
            max_pos_error = np.max(position_errors)
            mean_rot_error = np.mean(rotation_errors)
            max_rot_error = np.max(rotation_errors)
            
            current_pos_error = np.mean(position_errors[:current_end])
            future_pos_error = np.mean(position_errors[current_end:])
            current_rot_error = np.mean(rotation_errors[:current_end])
            future_rot_error = np.mean(rotation_errors[current_end:])
            
            stats_text = f"""
            SUMMARY STATISTICS
            
            Position Error:
            • Mean: {mean_pos_error:.4f}
            • Max: {max_pos_error:.4f}
            • Current frames: {current_pos_error:.4f}
            • Future frames: {future_pos_error:.4f}
            
            Rotation Error (degrees):
            • Mean: {mean_rot_error:.2f}°
            • Max: {max_rot_error:.2f}°
            • Current frames: {current_rot_error:.2f}°
            • Future frames: {future_rot_error:.2f}°
            
            Model Info:
            • Input frames: {self.m}
            • Target frames: {self.n}
            • Total frames: {self.m + self.n}
            """
            
            ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, 'camera_poses.png'), 
                           dpi=150, bbox_inches='tight')
            
            plt.show()
            
            return {
                'mean_position_error': mean_pos_error,
                'mean_rotation_error': mean_rot_error,
                'current_position_error': current_pos_error,
                'future_position_error': future_pos_error
            }
            
        except Exception as e:
            print(f"Error in camera pose visualization: {e}")
            print(f"Pred poses shape: {pred_poses.shape}, GT poses shape: {gt_poses.shape}")
            return {
                'mean_position_error': 0.0,
                'mean_rotation_error': 0.0,
                'current_position_error': 0.0,
                'future_position_error': 0.0
            }
    
    def compute_losses(self, results):
        """Compute losses between predictions and ground truth."""
        with torch.no_grad():
            # Move to same device
            predictions = {k: v.to(self.device) if v is not None else None for k, v in results['predictions'].items()}
            ground_truth = {k: v.to(self.device) if v is not None else None for k, v in results['ground_truth'].items()}
            
            # Compute losses - Pi3Losses.pi3_loss returns 3 values: point_map_loss, camera_pose_loss, confidence_loss
            point_map_loss, camera_pose_loss, confidence_loss = Pi3Losses.pi3_loss(predictions, ground_truth)
            total_loss = point_map_loss + camera_pose_loss + confidence_loss
            
            return {
                'total_loss': total_loss.item(),
                'point_map_loss': point_map_loss.item(),
                'camera_pose_loss': camera_pose_loss.item(),
                'confidence_loss': confidence_loss.item() if isinstance(confidence_loss, torch.Tensor) else confidence_loss
            }
    
    def run_evaluation(self, dataloader, num_samples=10, save_dir='./inference_results'):
        """
        Run evaluation on multiple samples.
        
        Args:
            dataloader: DataLoader with test data
            num_samples: Number of samples to evaluate
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        all_losses = []
        all_pose_errors = []
        
        print(f"🔍 Running evaluation on {num_samples} samples...")
        
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if i >= num_samples:
                break
            
            # Get video tensor
            X = batch[0]  # Current frames
            y = batch[1]  # Future frames
            video_tensor = torch.cat([X, y], dim=1)  # (1, T, C, H, W)
            video_tensor = preprocess_image(video_tensor.squeeze(0))[None]  # Preprocess
            
            # Run inference
            results = self.run_inference(video_tensor)
            
            # Compute losses
            losses = self.compute_losses(results)
            all_losses.append(losses)
            
            # Visualize every 5th sample
            if i % 5 == 0:
                sample_dir = os.path.join(save_dir, f'sample_{i:03d}')
                
                # Visualize point clouds for current and future frames
                for frame_idx in [0, self.m//2, self.m + self.n//2]:
                    if frame_idx < self.m + self.n:
                        self.visualize_point_clouds(results, sample_dir, frame_idx)
                        # Also visualize local points if available
                        self.visualize_local_points(results, sample_dir, frame_idx)
                
                # Visualize camera poses
                pose_errors = self.visualize_camera_poses(results, sample_dir)
                all_pose_errors.append(pose_errors)
        
        # Summary statistics
        self._print_evaluation_summary(all_losses, all_pose_errors, save_dir)
    
    def _print_evaluation_summary(self, all_losses, all_pose_errors, save_dir):
        """Print and save evaluation summary."""
        # Loss statistics
        total_losses = [l['total_loss'] for l in all_losses]
        point_losses = [l['point_map_loss'] for l in all_losses]
        pose_losses = [l['camera_pose_loss'] for l in all_losses]
        
        # Pose error statistics
        if all_pose_errors:
            mean_pos_errors = [pe['mean_position_error'] for pe in all_pose_errors]
            mean_rot_errors = [pe['mean_rotation_error'] for pe in all_pose_errors]
            current_pos_errors = [pe['current_position_error'] for pe in all_pose_errors]
            future_pos_errors = [pe['future_position_error'] for pe in all_pose_errors]
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 EVALUATION SUMMARY ({len(all_losses)} samples)")
        print(f"{'='*60}")
        
        print(f"\n🔥 LOSS STATISTICS:")
        print(f"  Total Loss:     {np.mean(total_losses):.6f} ± {np.std(total_losses):.6f}")
        print(f"  Point Map Loss: {np.mean(point_losses):.6f} ± {np.std(point_losses):.6f}")
        print(f"  Camera Pose Loss: {np.mean(pose_losses):.6f} ± {np.std(pose_losses):.6f}")
        
        if all_pose_errors:
            print(f"\n🎯 POSE ERROR STATISTICS:")
            print(f"  Mean Position Error: {np.mean(mean_pos_errors):.6f} ± {np.std(mean_pos_errors):.6f}")
            print(f"  Mean Rotation Error: {np.mean(mean_rot_errors):.2f}° ± {np.std(mean_rot_errors):.2f}°")
            print(f"  Current Frames Pos Error: {np.mean(current_pos_errors):.6f}")
            print(f"  Future Frames Pos Error:  {np.mean(future_pos_errors):.6f}")
        
        # Save summary to JSON
        summary = {
            'num_samples': len(all_losses),
            'losses': {
                'total_loss': {'mean': float(np.mean(total_losses)), 'std': float(np.std(total_losses))},
                'point_map_loss': {'mean': float(np.mean(point_losses)), 'std': float(np.std(point_losses))},
                'camera_pose_loss': {'mean': float(np.mean(pose_losses)), 'std': float(np.std(pose_losses))}
            }
        }
        
        if all_pose_errors:
            summary['pose_errors'] = {
                'mean_position_error': {'mean': float(np.mean(mean_pos_errors)), 'std': float(np.std(mean_pos_errors))},
                'mean_rotation_error': {'mean': float(np.mean(mean_rot_errors)), 'std': float(np.std(mean_rot_errors))},
                'current_position_error': float(np.mean(current_pos_errors)),
                'future_position_error': float(np.mean(future_pos_errors))
            }
        
        with open(os.path.join(save_dir, 'evaluation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Results saved to: {save_dir}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Pi3 Model Inference and Visualization")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Root directory containing test image folders")
    parser.add_argument("--m", type=int, default=3, help="Number of input frames")
    parser.add_argument("--n", type=int, default=3, help="Number of target frames")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--save_dir", type=str, default="./inference_results", 
                       help="Directory to save visualization results")
    parser.add_argument("--single_sample", action='store_true', 
                       help="Run inference on a single sample for quick testing")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = Pi3InferenceVisualizer(
        checkpoint_path=args.checkpoint,
        m=args.m,
        n=args.n,
        device=args.device
    )
    
    # Create test dataset
    image_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
                  if os.path.isdir(os.path.join(args.data_dir, d))]
    
    print(f"📁 Found {len(image_dirs)} test directories")
    
    dataset = SequenceLearningDataset(
        image_dirs=image_dirs,
        m=args.m,
        n=args.n,
        transform=get_default_transforms()
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    if args.single_sample:
        # Quick test on single sample
        print("🚀 Running single sample inference...")
        batch = next(iter(dataloader))
        
        X = batch[0]  # Current frames
        y = batch[1]  # Future frames
        video_tensor = torch.cat([X, y], dim=1)  # (1, T, C, H, W)
        video_tensor = preprocess_image(video_tensor.squeeze(0))[None]
        
        # Run inference
        results = visualizer.run_inference(video_tensor)
        
        # Compute losses
        losses = visualizer.compute_losses(results)
        print(f"📊 Losses: {losses}")
        
        # Visualize
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Point clouds for different frames
        for frame_idx in range(args.m + args.n):
            if frame_idx < args.m + args.n:
                visualizer.visualize_point_clouds(results, args.save_dir, frame_idx)
                # Also visualize local points if available
                visualizer.visualize_local_points(results, args.save_dir, frame_idx)
        
        # Camera poses
        pose_errors = visualizer.visualize_camera_poses(results, args.save_dir)
        print(f"🎯 Pose errors: {pose_errors}")
        
    else:
        # Full evaluation
        visualizer.run_evaluation(dataloader, args.num_samples, args.save_dir)


if __name__ == "__main__":
    main()