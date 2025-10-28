#!/usr/bin/env python3
"""
Sample inference script for loading trained model checkpoints and running inference.
Demonstrates loading from both local files and S3.
"""

import os
import sys
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Add Pi3 to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pi3"))

# Import configuration and model utilities
from config.defaults import get_cfg_defaults, update_config
from utils.model_factory import create_model
from s3_utils import download_from_s3_uri
from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image

# Add YACS CfgNode to safe globals for PyTorch 2.6+
import yacs.config
torch.serialization.add_safe_globals([yacs.config.CfgNode])


def load_checkpoint(checkpoint_path, cfg, device='cuda'):
    """
    Load a model checkpoint from either local file or S3.
    
    Args:
        checkpoint_path: Path to checkpoint (local file or s3:// URI)
        cfg: Configuration object
        device: Device to load model on
    
    Returns:
        model: Loaded model ready for inference
        checkpoint: Full checkpoint dictionary
    """
    print(f"🔄 Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    if checkpoint_path.startswith('s3://'):
        print("📥 Downloading from S3...")
        # Download to local temp file first
        local_temp_path = os.path.join("checkpoints", "temp_downloaded_model.pt")
        os.makedirs("checkpoints", exist_ok=True)
        
        success = download_from_s3_uri(checkpoint_path, local_temp_path, overwrite=True)
        if not success:
            raise RuntimeError(f"Failed to download checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(local_temp_path, map_location=device)
    else:
        print("📂 Loading from local file...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get DINOv3 encoder path (required for model creation)
    dinov3_local_path = "dinov3/dinov3_vitl14_448.pth"
    if not os.path.exists(dinov3_local_path):
        print("⚠️  DINOv3 checkpoint not found locally, model will download it")
        dinov3_local_path = None
    
    # Create model
    print(f"🏗️  Creating {cfg.MODEL.ARCHITECTURE} model...")
    model = create_model(cfg, dinov3_local_path)
    
    # Load state dict
    print("📦 Loading model weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    if 'global_step' in checkpoint:
        print(f"   Global step: {checkpoint['global_step']}")
    if 'best_loss' in checkpoint:
        print(f"   Best loss: {checkpoint['best_loss']:.6f}")
    
    return model, checkpoint


def load_sample_images(image_paths, target_size=518):
    """
    Load and preprocess sample images for inference.
    
    Args:
        image_paths: List of paths to image files
        target_size: Target size for preprocessing
    
    Returns:
        tensor: Preprocessed image tensor [T, C, H, W]
        original_images: List of original PIL images
    """
    images = []
    original_images = []
    
    for path in image_paths:
        # Load image
        img = Image.open(path).convert('RGB')
        original_images.append(img)
        
        # Convert to numpy array
        img_np = np.array(img)
        images.append(img_np)
    
    # Stack images into video tensor [T, H, W, C]
    video_np = np.stack(images, axis=0)

    # convert to tensor
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()  # [T, C, H, W]
    
    # Preprocess for model (converts to tensor and normalizes)
    video_tensor = preprocess_image(video_tensor, target_size=target_size, patch_size=14)
    
    return video_tensor, original_images


def run_inference(model, video_tensor, cfg):
    """
    Run inference on preprocessed video tensor.
    
    Args:
        model: Loaded model
        video_tensor: Preprocessed video tensor [T, C, H, W]
        cfg: Configuration object
    
    Returns:
        predictions: Model predictions dictionary
    """
    device = next(model.parameters()).device
    
    # Add batch dimension and move to device
    video_batch = video_tensor.unsqueeze(0).to(device)  # [1, T, C, H, W]
    
    print(f"🔮 Running inference on {video_tensor.shape[0]} frames...")
    
    with torch.no_grad():
        predictions = model(video_batch)

    # Remove batch dimension from predictions
    for key in predictions:
        if isinstance(predictions[key], torch.Tensor) and predictions[key].shape[0] == 1:
            predictions[key] = predictions[key].squeeze(0)
    
    return predictions


def visualize_predictions(predictions, original_images, save_path="inference_results"):
    """
    Visualize model predictions including depth, motion, segmentation, etc.
    
    Args:
        predictions: Model predictions dictionary
        original_images: List of original PIL images
        save_path: Directory to save visualizations
    """
    os.makedirs(save_path, exist_ok=True)
    n_frames = len(original_images)
    
    # Visualize depth predictions
    if 'local_points' in predictions:
        print("📊 Visualizing depth predictions...")
        depth = predictions['local_points'][..., 2].cpu().numpy()  # [T, H, W]
        
        fig, axes = plt.subplots(2, n_frames, figsize=(4*n_frames, 8))
        if n_frames == 1:
            axes = axes.reshape(2, 1)
        
        for t in range(n_frames):
            # Original image
            axes[0, t].imshow(original_images[t])
            axes[0, t].set_title(f'Frame {t}')
            axes[0, t].axis('off')
            
            # Depth map
            im = axes[1, t].imshow(depth[t], cmap='viridis')
            axes[1, t].set_title(f'Depth {t}')
            axes[1, t].axis('off')
            plt.colorbar(im, ax=axes[1, t], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'depth_predictions.png'), dpi=150)
        plt.close()
    
    # Visualize segmentation predictions
    if 'segmentation' in predictions:
        print("📊 Visualizing segmentation predictions...")
        seg = predictions['segmentation'].cpu().numpy()  # [T, H, W, num_classes]
        seg_argmax = np.argmax(seg, axis=-1)  # [T, H, W]
        
        # Define colors for each class
        colors = np.array([
            [128, 64, 128],   # road
            [244, 35, 232],   # sidewalk  
            [70, 70, 70],     # building
            [102, 102, 156],  # wall
            [190, 153, 153],  # fence
            [153, 153, 153],  # pole
            [250, 170, 30],   # traffic light
            [220, 220, 0],    # traffic sign
            [107, 142, 35],   # vegetation
        ])
        
        fig, axes = plt.subplots(2, n_frames, figsize=(4*n_frames, 8))
        if n_frames == 1:
            axes = axes.reshape(2, 1)
            
        for t in range(n_frames):
            # Original image
            axes[0, t].imshow(original_images[t])
            axes[0, t].set_title(f'Frame {t}')
            axes[0, t].axis('off')
            
            # Segmentation map
            seg_colored = colors[seg_argmax[t]] if seg_argmax[t].max() < len(colors) else seg_argmax[t]
            axes[1, t].imshow(seg_colored)
            axes[1, t].set_title(f'Segmentation {t}')
            axes[1, t].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'segmentation_predictions.png'), dpi=150)
        plt.close()
    
    # Visualize motion predictions
    if 'motion' in predictions:
        print("📊 Visualizing motion predictions...")
        motion = predictions['motion'].cpu().numpy()  # [T, H, W, 3]
        
        fig, axes = plt.subplots(2, n_frames, figsize=(4*n_frames, 8))
        if n_frames == 1:
            axes = axes.reshape(2, 1)
            
        for t in range(n_frames):
            # Original image
            axes[0, t].imshow(original_images[t])
            axes[0, t].set_title(f'Frame {t}')
            axes[0, t].axis('off')
            
            # Motion magnitude
            motion_mag = np.linalg.norm(motion[t, ..., :2], axis=-1)
            im = axes[1, t].imshow(motion_mag, cmap='hot')
            axes[1, t].set_title(f'Motion Magnitude {t}')
            axes[1, t].axis('off')
            plt.colorbar(im, ax=axes[1, t], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'motion_predictions.png'), dpi=150)
        plt.close()
    
    print(f"✅ Visualizations saved to {save_path}/")


def main():
    """Main inference demo function."""

    
    # Load configuration
    cfg = get_cfg_defaults()
    cfg = update_config(cfg)
    
    # Example checkpoint paths (modify these for your checkpoints)
    # Local checkpoint
    local_checkpoint = "checkpoints/best_model.pt"
    
    # S3 checkpoint  
    s3_checkpoint = "s3://research-datasets/autonomy_checkpoints/matt-segmentation-prediction_best_model.pt"
    
    # Choose which checkpoint to use
    checkpoint_path = local_checkpoint  # Change to s3_checkpoint to test S3 loading
    
    # Check if local checkpoint exists, otherwise try S3
    if checkpoint_path == local_checkpoint and not os.path.exists(checkpoint_path):
        print(f"⚠️  Local checkpoint not found at {checkpoint_path}, trying S3...")
        checkpoint_path = s3_checkpoint
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_checkpoint(checkpoint_path, cfg, device)
    
    # Prepare sample images
    # You can modify this to use your own images
    sample_image_paths = []
    
    # Try to find sample images
    if os.path.exists("sample_images"):
        sample_image_paths = [
            os.path.join("sample_images", f"frame_{i}.png") 
            for i in range(cfg.MODEL.M + cfg.MODEL.N)
            if os.path.exists(os.path.join("sample_images", f"frame_{i}.png"))
        ]
    
    if not sample_image_paths:
        print("⚠️  No sample images found. Creating synthetic test images...")
        # Create synthetic test images
        os.makedirs("sample_images", exist_ok=True)
        for i in range(cfg.MODEL.M + cfg.MODEL.N):
            # Create a simple gradient image
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[:, :, 0] = np.linspace(0, 255, 640).astype(np.uint8)  # Red gradient
            img[:, :, 1] = np.linspace(0, 255, 480).reshape(-1, 1).astype(np.uint8)  # Green gradient
            img[:, :, 2] = 128  # Constant blue
            
            # Add frame number
            from PIL import ImageDraw
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            draw.text((10, 10), f"Frame {i}", fill=(255, 255, 255))
            
            path = os.path.join("sample_images", f"frame_{i}.png")
            pil_img.save(path)
            sample_image_paths.append(path)
            
        print(f"✅ Created {len(sample_image_paths)} synthetic test images")
    
    # Load and preprocess images
    video_tensor, original_images = load_sample_images(sample_image_paths)
    print(f"📷 Loaded {len(original_images)} frames")
    
    # Run inference
    predictions = run_inference(model, video_tensor, cfg)
    
    # Print prediction keys and shapes
    print("\n📋 Model predictions:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: {tuple(value.shape)} ({value.dtype})")
        else:
            print(f"  - {key}: {type(value)}")
    
    # Visualize results
    # visualize_predictions(predictions, original_images)
    
    # # Save raw predictions
    # print("\n💾 Saving raw predictions...")
    # predictions_cpu = {}
    # for key, value in predictions.items():
    #     if isinstance(value, torch.Tensor):
    #         predictions_cpu[key] = value.cpu().numpy()
    #     else:
    #         predictions_cpu[key] = value
    
    # np.savez_compressed(
    #     "inference_results/raw_predictions.npz",
    #     **predictions_cpu
    # )
    # print("✅ Saved raw predictions to inference_results/raw_predictions.npz")
    
    print("\n🎉 Inference complete!")


if __name__ == "__main__":
    main()