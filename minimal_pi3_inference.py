#!/usr/bin/env python3
"""
Minimal standalone Pi3 inference script.
Downloads pretrained weights and runs inference on sample images.
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add Pi3 to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pi3"))

from pi3.models.pi3 import Pi3


def load_pretrained_pi3(device='cuda'):
    """Load pretrained Pi3 model from HuggingFace Hub."""
    print("🔄 Loading pretrained Pi3 model...")
    
    # Load model directly from HuggingFace Hub
    model = Pi3.from_pretrained("yyfz233/Pi3")
    model = model.to(device)
    model.eval()
    
    print("✅ Pi3 model loaded successfully!")
    return model


def preprocess_images(image_paths, target_width=518, target_height=294):
    """
    Simple preprocessing for Pi3 input.
    
    Args:
        image_paths: List of image file paths
        target_size: Target size for resizing
        
    Returns:
        torch.Tensor: Preprocessed images [T, C, H, W]
    """
    images = []
    
    for path in image_paths:
        # Load and convert to RGB
        img = Image.open(path).convert('RGB')
        
        # Resize to target size
        img = img.resize((target_width, target_height), Image.LANCZOS)
        
        # Convert to numpy and normalize to [0, 1]
        img_np = np.array(img, dtype=np.float32) / 255.0
        
        # Convert to tensor and add to list
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [C, H, W]
        images.append(img_tensor)
    
    # Stack into video tensor [T, C, H, W]
    video_tensor = torch.stack(images, dim=0)
    
    return video_tensor


def run_pi3_inference(model, video_tensor):
    """
    Run Pi3 inference on preprocessed video tensor.
    
    Args:
        model: Pi3 model
        video_tensor: Input tensor [T, C, H, W]
        
    Returns:
        dict: Model predictions
    """
    device = next(model.parameters()).device
    
    # Add batch dimension [1, T, C, H, W]
    video_batch = video_tensor.unsqueeze(0).to(device)
    
    print(f"🔮 Running Pi3 inference on {video_tensor.shape[0]} frames...")
    
    with torch.no_grad():
        predictions = model(video_batch)
    
    # Remove batch dimension from predictions
    for key in predictions:
        if isinstance(predictions[key], torch.Tensor) and predictions[key].shape[0] == 1:
            predictions[key] = predictions[key].squeeze(0)
    
    return predictions


def visualize_results(predictions, original_images, save_path="pi3_results"):
    """Visualize Pi3 predictions."""
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
        print(f"✅ Depth visualization saved to {save_path}/depth_predictions.png")
    
    # Visualize confidence predictions
    if 'conf' in predictions:
        print("📊 Visualizing confidence predictions...")
        conf = predictions['conf'][..., 0].cpu().numpy()  # [T, H, W]
        
        fig, axes = plt.subplots(1, n_frames, figsize=(4*n_frames, 4))
        if n_frames == 1:
            axes = [axes]
        
        for t in range(n_frames):
            im = axes[t].imshow(conf[t], cmap='hot')
            axes[t].set_title(f'Confidence {t}')
            axes[t].axis('off')
            plt.colorbar(im, ax=axes[t], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confidence_predictions.png'), dpi=150)
        plt.close()
        print(f"✅ Confidence visualization saved to {save_path}/confidence_predictions.png")


def create_sample_images(save_dir="sample_images"):
    """Create synthetic sample images if no real images are available."""
    os.makedirs(save_dir, exist_ok=True)
    image_paths = []
    
    print("🎨 Creating synthetic sample images...")
    
    for i in range(3):  # Create 3 sample frames
        # Create a simple gradient image with some shapes
        img = np.zeros((518, 518, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(518):
            for x in range(518):
                img[y, x, 0] = min(255, (x + i * 50) % 256)  # Red gradient
                img[y, x, 1] = min(255, (y + i * 30) % 256)  # Green gradient
                img[y, x, 2] = min(255, (x + y + i * 20) % 256)  # Blue gradient
        
        # Add some shapes to make it more interesting
        center_x, center_y = 259 + i * 20, 259 + i * 10
        for y in range(max(0, center_y - 50), min(518, center_y + 50)):
            for x in range(max(0, center_x - 50), min(518, center_x + 50)):
                if (x - center_x) ** 2 + (y - center_y) ** 2 < 2500:  # Circle
                    img[y, x] = [255, 255, 255]  # White circle
        
        # Save image
        img_pil = Image.fromarray(img)
        img_path = os.path.join(save_dir, f"frame_{i}.png")
        img_pil.save(img_path)
        image_paths.append(img_path)
    
    print(f"✅ Created {len(image_paths)} sample images in {save_dir}/")
    return image_paths


def main():
    """Main inference function."""
    print("🚀 Pi3 Minimal Inference")
    print("=" * 50)
    
    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Load pretrained model
    model = load_pretrained_pi3(device)
    
    # Get sample images
    sample_image_paths = []
    
    # Try to find existing images
    if os.path.exists("sample_images"):
        sample_image_paths = [
            os.path.join("sample_images", f"frame_{i}.png") 
            for i in range(5)
            if os.path.exists(os.path.join("sample_images", f"frame_{i}.png"))
        ]
    
    # If no images found, create synthetic ones
    if not sample_image_paths:
        sample_image_paths = create_sample_images()
    
    print(f"📂 Using {len(sample_image_paths)} input images")
    
    # Load original images for visualization
    original_images = [Image.open(path) for path in sample_image_paths]
    
    # Preprocess images
    print("🔄 Preprocessing images...")
    video_tensor = preprocess_images(sample_image_paths)
    print(f"   Input shape: {video_tensor.shape}")
    
    # Run inference
    print("🔮 Running Pi3 inference...")
    predictions = run_pi3_inference(model, video_tensor)

    # dino features
    dino_features = predictions['dino_features']

    print(f"   DINO features shape: {dino_features.shape}")


    pi3_features = predictions['pi3_features']
    print(f"   Pi3 features shape: {pi3_features.shape}")
    
    # Print prediction info
    print("\n📋 Prediction Summary:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"   - {key}: shape {value.shape}, dtype {value.dtype}")
    
    # Visualize results
    print("\n📊 Creating visualizations...")
    visualize_results(predictions, original_images)
    
    print("\n✅ Pi3 inference completed successfully!")
    print(f"   Results saved in: pi3_results/")
    
    # Print some statistics
    if 'local_points' in predictions:
        depth = predictions['local_points'][..., 2]
        print(f"\n📏 Depth Statistics:")
        print(f"   - Min depth: {depth.min():.3f}m")
        print(f"   - Max depth: {depth.max():.3f}m")
        print(f"   - Mean depth: {depth.mean():.3f}m")
    
    if 'conf' in predictions:
        conf = predictions['conf']
        print(f"\n📊 Confidence Statistics:")
        print(f"   - Min confidence: {conf.min():.3f}")
        print(f"   - Max confidence: {conf.max():.3f}")
        print(f"   - Mean confidence: {conf.mean():.3f}")


if __name__ == "__main__":
    main()