"""
Load PPGeo MotionNet Stage 2 with ResNet152 backbone.
This script loads the checkpoint at: /home/matthew_strong/Desktop/autonomy-wild/checkpoints/ppgeo_stage2_step_1100.pt

Usage:
    python load_motionnet_resnet152.py

The model expects input in PPGeo format:
    inputs = {
        ("color_aug", -1, 0): previous_frame,  # Shape: (B, 3, 192, 640)
        ("color_aug", 0, 0): current_frame,    # Shape: (B, 3, 192, 640)
    }

Output:
    axisangle1, translation1: Motion for frame 1
    axisangle2, translation2: Motion for frame 2
    All outputs have shape: (B, 2, 1, 3)
"""

import torch
import torch.nn as nn
from ppgeo_motionnet import MotionNet
import numpy as np


def load_motionnet_resnet152(checkpoint_path=None, device='cuda'):
    """
    Load MotionNet with ResNet152 backbone from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (optional, uses default if None)
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded MotionNet model ready for inference
    """
    
    if checkpoint_path is None:
        checkpoint_path = "/home/matthew_strong/Desktop/autonomy-wild/checkpoints/ppgeo_stage2_step_1100.pt"
    
    print(f"Loading MotionNet ResNet152 from: {checkpoint_path}")
    
    # Create model with ResNet152 backbone
    model = MotionNet(resnet_layers=152)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"✅ Successfully loaded checkpoint")
    print(f"   Training step: {checkpoint.get('global_step', 'unknown')}")
    print(f"   Training loss: {checkpoint.get('loss', 'N/A'):.6f}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def preprocess_images(frame1, frame2, normalize=True):
    """
    Preprocess image pair for MotionNet input.
    
    Args:
        frame1: Previous frame (numpy array or torch tensor)
        frame2: Current frame (numpy array or torch tensor)
        normalize: Whether to normalize images (default: True)
    
    Returns:
        inputs: Dictionary in PPGeo format
    """
    
    # Convert numpy to torch if needed
    if isinstance(frame1, np.ndarray):
        frame1 = torch.from_numpy(frame1).float()
    if isinstance(frame2, np.ndarray):
        frame2 = torch.from_numpy(frame2).float()
    
    # Ensure correct shape (B, C, H, W)
    if frame1.dim() == 3:
        frame1 = frame1.unsqueeze(0)
    if frame2.dim() == 3:
        frame2 = frame2.unsqueeze(0)
    
    # Normalize if requested
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frame1 = (frame1 - mean) / std
        frame2 = (frame2 - mean) / std
    
    # Create PPGeo format input
    inputs = {
        ("color_aug", -1, 0): frame1,  # Previous frame
        ("color_aug", 0, 0): frame2,    # Current frame
    }
    
    return inputs


def estimate_motion(model, frame1, frame2, device='cuda'):
    """
    Estimate motion between two frames.
    
    Args:
        model: Loaded MotionNet model
        frame1: Previous frame (H, W, 3) numpy array or (3, H, W) tensor
        frame2: Current frame (H, W, 3) numpy array or (3, H, W) tensor
        device: Device to run inference on
    
    Returns:
        motion_dict: Dictionary containing rotation and translation for both frames
    """
    
    # Preprocess inputs
    inputs = preprocess_images(frame1, frame2)
    
    # Move to device
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    
    # Run inference
    with torch.no_grad():
        axisangle1, translation1, axisangle2, translation2 = model(inputs)
    
    # Convert to numpy
    motion_dict = {
        'frame1_rotation': axisangle1[0, 0, 0].cpu().numpy(),      # (3,)
        'frame1_translation': translation1[0, 0, 0].cpu().numpy(),  # (3,)
        'frame2_rotation': axisangle2[0, 0, 0].cpu().numpy(),      # (3,)
        'frame2_translation': translation2[0, 0, 0].cpu().numpy(),  # (3,)
    }
    
    return motion_dict


def main():
    """Example usage"""
    
    # Load model
    model = load_motionnet_resnet152()
    
    # Create example frames (normally you'd load real images)
    height, width = 192, 640
    frame1 = torch.randn(3, height, width)
    frame2 = torch.randn(3, height, width)
    
    # Estimate motion
    motion = estimate_motion(model, frame1, frame2)
    
    print("\nMotion estimation results:")
    print(f"Frame 1 rotation (axis-angle): {motion['frame1_rotation']}")
    print(f"Frame 1 translation: {motion['frame1_translation']}")
    print(f"Frame 2 rotation (axis-angle): {motion['frame2_rotation']}")
    print(f"Frame 2 translation: {motion['frame2_translation']}")
    
    print("\n✨ Ready for motion estimation!")
    print("Use estimate_motion(model, frame1, frame2) with your own images.")


if __name__ == "__main__":
    main()