"""
Simple example to load and use MotionNet ResNet152 Stage 2 checkpoint.
Just update the checkpoint_path variable and run!
"""

import torch
from ppgeo_motionnet import MotionNet


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load MotionNet Stage 2 checkpoint"""
    
    # Create model with ResNet152 backbone
    model = MotionNet(resnet_layers=152)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded MotionNet ResNet152 from {checkpoint_path}")
    return model


# Example usage:
if __name__ == "__main__":
    # Update this path to your checkpoint
    checkpoint_path = "path/to/your/ppgeo_stage2_resnet152.pt"
    
    # Load model
    model = load_checkpoint(checkpoint_path)
    
    # Create sample input (2 consecutive frames)
    batch_size = 1
    height, width = 192, 640  # PPGeo resolution
    
    inputs = {
        ("color_aug", -1, 0): torch.randn(batch_size, 3, height, width).cuda(),  # Frame t-1
        ("color_aug", 0, 0): torch.randn(batch_size, 3, height, width).cuda(),   # Frame t
    }
    
    # Run inference
    with torch.no_grad():
        axisangle1, translation1, axisangle2, translation2 = model(inputs)
    
    print(f"Output shapes:")
    print(f"  Axis-angle 1: {axisangle1.shape}")
    print(f"  Translation 1: {translation1.shape}")
    print(f"  Axis-angle 2: {axisangle2.shape}") 
    print(f"  Translation 2: {translation2.shape}")