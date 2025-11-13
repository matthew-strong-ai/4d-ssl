"""
Standalone script to load MotionNet ResNet152 Stage 2 checkpoint.
This script demonstrates how to load a pre-trained MotionNet model
from a Stage 2 checkpoint file.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import os

# Import the MotionNet model
from ppgeo_motionnet import MotionNet


def load_motionnet_stage2(checkpoint_path: str, resnet_layers: int = 152, device: str = 'cuda'):
    """
    Load a MotionNet Stage 2 checkpoint.
    
    Args:
        checkpoint_path: Path to the Stage 2 checkpoint file
        resnet_layers: Number of ResNet layers (18, 34, 50, 101, 152)
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded MotionNet model
        checkpoint: Full checkpoint dictionary with metadata
    """
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    # Create the model
    print(f"Creating MotionNet with ResNet{resnet_layers} encoder...")
    model = MotionNet(resnet_layers=resnet_layers)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Standard format from train_ppgeo.py
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"   Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    elif 'state_dict' in checkpoint:
        # Alternative format
        model.load_state_dict(checkpoint['state_dict'])
        print("✅ Loaded model state dict")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print("✅ Loaded model weights")
    
    # Move model to device
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, checkpoint


def test_inference(model: MotionNet, device: str = 'cuda'):
    """
    Test inference with the loaded model.
    
    Args:
        model: Loaded MotionNet model
        device: Device to run inference on
    """
    print("\nTesting inference...")
    
    # Create dummy input in PPGeo format
    batch_size = 1
    height, width = 192, 640  # PPGeo default resolution
    
    inputs = {
        ("color_aug", -1, 0): torch.randn(batch_size, 3, height, width).to(device),  # Previous frame
        ("color_aug", 0, 0): torch.randn(batch_size, 3, height, width).to(device),   # Current frame
    }
    
    # Run inference
    with torch.no_grad():
        axisangle1, translation1, axisangle2, translation2 = model(inputs)
    
    print(f"✅ Inference successful!")
    print(f"   Axisangle 1 shape: {axisangle1.shape}")
    print(f"   Translation 1 shape: {translation1.shape}")
    print(f"   Axisangle 2 shape: {axisangle2.shape}")
    print(f"   Translation 2 shape: {translation2.shape}")
    
    # Print sample outputs
    print(f"\nSample outputs:")
    print(f"   Axisangle 1: {axisangle1[0, 0, 0].cpu().numpy()}")
    print(f"   Translation 1: {translation1[0, 0, 0].cpu().numpy()}")


def main():
    """
    Main function to demonstrate loading and using MotionNet Stage 2.
    """
    # Configuration
    checkpoint_path = "best_ppgeo_stage2_motionnet.pt"  # Change this to your checkpoint path
    resnet_layers = 152  # Can be 18, 34, 50, 101, or 152
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Load the model
        model, checkpoint = load_motionnet_stage2(
            checkpoint_path=checkpoint_path,
            resnet_layers=resnet_layers,
            device=device
        )
        
        # Print model information
        print(f"\nModel Information:")
        print(f"   Architecture: MotionNet with ResNet{resnet_layers}")
        print(f"   Device: {device}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Test inference
        test_inference(model, device)
        
        # Example: Extract encoder features
        print("\nExtracting encoder features...")
        with torch.no_grad():
            test_image = torch.randn(1, 3, 192, 640).to(device)
            features = model.visual_encoder(test_image, normalize=True)
            print(f"   Number of feature levels: {len(features)}")
            for i, feat in enumerate(features):
                print(f"   Level {i}: {feat.shape}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    print("\n✨ Successfully loaded MotionNet Stage 2 checkpoint!")
    print("You can now use the model for motion estimation.")


if __name__ == "__main__":
    main()