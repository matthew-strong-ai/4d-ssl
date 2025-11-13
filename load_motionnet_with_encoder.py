"""
Load PPGeo MotionNet Stage 2 with ResNet152 backbone and demonstrate encoder usage.
This script loads the checkpoint and shows how to extract features using the encoder.

The checkpoint is at: /home/matthew_strong/Desktop/autonomy-wild/checkpoints/ppgeo_stage2_step_1100.pt
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


def extract_encoder_features(model, image, normalize=True, device='cuda'):
    """
    Extract multi-scale features using the ResNet152 encoder.
    
    Args:
        model: Loaded MotionNet model
        image: Input image (numpy array or torch tensor)
        normalize: Whether to normalize the image
        device: Device to run on
    
    Returns:
        features: List of feature maps at different scales
    """
    
    # Convert numpy to torch if needed
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    
    # Ensure correct shape (B, C, H, W)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Move to device
    image = image.to(device)
    
    # Extract features using the encoder
    with torch.no_grad():
        features = model.visual_encoder(image, normalize=normalize)
    
    return features


def demonstrate_encoder(model, device='cuda'):
    """
    Demonstrate the encoder functionality and show feature map sizes.
    """
    
    print("\n" + "="*60)
    print("ResNet152 Encoder Demonstration")
    print("="*60)
    
    # Create a sample image
    height, width = 192, 640
    sample_image = torch.randn(1, 3, height, width).to(device)
    
    # Extract features
    features = extract_encoder_features(model, sample_image, normalize=True, device=device)
    
    print(f"\nEncoder outputs {len(features)} feature maps at different scales:")
    print(f"Input image shape: {sample_image.shape}")
    print("\nFeature map shapes:")
    
    for i, feat in enumerate(features):
        scale = 2 ** (i + 1) if i < 4 else 2 ** 4  # ResNet downsampling pattern
        print(f"  Level {i}: {feat.shape} (1/{scale} resolution)")
    
    # Show encoder channel configuration
    if hasattr(model.visual_encoder, 'num_ch_enc'):
        print(f"\nEncoder channel configuration: {model.visual_encoder.num_ch_enc}")
    
    # Access the underlying ResNet model
    print("\n" + "-"*40)
    print("ResNet152 Architecture Details:")
    print("-"*40)
    
    resnet = model.visual_encoder.encoder
    print(f"Conv1: {resnet.conv1}")
    print(f"Layer1: {len(resnet.layer1)} blocks, output channels: {resnet.layer1[-1].conv3.out_channels}")
    print(f"Layer2: {len(resnet.layer2)} blocks, output channels: {resnet.layer2[-1].conv3.out_channels}")
    print(f"Layer3: {len(resnet.layer3)} blocks, output channels: {resnet.layer3[-1].conv3.out_channels}")
    print(f"Layer4: {len(resnet.layer4)} blocks, output channels: {resnet.layer4[-1].conv3.out_channels}")
    
    return features


def use_encoder_for_custom_task(model, image, device='cuda'):
    """
    Example of using the encoder for a custom task (e.g., feature matching, depth estimation).
    """
    
    print("\n" + "="*60)
    print("Using Encoder for Custom Tasks")
    print("="*60)
    
    # Extract multi-scale features
    features = extract_encoder_features(model, image, normalize=True, device=device)
    
    # Example: Use the last feature map for a custom head
    last_features = features[-1]  # Highest level features
    print(f"\nLast feature map shape: {last_features.shape}")
    
    # Example: Global average pooling for image-level features
    global_features = torch.nn.functional.adaptive_avg_pool2d(last_features, (1, 1))
    global_features = global_features.squeeze(-1).squeeze(-1)
    print(f"Global features shape: {global_features.shape}")
    
    # Example: Get intermediate features for dense prediction
    mid_features = features[2]  # 1/8 resolution features
    print(f"Mid-level features shape: {mid_features.shape}")
    
    return features


def main():
    """Example usage showing both motion estimation and encoder features"""
    
    # Load model
    model = load_motionnet_resnet152()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Demonstrate encoder functionality
    features = demonstrate_encoder(model, device)
    
    # Create example image
    height, width = 192, 640
    example_image = torch.randn(3, height, width)
    
    # Show how to use encoder for custom tasks
    use_encoder_for_custom_task(model, example_image, device)
    
    # Original motion estimation functionality
    print("\n" + "="*60)
    print("Motion Estimation (Original Functionality)")
    print("="*60)
    
    frame1 = torch.randn(3, height, width)
    frame2 = torch.randn(3, height, width)
    
    inputs = {
        ("color_aug", -1, 0): frame1.unsqueeze(0).to(device),
        ("color_aug", 0, 0): frame2.unsqueeze(0).to(device),
    }
    
    with torch.no_grad():
        axisangle1, translation1, axisangle2, translation2 = model(inputs)
    
    print(f"\nMotion outputs:")
    print(f"  Rotation: {axisangle1[0, 0, 0].cpu().numpy()}")
    print(f"  Translation: {translation1[0, 0, 0].cpu().numpy()}")
    
    print("\n✨ The encoder can be accessed via: model.visual_encoder")
    print("   Use extract_encoder_features() to get multi-scale features")


if __name__ == "__main__":
    main()