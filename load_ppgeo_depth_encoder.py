"""
Load PPGeo trained model and extract the depth encoder for use.
"""

import torch
import os
from ppgeo_model import PPGeoModel, PPGeoDepthAnythingEncoder

def load_ppgeo_depth_encoder(checkpoint_path, device='cuda'):
    """
    Load a trained PPGeo model and return the depth encoder.
    
    Args:
        checkpoint_path: Path to the saved PPGeo checkpoint (.pt file)
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        PPGeoDepthAnythingEncoder: The loaded depth encoder
    """
    
    # Load checkpoint
    print(f"📦 Loading PPGeo checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create PPGeo model with same config as training
    model = PPGeoModel(
        encoder_name="dinov3",
        img_size=(160, 320),  # Default training size
        min_depth=0.1,
        max_depth=100.0,
        scales=[0, 1, 2, 3]  # Stage 1 scales
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"🔢 Model loss: {checkpoint.get('loss', 'unknown')}")
    
    # Extract and return the depth encoder
    depth_encoder = model.encoder
    
    print(f"🧠 Extracted depth encoder: {type(depth_encoder).__name__}")
    print(f"📏 Embedding dimension: {depth_encoder.embed_dim}")
    
    return depth_encoder

def test_depth_encoder(encoder, device='cuda'):
    """
    Test the depth encoder with a dummy input.
    
    Args:
        encoder: PPGeoDepthAnythingEncoder instance
        device: Device for testing
    """
    print("\n🧪 Testing depth encoder...")
    
    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        features = encoder(dummy_input)
    
    print(f"✅ Encoder output: {len(features)} feature levels")
    for i, (patch_feats, cls_token) in enumerate(features):
        print(f"  Level {i}: patch_features={patch_feats.shape}, cls_token={cls_token.shape}")
    
    return features

if __name__ == "__main__":
    # Example usage

    import ipdb; ipdb.set_trace()
    checkpoint_path = "checkpoints/ppgeo_stage1_step_1800.pt"  # Update this path
    
    if os.path.exists(checkpoint_path):
        # Load the depth encoder
        depth_encoder = load_ppgeo_depth_encoder(checkpoint_path)
        
        # Test it
        test_depth_encoder(depth_encoder)
        
        print(f"\n🎯 Depth encoder ready for use!")
        print(f"   Use: features = depth_encoder(rgb_images)")
        print(f"   Input: RGB images [B, 3, H, W]")
        print(f"   Output: List of (patch_features, cls_token) tuples")
        
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for f in os.listdir("."):
            if f.endswith('.pt'):
                print(f"  {f}")