"""
Load PPGeo trained model and extract the depth encoder for use.
"""

import torch
import os
from ppgeo_model import PPGeoModel, PPGeoDepthAnythingEncoder

def load_ppgeo_depth_encoder(checkpoint_path, encoder_name="dinov3", resnet_layers=18, device='cuda'):
    """
    Load a trained PPGeo model and return the depth encoder.
    
    Args:
        checkpoint_path: Path to the saved PPGeo checkpoint (.pt file)
        encoder_name: Type of encoder ("dinov3", "resnet", etc.)
        resnet_layers: Number of ResNet layers if using ResNet
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        The loaded depth encoder (ViT or ResNet)
    """
    
    # Load checkpoint
    print(f"📦 Loading PPGeo checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create PPGeo model with same config as training
    model = PPGeoModel(
        encoder_name=encoder_name,
        img_size=(160, 320),  # Default training size
        min_depth=0.1,
        max_depth=100.0,
        scales=[0, 1, 2, 3],  # Stage 1 scales
        resnet_layers=resnet_layers
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded {encoder_name} model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"🔢 Model loss: {checkpoint.get('loss', 'unknown')}")
    
    # Extract and return the depth encoder
    depth_encoder = model.encoder
    
    print(f"🧠 Extracted depth encoder: {type(depth_encoder).__name__}")
    
    if encoder_name == "resnet":
        print(f"🏗️ ResNet-{resnet_layers} with {depth_encoder.num_ch_enc} channels")
    else:
        print(f"📏 Embedding dimension: {depth_encoder.embed_dim}")
    
    return depth_encoder

def test_depth_encoder(encoder, encoder_type="vit", device='cuda'):
    """
    Test the depth encoder with a dummy input.
    
    Args:
        encoder: Depth encoder instance (ViT or ResNet)
        encoder_type: Type of encoder ("vit" or "resnet")
        device: Device for testing
    """
    print("\n🧪 Testing depth encoder...")
    
    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        if encoder_type == "resnet":
            # ResNet encoder
            features = encoder(dummy_input, normalize=True)
            print(f"✅ ResNet encoder output: {len(features)} feature levels")
            for i, feat in enumerate(features):
                print(f"  Level {i}: {feat.shape}")
        else:
            # ViT encoder
            features = encoder(dummy_input)
            print(f"✅ ViT encoder output: {len(features)} feature levels")
            for i, (patch_feats, cls_token) in enumerate(features):
                print(f"  Level {i}: patch_features={patch_feats.shape}, cls_token={cls_token.shape}")
    
    return features

if __name__ == "__main__":
    # Example usage for ViT encoder
    checkpoint_path = "checkpoints/ppgeo_stage1_step_1800.pt"  # ViT checkpoint
    
    if os.path.exists(checkpoint_path):
        # Load the ViT depth encoder
        depth_encoder = load_ppgeo_depth_encoder(checkpoint_path, encoder_name="dinov3")
        
        # Test it
        test_depth_encoder(depth_encoder, encoder_type="vit")
        
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