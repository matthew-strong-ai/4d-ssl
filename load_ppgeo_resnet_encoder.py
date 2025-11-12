"""
Load PPGeo ResNet encoder from checkpoint.
Extract the depth encoder for use in other applications.
"""

import torch
import torch.nn as nn
from ppgeo_model import PPGeoModel
from ppgeo_resnet import PPGeoResnetEncoder

def load_ppgeo_resnet_encoder(checkpoint_path: str, resnet_layers: int = 152, device: str = "cuda"):
    """
    Load PPGeo ResNet encoder from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to the PPGeo checkpoint (.pt file)
        resnet_layers: Number of ResNet layers (18, 34, 50, 101, 152)
        device: Device to load the model on
    
    Returns:
        encoder: The loaded ResNet encoder
        model_info: Dictionary with checkpoint metadata
    """
    
    print(f"🔧 Loading PPGeo ResNet-{resnet_layers} checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"📦 Checkpoint loaded - Epoch: {checkpoint.get('epoch', 'unknown')}, Loss: {checkpoint.get('loss', 'unknown'):.4f}")
    
    # Create PPGeo model with ResNet encoder
    model = PPGeoModel(
        encoder_name="resnet",
        img_size=(160, 320),
        min_depth=0.1,
        max_depth=100.0,
        scales=[0, 1, 2, 3],
        resnet_layers=resnet_layers
    )
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    print(f"✅ PPGeo ResNet-{resnet_layers} model loaded successfully")
    
    # Extract just the encoder
    encoder = model.encoder
    
    # Print encoder info
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"🔢 Encoder parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")
    print(f"📐 Encoder channels: {encoder.num_ch_enc}")
    
    # Test forward pass
    with torch.no_grad():
        test_input = torch.randn(1, 3, 160, 320).to(device)
        features = encoder(test_input, normalize=True)
        print(f"🧪 Test forward pass successful:")
        for i, feat in enumerate(features):
            print(f"   Scale {i}: {feat.shape}")
    
    model_info = {
        'epoch': checkpoint.get('epoch', None),
        'loss': checkpoint.get('loss', None),
        'resnet_layers': resnet_layers,
        'encoder_channels': encoder.num_ch_enc,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    
    return encoder, model_info


def load_standalone_resnet_encoder(checkpoint_path: str, resnet_layers: int = 152, device: str = "cuda"):
    """
    Load only the ResNet encoder part (useful for transfer learning).
    
    Args:
        checkpoint_path: Path to the PPGeo checkpoint
        resnet_layers: Number of ResNet layers
        device: Device to load on
        
    Returns:
        encoder: Standalone ResNet encoder
    """
    
    print(f"🔧 Creating standalone ResNet-{resnet_layers} encoder...")
    
    # Create standalone encoder
    encoder = PPGeoResnetEncoder(
        num_layers=resnet_layers, 
        pretrained=False,  # We'll load from checkpoint
        num_input_images=1
    )
    
    # Load full model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    full_model_state = checkpoint['model_state_dict']
    
    # Extract only encoder weights
    encoder_state = {}
    for key, value in full_model_state.items():
        if key.startswith('encoder.'):
            new_key = key.replace('encoder.', '')  # Remove 'encoder.' prefix
            encoder_state[new_key] = value
    
    # Load encoder weights
    missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state, strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys: {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")
    
    encoder.to(device)
    encoder.eval()
    
    print(f"✅ Standalone ResNet-{resnet_layers} encoder loaded")
    
    return encoder


def test_encoder_inference(encoder, device: str = "cuda"):
    """Test the encoder on sample data."""
    
    print("🧪 Testing encoder inference...")
    
    encoder.eval()
    with torch.no_grad():
        # Test with batch of images
        batch_size = 2
        test_images = torch.randn(batch_size, 3, 160, 320).to(device)
        
        # Forward pass
        features = encoder(test_images, normalize=True)
        
        print(f"✅ Inference test successful with batch size {batch_size}")
        print(f"📊 Feature scales:")
        for i, feat in enumerate(features):
            print(f"   Scale {i}: {feat.shape} | Range: [{feat.min():.3f}, {feat.max():.3f}]")
        
        return features


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Load PPGeo ResNet encoder from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to PPGeo checkpoint')
    parser.add_argument('--layers', type=int, default=152, choices=[18, 34, 50, 101, 152], 
                       help='Number of ResNet layers')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--standalone', action='store_true', help='Load as standalone encoder')
    
    args = parser.parse_args()
    
    if args.standalone:
        encoder = load_standalone_resnet_encoder(args.checkpoint, args.layers, args.device)
    else:
        encoder, info = load_ppgeo_resnet_encoder(args.checkpoint, args.layers, args.device)
        print(f"📋 Model info: {info}")
    
    # Test the encoder
    features = test_encoder_inference(encoder, args.device)
    
    print("🎉 Encoder loading and testing completed!")