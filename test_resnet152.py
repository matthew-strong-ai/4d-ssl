"""
Quick test script for ResNet-152 PPGeo training.
"""

import torch
from ppgeo_model import PPGeoModel

def test_resnet152():
    """Test ResNet-152 configuration."""
    print("🧪 Testing PPGeo ResNet-152 Configuration")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Create ResNet-152 model
    print("\n1. Creating ResNet-152 model...")
    model = PPGeoModel(
        encoder_name="resnet",
        img_size=(160, 320),
        min_depth=0.1,
        max_depth=100.0,
        scales=[0, 1, 2, 3],
        resnet_layers=152
    ).to(device).eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    depth_params = sum(p.numel() for p in model.encoder.parameters())
    pose_params = sum(p.numel() for p in model.pose_encoder.parameters())
    
    print(f"✅ ResNet-152 model created successfully!")
    print(f"📊 Total parameters: {total_params/1e6:.1f}M")
    print(f"   - Depth encoder: {depth_params/1e6:.1f}M")
    print(f"   - Pose encoder: {pose_params/1e6:.1f}M")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 2
    inputs = {}
    
    for frame_id in [-1, 0, 1]:
        for scale in [0]:
            inputs[("color_aug", frame_id, scale)] = torch.randn(batch_size, 3, 160, 320).to(device)
    
    with torch.no_grad():
        outputs, updated_inputs = model(inputs)
    
    print(f"✅ Forward pass successful!")
    
    # Check outputs
    print(f"\n3. Output analysis...")
    for key, value in outputs.items():
        if key[0] == "disp":
            print(f"   Depth {key}: {value.shape}")
        elif key[0] in ["axisangle", "translation"]:
            print(f"   Pose {key}: {value.shape}")
    
    # Memory usage
    if device.type == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"📈 Peak GPU memory: {memory_used:.2f} GB")
    
    print(f"\n🎉 ResNet-152 test completed successfully!")
    print(f"\n💡 To train with ResNet-152:")
    print(f"   python train_ppgeo.py --config config_ppgeo_resnet152.yaml")
    
    return model

if __name__ == "__main__":
    test_resnet152()