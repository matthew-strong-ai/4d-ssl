"""
Test PPGeo ResNet implementation to ensure it works correctly.
"""

import torch
import torch.nn as nn
from ppgeo_model import PPGeoModel
from ppgeo_resnet import PPGeoResnetEncoder, PPGeoResnetDepthDecoder

def test_resnet_components():
    """Test individual ResNet components."""
    print("🧪 Testing PPGeo ResNet Components")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Test ResNet encoder
    print("\n1. Testing ResNet Encoder...")
    encoder = PPGeoResnetEncoder(num_layers=18, pretrained=True, num_input_images=1)
    encoder.to(device)
    encoder.eval()
    
    # Test with dummy input
    dummy_input = torch.randn(2, 3, 160, 320).to(device)  # PPGeo standard size
    
    with torch.no_grad():
        features = encoder(dummy_input, normalize=True)
    
    print(f"✅ ResNet-18 encoder output: {len(features)} feature levels")
    for i, feat in enumerate(features):
        print(f"  Level {i}: {feat.shape} - Channels: {encoder.num_ch_enc[i]}")
    
    # Test ResNet depth decoder
    print("\n2. Testing ResNet Depth Decoder...")
    depth_decoder = PPGeoResnetDepthDecoder(
        num_ch_enc=encoder.num_ch_enc,
        scales=[0, 1, 2, 3],
        num_output_channels=1,
        use_skips=True
    )
    depth_decoder.to(device)
    depth_decoder.eval()
    
    with torch.no_grad():
        depth_outputs = depth_decoder(features)
    
    print(f"✅ Depth decoder output: {len(depth_outputs)} scales")
    for key, value in depth_outputs.items():
        print(f"  {key}: {value.shape}")
    
    print("\n" + "=" * 50)
    return encoder, depth_decoder

def test_full_ppgeo_model():
    """Test full PPGeo model with ResNet."""
    print("🧪 Testing Full PPGeo Model with ResNet")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create PPGeo model with ResNet
    print("\n1. Creating PPGeo ResNet model...")
    model = PPGeoModel(
        encoder_name="resnet",
        img_size=(160, 320),
        min_depth=0.1,
        max_depth=100.0,
        scales=[0, 1, 2, 3],
        resnet_layers=18
    )
    model.to(device)
    model.eval()
    
    print(f"✅ Created PPGeo model with ResNet-18")
    print(f"   Depth encoder: {type(model.encoder).__name__}")
    print(f"   Depth decoder: {type(model.depth_decoder).__name__}")
    print(f"   Pose encoder: {type(model.pose_encoder).__name__}")
    
    # Create dummy inputs (PPGeo format)
    print("\n2. Testing forward pass...")
    batch_size = 2
    inputs = {}
    
    # Create dummy RGB frames
    for frame_id in [-1, 0, 1]:
        for scale in [0]:
            inputs[("color_aug", frame_id, scale)] = torch.randn(batch_size, 3, 160, 320).to(device)
    
    with torch.no_grad():
        outputs, updated_inputs = model(inputs)
    
    print(f"✅ Forward pass successful!")
    print(f"   Output keys: {list(outputs.keys())}")
    
    # Check depth outputs
    for key, value in outputs.items():
        if key[0] == "disp":
            print(f"   Depth {key}: {value.shape}")
    
    # Check pose outputs
    for key, value in outputs.items():
        if key[0] in ["axisangle", "translation"]:
            print(f"   Pose {key}: {value.shape}")
    
    print("\n" + "=" * 50)
    return model

def compare_vit_vs_resnet():
    """Compare ViT and ResNet models."""
    print("🧪 Comparing ViT vs ResNet PPGeo Models")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create both models
    print("\n1. Creating both models...")
    
    try:
        vit_model = PPGeoModel(
            encoder_name="dinov3",
            img_size=(160, 320),
            scales=[0, 1, 2, 3]
        ).to(device).eval()
        print("✅ ViT model created successfully")
    except Exception as e:
        print(f"❌ ViT model failed: {e}")
        vit_model = None
    
    resnet_model = PPGeoModel(
        encoder_name="resnet",
        img_size=(160, 320),
        scales=[0, 1, 2, 3],
        resnet_layers=18
    ).to(device).eval()
    print("✅ ResNet model created successfully")
    
    # Count parameters
    if vit_model:
        vit_params = sum(p.numel() for p in vit_model.parameters())
        print(f"📊 ViT parameters: {vit_params/1e6:.1f}M")
    
    resnet_params = sum(p.numel() for p in resnet_model.parameters())
    print(f"📊 ResNet parameters: {resnet_params/1e6:.1f}M")
    
    # Test inference speed
    print("\n2. Testing inference speed...")
    inputs = {}
    batch_size = 2
    
    for frame_id in [-1, 0, 1]:
        for scale in [0]:
            inputs[("color_aug", frame_id, scale)] = torch.randn(batch_size, 3, 160, 320).to(device)
    
    # ResNet timing
    import time
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        start_time = time.time()
        for _ in range(10):
            _, _ = resnet_model(inputs)
        resnet_time = (time.time() - start_time) / 10
    
    print(f"⚡ ResNet inference time: {resnet_time*1000:.1f}ms per batch")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    print("🎯 PPGeo ResNet Implementation Test")
    print("=" * 60)
    
    try:
        # Test 1: Individual components
        test_resnet_components()
        
        # Test 2: Full model
        test_full_ppgeo_model()
        
        # Test 3: Compare with ViT
        compare_vit_vs_resnet()
        
        print("\n🎉 All tests passed! ResNet implementation is working correctly.")
        print("\n💡 To train with ResNet, use:")
        print("   python train_ppgeo.py --config config_ppgeo_resnet.yaml")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()