#!/usr/bin/env python3
"""
Standalone test script for MapAnything model creation.

This script tests the MapAnything model creation process without running
the full training pipeline. Useful for debugging model initialization.

Usage:
    python test_mapanything.py
    python test_mapanything.py --config config/mapanything_example.yaml
"""

import argparse
import os
import sys
import torch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.defaults import get_cfg_defaults, update_config
from utils.model_factory import create_model, validate_model_config, get_model_info


def test_mapanything_creation():
    """Test MapAnything model creation with various configurations."""
    
    print("=" * 60)
    print("🧪 MapAnything Model Creation Test")
    print("=" * 60)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test MapAnything model creation")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--opts', nargs='+', help='Config overrides')
    args = parser.parse_args()
    
    try:
        # Load configuration
        print("\n📋 Loading configuration...")
        cfg = get_cfg_defaults()
        cfg = update_config(cfg, args)
        
        # Override to use MapAnything if not already set
        if cfg.MODEL.ARCHITECTURE.lower() != "mapanything":
            print(f"⚠️  Config has ARCHITECTURE='{cfg.MODEL.ARCHITECTURE}', overriding to 'MapAnything'")
            cfg.defrost()
            cfg.MODEL.ARCHITECTURE = "MapAnything"
            cfg.freeze()
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Architecture: {cfg.MODEL.ARCHITECTURE}")
        
        # Validate configuration
        print("\n🔍 Validating model configuration...")
        validate_model_config(cfg)
        
        # Print model info
        print("\n📊 Model Configuration:")
        model_info = get_model_info(cfg)
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        # Print MapAnything specific config
        print("\n🗺️  MapAnything Specific Configuration:")
        ma_config = cfg.MODEL.MAPANYTHING
        config_items = [
            ("HF Model", ma_config.HF_MODEL_NAME),
            ("Config Path", ma_config.CONFIG_PATH),
            ("Checkpoint", ma_config.CHECKPOINT_NAME),
            ("Machine", ma_config.MACHINE),
            ("Task", ma_config.TASK),
            ("Resolution", ma_config.RESOLUTION),
            ("Patch Size", ma_config.PATCH_SIZE),
            ("AMP", f"{ma_config.TRAINED_WITH_AMP} ({ma_config.AMP_DTYPE})"),
            ("Data Norm", ma_config.DATA_NORM_TYPE),
            ("Use Torch Hub", ma_config.USE_TORCH_HUB),
        ]
        
        for name, value in config_items:
            print(f"   {name}: {value}")
        
        # Check device availability
        print(f"\n🖥️  Device Information:")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name()}")
            print(f"   CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test model creation
        import pdb; pdb.set_trace()

        print(f"\n🏗️  Creating MapAnything model...")
        model = create_model(cfg)
        
        print(f"\n✅ MapAnything model created successfully!")
        print(f"   Model type: {type(model).__name__}")
        
        # Try to get model info if available
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Non-trainable parameters: {total_params - trainable_params:,}")
        except Exception as e:
            print(f"   Parameter count unavailable: {e}")
        
        # Test model device
        try:
            device = next(model.parameters()).device
            print(f"   Model device: {device}")
        except Exception as e:
            print(f"   Device info unavailable: {e}")
        
        print(f"\n🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        print(f"\n📝 This might be expected if MapAnything dependencies aren't installed")
        print(f"   or if the model files aren't available yet.")
        
        # Print helpful debug info
        print(f"\n🔧 Debug Information:")
        print(f"   Python version: {sys.version}")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   Working directory: {os.getcwd()}")
        
        import traceback
        print(f"\n📋 Full traceback:")
        traceback.print_exc()
        
        return False


def test_config_variations():
    """Test MapAnything with different configuration variations."""
    
    print(f"\n" + "=" * 60)
    print("🔄 Testing Configuration Variations")
    print("=" * 60)
    
    variations = [
        {
            "name": "Default Configuration",
            "overrides": {}
        },
        {
            "name": "High Resolution", 
            "overrides": {
                "MODEL.MAPANYTHING.RESOLUTION": 1024,
                "MODEL.MAPANYTHING.PATCH_SIZE": 16
            }
        },
        {
            "name": "FP16 Mode",
            "overrides": {
                "MODEL.MAPANYTHING.AMP_DTYPE": "fp16"
            }
        },
        {
            "name": "Custom Task",
            "overrides": {
                "MODEL.MAPANYTHING.TASK": "custom_task",
                "MODEL.MAPANYTHING.MACHINE": "local"
            }
        }
    ]
    
    for i, variation in enumerate(variations, 1):
        print(f"\n📝 Variation {i}: {variation['name']}")
        print("-" * 40)
        
        try:
            cfg = get_cfg_defaults()
            cfg.defrost()
            cfg.MODEL.ARCHITECTURE = "MapAnything"
            
            # Apply overrides
            for key, value in variation['overrides'].items():
                keys = key.split('.')
                obj = cfg
                for k in keys[:-1]:
                    obj = getattr(obj, k)
                setattr(obj, keys[-1], value)
                print(f"   Override: {key} = {value}")
            
            cfg.freeze()
            
            # Validate
            validate_model_config(cfg)
            print(f"   ✅ Configuration valid")
            
        except Exception as e:
            print(f"   ❌ Configuration invalid: {e}")


if __name__ == "__main__":
    print("🚀 Starting MapAnything Model Test")
    
    # Test basic model creation
    success = test_mapanything_creation()
    
    if success:
        # Test configuration variations
        test_config_variations()
    
    print(f"\n" + "=" * 60)
    print("🏁 Test Complete")
    print("=" * 60)