#!/usr/bin/env python3
"""
Quick test of optimized GSAM2 - minimal example
"""

import numpy as np
import time

def quick_test():
    """Quick test with minimal setup."""
    print("🚀 Quick GSAM2 Optimization Test")
    print("=" * 40)
    
    try:
        # Import the optimized class
        from gsam2_optimized_complete import OptimizedGSAM2
        
        # Create small test frames (faster for testing)
        print("📸 Creating test frames...")
        frames = [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(3)]
        print(f"   Created {len(frames)} frames of size 240x320")
        
        # Initialize optimized GSAM2
        print("🔧 Initializing OptimizedGSAM2...")
        gsam2 = OptimizedGSAM2()
        print("   ✅ Initialization complete")
        
        # Test processing
        print("⚡ Testing optimized processing...")
        start_time = time.time()
        
        results = gsam2.process_frames_optimized(
            rgb_frames=frames,
            prompt="vehicle. person.",
            use_batching=True,
            batch_size=2,
            box_threshold=0.3,
            text_threshold=0.25
        )
        
        end_time = time.time()
        
        # Results
        print(f"✅ Processing completed!")
        print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
        print(f"📊 Frames processed: {results['num_frames']}")
        print(f"🎯 Objects detected: {results['num_objects']}")
        print(f"🏷️  Labels: {results['labels']}")
        
        if results['num_objects'] > 0:
            print(f"📦 Bounding boxes shape: {results['boxes'].shape}")
            print(f"📈 Detection scores: {results['scores']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   - SAM2 models")
        print("   - Grounding DINO")
        print("   - transformers, torch, cv2, PIL")
        return False
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("💡 This might be due to:")
        print("   - Missing model checkpoints")
        print("   - GPU memory issues")
        print("   - CUDA/PyTorch compatibility")
        return False


if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n🎉 Optimized GSAM2 is working correctly!")
        print("💡 You can now use it in your projects:")
        print("   from gsam2_optimized_complete import OptimizedGSAM2")
        print("   gsam2 = OptimizedGSAM2()")
        print("   results = gsam2.process_frames_optimized(frames, prompt)")
    else:
        print("\n🔧 Please check the setup and try again.")