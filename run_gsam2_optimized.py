#!/usr/bin/env python3
"""
Example script showing how to use the optimized GSAM2.

This script demonstrates:
1. Basic usage with dummy frames
2. Real usage with image files
3. Performance comparison
"""

import numpy as np
import cv2
import glob
import time
from gsam2_optimized_complete import OptimizedGSAM2, benchmark_gsam2_performance


def load_frames_from_images(image_folder: str, max_frames: int = 10):
    """Load RGB frames from a folder of images."""
    image_paths = sorted(glob.glob(f"{image_folder}/*.jpg") + glob.glob(f"{image_folder}/*.png"))
    
    if not image_paths:
        print(f"No images found in {image_folder}")
        return []
    
    frames = []
    for i, path in enumerate(image_paths[:max_frames]):
        # Load image and convert BGR to RGB
        img = cv2.imread(path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img_rgb)
            print(f"Loaded frame {i+1}: {path}")
        
        if len(frames) >= max_frames:
            break
    
    return frames


def example_with_dummy_frames():
    """Example using randomly generated frames."""
    print("🎬 Example 1: Using dummy frames")
    print("-" * 40)
    
    # Create dummy RGB frames
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
    print(f"Created {len(frames)} dummy frames (480x640)")
    
    # Initialize optimized GSAM2
    gsam2 = OptimizedGSAM2()
    
    # Process frames
    start_time = time.time()
    results = gsam2.process_frames_optimized(
        frames, 
        "person. vehicle. bicycle.",
        use_batching=True,
        batch_size=4
    )
    process_time = time.time() - start_time
    
    print(f"✅ Processing completed in {process_time:.2f}s")
    print(f"📊 Results: {results['num_objects']} objects detected in {results['num_frames']} frames")
    print(f"🏷️ Labels: {results['labels']}")
    

def example_with_real_images(image_folder: str):
    """Example using real image files."""
    print(f"\n🖼️ Example 2: Using images from {image_folder}")
    print("-" * 40)
    
    # Load frames from images
    frames = load_frames_from_images(image_folder, max_frames=8)
    
    if not frames:
        print("❌ No frames loaded, skipping this example")
        return
    
    print(f"Loaded {len(frames)} frames")
    
    # Initialize optimized GSAM2
    gsam2 = OptimizedGSAM2()
    
    # Process frames with different prompts
    prompts = [
        "person. vehicle.",
        "car. truck. bicycle.",
        "traffic light. road sign."
    ]
    
    for prompt in prompts:
        print(f"\n🔍 Testing prompt: '{prompt}'")
        start_time = time.time()
        
        results = gsam2.process_frames_optimized(
            frames, 
            prompt,
            use_batching=True,
            batch_size=4,
            box_threshold=0.3,
            text_threshold=0.25
        )
        
        process_time = time.time() - start_time
        print(f"⚡ Time: {process_time:.2f}s")
        print(f"📊 Objects: {results['num_objects']} ({results['labels']})")


def performance_comparison():
    """Compare original vs optimized performance."""
    print(f"\n⚡ Example 3: Performance Comparison")
    print("-" * 40)
    
    # Create test frames
    frames = [np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8) for _ in range(6)]
    prompt = "vehicle. person."
    
    try:
        # Run benchmark
        benchmark_results = benchmark_gsam2_performance(frames, prompt)
        
        speedup = benchmark_results['speedup']
        if speedup > 10:
            print(f"🚀 Optimization achieved {speedup:.1f}x speedup!")
        elif speedup > 2:
            print(f"⚡ Good optimization: {speedup:.1f}x speedup")
        else:
            print(f"📈 Modest optimization: {speedup:.1f}x speedup")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print("💡 This might happen if original GSAM2 has issues")


def main():
    """Main function to run all examples."""
    print("🚀 OptimizedGSAM2 Demo")
    print("=" * 50)
    
    try:
        # Example 1: Dummy frames
        example_with_dummy_frames()
        
        # Example 2: Real images (if available)
        image_folders = [
            "./sample_images",
            "./images", 
            "./test_images"
        ]
        
        for folder in image_folders:
            import os
            if os.path.exists(folder):
                example_with_real_images(folder)
                break
        else:
            print(f"\n💡 To test with real images, create a folder with .jpg/.png files:")
            print("   mkdir sample_images")
            print("   # Add some .jpg or .png files to sample_images/")
        
        # Example 3: Performance comparison (optional)
        try_benchmark = input("\n🤔 Run performance benchmark? (y/n): ").lower().startswith('y')
        if try_benchmark:
            performance_comparison()
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure SAM2 and Grounding DINO are properly installed")


if __name__ == "__main__":
    main()