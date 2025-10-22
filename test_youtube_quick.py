#!/usr/bin/env python3
"""Quick test for YouTube S3 Dataset"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.youtube_s3_dataset import YouTubeS3Dataset

def test_youtube_quick():
    """Quick test of the YouTube S3 dataset"""
    
    print("🚀 Quick YouTube S3 Dataset Test...")
    print("-" * 50)
    
    # Create dataset with minimal parameters
    dataset = YouTubeS3Dataset(
        bucket_name="research-datasets",
        root_prefix="openDV-YouTube/full_images/",
        m=2,  # Small values for quick test
        n=2,
        cache_dir="./youtube_cache_quick",
        refresh_cache=False,
        skip_frames=100,  # Smaller skip
        min_sequence_length=10,  # Lower minimum
        max_workers=4,
        verbose=True
    )
    
    print(f"✅ Dataset created with {len(dataset)} sequences")
    
    if len(dataset) > 0:
        print("\n🧪 Testing single sample...")
        try:
            current, future, metadata = dataset[0]
            print(f"   ✅ Sample loaded successfully")
            print(f"   Current shape: {current.shape}")
            print(f"   Future shape: {future.shape}")
            print(f"   Channel: {metadata['channel']}")
            print(f"   Video: {metadata['video']}")
        except Exception as e:
            print(f"   ❌ Error loading sample: {e}")
    else:
        print("❌ No sequences found in dataset")

if __name__ == "__main__":
    test_youtube_quick()