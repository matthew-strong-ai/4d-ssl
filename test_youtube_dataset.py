#!/usr/bin/env python3
"""Test script for YouTube S3 Dataset"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.youtube_s3_dataset import YouTubeS3Dataset
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm


def test_youtube_dataset():
    """Test the YouTube S3 dataset implementation"""
    
    print("🚀 Testing YouTube S3 Dataset...")
    print("-" * 60)
    
    # Create dataset with small parameters for testing
    dataset = YouTubeS3Dataset(
        bucket_name="research-datasets",
        root_prefix="openDV-YouTube/full_images/",
        m=3,  # 3 input frames
        n=3,  # 3 future frames
        cache_dir="./youtube_cache",
        refresh_cache=False,  # Use True on first run
        skip_frames=300,  # Skip first 300 frames
        min_sequence_length=50,
        max_workers=8,
        verbose=True
    )

    import pdb; pdb.set_trace()
    
    # Test basic functionality
    print(f"\n✅ Dataset created successfully!")
    print(f"   Total sequences available: {len(dataset)}")
    
    # Test loading a single sample
    print("\n🧪 Testing single sample loading...")
    start_time = time.time()
    try:
        current, future, metadata = dataset[0]
        load_time = time.time() - start_time
        
        print(f"   ✅ Successfully loaded first sample in {load_time:.2f} seconds")
        print(f"   Current frames shape: {current.shape}")
        print(f"   Future frames shape: {future.shape}")
        print(f"   Channel: {metadata['channel']}")
        print(f"   Video: {metadata['video']}")
        print(f"   Start frame index: {metadata['start_frame_idx']}")
    except Exception as e:
        print(f"   ❌ Error loading sample: {e}")
        return
    
    # Test DataLoader
    print("\n🔄 Testing DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,  # Start with 0 for testing
        pin_memory=True
    )
    
    # Load a few batches
    print("   Loading 3 batches...")
    batch_progress = tqdm(enumerate(dataloader), total=3, desc="Loading batches")
    for i, (current_batch, future_batch, metadata_batch) in batch_progress:
        print(f"   Batch {i+1}:")
        print(f"      Current shape: {current_batch.shape}")
        print(f"      Future shape: {future_batch.shape}")
        print(f"      Channels: {metadata_batch['channel']}")
        
        if i >= 2:  # Only test 3 batches
            break
    
    print("\n✅ All tests passed!")
    
    # Print dataset statistics
    dataset.print_dataset_info()
    
    # Test with multiple workers
    print("\n🔄 Testing with multiple workers...")
    dataloader_multi = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )
    
    start_time = time.time()
    multi_progress = tqdm(enumerate(dataloader_multi), total=5, desc="Multi-worker batches")
    for i, (current_batch, future_batch, _) in multi_progress:
        if i >= 5:  # Test 5 batches
            break
    
    elapsed = time.time() - start_time
    print(f"   ✅ Loaded 5 batches with 2 workers in {elapsed:.2f} seconds")
    print(f"   Average time per batch: {elapsed/5:.2f} seconds")


def test_cache_functionality():
    """Test the caching functionality"""
    print("\n🧪 Testing cache functionality...")
    
    # First run - build cache
    print("   Building cache (first run)...")
    start_time = time.time()
    dataset1 = YouTubeS3Dataset(
        bucket_name="research-datasets",
        root_prefix="openDV-YouTube/full_images/",
        m=3,
        n=3,
        cache_dir="./youtube_cache_test",
        refresh_cache=True,  # Force rebuild
        verbose=False
    )
    build_time = time.time() - start_time
    print(f"   Cache build time: {build_time:.2f} seconds")
    
    # Second run - load from cache
    print("   Loading from cache (second run)...")
    start_time = time.time()
    dataset2 = YouTubeS3Dataset(
        bucket_name="research-datasets",
        root_prefix="openDV-YouTube/full_images/",
        m=3,
        n=3,
        cache_dir="./youtube_cache_test",
        refresh_cache=False,  # Use cache
        verbose=False
    )
    load_time = time.time() - start_time
    print(f"   Cache load time: {load_time:.2f} seconds")
    print(f"   Speedup: {build_time/load_time:.1f}x")
    
    # Verify same data
    assert len(dataset1) == len(dataset2), "Dataset lengths don't match!"
    print("   ✅ Cache test passed!")


if __name__ == "__main__":
    # Run tests
    test_youtube_dataset()
    test_cache_functionality()
    
    print("\n🎉 All tests completed!")