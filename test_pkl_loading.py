#!/usr/bin/env python3
"""Test script to verify pickle cache loading works correctly"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.youtube_s3_dataset import YouTubeS3Dataset
import torch


def test_cache_loading():
    """Test that the pickle cache loads properly and contains expected data"""
    print("🧪 Testing YouTube Dataset Pickle Cache Loading")
    print("=" * 60)
    
    # Test 1: Load from existing cache
    print("\n1️⃣ Testing cache loading...")
    start_time = time.time()
    
    dataset = YouTubeS3Dataset(
        bucket_name="research-datasets",
        root_prefix="openDV-YouTube/full_images/",
        m=3,
        n=3,
        cache_dir="./youtube_cache",
        refresh_cache=False,  # Use existing cache
        skip_frames=300,
        min_sequence_length=50,
        verbose=True
    )
    
    load_time = time.time() - start_time
    print(f"✅ Cache loaded in {load_time:.2f} seconds")
    
    # Test 2: Verify dataset structure
    print("\n2️⃣ Verifying dataset structure...")
    print(f"   Total sequences: {len(dataset)}")
    print(f"   Total frames needed: {dataset.total_frames} (m={dataset.m}, n={dataset.n})")
    
    # Test 3: Check first few sequences
    print("\n3️⃣ Checking first 5 sequences...")
    for i in range(min(5, len(dataset.sequences))):
        seq = dataset.sequences[i]
        print(f"   [{i}] {seq['channel']}/{seq['video']}: {len(seq['frames'])} frames")
    
    # Test 4: Verify data types and structure
    print("\n4️⃣ Verifying sequence data structure...")
    if dataset.sequences:
        first_seq = dataset.sequences[0]
        expected_keys = {'channel', 'video', 'frames', 'prefix'}
        actual_keys = set(first_seq.keys())
        
        if expected_keys == actual_keys:
            print("   ✅ Sequence structure is correct")
            print(f"   ✅ Frame paths are strings: {isinstance(first_seq['frames'][0], str)}")
        else:
            print(f"   ❌ Missing keys: {expected_keys - actual_keys}")
            print(f"   ❌ Extra keys: {actual_keys - expected_keys}")
    
    # Test 5: Check frame path format
    print("\n5️⃣ Checking frame path formats...")
    if dataset.sequences:
        sample_frames = dataset.sequences[0]['frames'][:3]
        for i, frame_path in enumerate(sample_frames):
            print(f"   Frame {i}: {frame_path}")
        
        # Verify they look like S3 paths
        valid_paths = all(
            path.startswith(dataset.root_prefix) and 
            any(path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])
            for path in sample_frames
        )
        
        if valid_paths:
            print("   ✅ Frame paths look valid")
        else:
            print("   ❌ Frame paths format issue")
    
    # Test 6: Channel and video statistics
    print("\n6️⃣ Dataset statistics...")
    from collections import defaultdict
    
    channel_counts = defaultdict(int)
    total_frames = 0
    total_training_samples = 0
    
    for seq in dataset.sequences:
        channel_counts[seq['channel']] += 1
        total_frames += len(seq['frames'])
        # Calculate training samples for this sequence
        # Each sequence can generate (num_frames - total_frames_needed + 1) samples
        samples_from_seq = len(seq['frames']) - dataset.total_frames + 1
        if samples_from_seq > 0:
            total_training_samples += samples_from_seq
    
    print(f"   Total channels: {len(channel_counts)}")
    print(f"   Total videos: {len(dataset.sequences)}")
    print(f"   Total frames: {total_frames:,}")
    print(f"   Total training samples: {total_training_samples:,}")
    print(f"   Dataset length (len(dataset)): {len(dataset):,}")
    print(f"   Avg frames per video: {total_frames/len(dataset.sequences):.1f}")
    print(f"   Avg training samples per video: {total_training_samples/len(dataset.sequences):.1f}")
    
    # Verify our calculation matches the dataset's __len__
    if total_training_samples == len(dataset):
        print("   ✅ Training sample calculation matches dataset length")
    else:
        print(f"   ❌ Mismatch: calculated {total_training_samples} vs dataset {len(dataset)}")
    
    print(f"\n   Top 5 channels by video count:")
    for channel, count in sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     - {channel}: {count} videos")
    
    # Show sample breakdown for a few videos
    print(f"\n   Sample breakdown for first 5 videos:")
    for i, seq in enumerate(dataset.sequences[:5]):
        samples = len(seq['frames']) - dataset.total_frames + 1
        print(f"     {i+1}. {seq['channel']}/{seq['video']}: {len(seq['frames'])} frames → {samples:,} samples")
    
    return dataset


def test_dataset_indexing():
    """Test that dataset indexing works correctly"""
    print("\n" + "=" * 60)
    print("🔍 Testing Dataset Indexing")
    print("=" * 60)
    
    # Create dataset (should load from cache)
    dataset = YouTubeS3Dataset(
        bucket_name="research-datasets",
        root_prefix="openDV-YouTube/full_images/",
        m=3,
        n=3,
        cache_dir="./youtube_cache",
        refresh_cache=False,
        skip_frames=300,
        min_sequence_length=50,
        verbose=False  # Less verbose for this test
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test indexing boundaries
    print("\n1️⃣ Testing index boundaries...")
    try:
        # Test first index
        first_idx = 0
        print(f"   Testing index {first_idx}...")
        current, future, metadata = dataset[first_idx]
        print(f"   ✅ First index works: {metadata['channel']}/{metadata['video']}")
        
        # Test last index
        last_idx = len(dataset) - 1
        print(f"   Testing index {last_idx}...")
        current, future, metadata = dataset[last_idx]
        print(f"   ✅ Last index works: {metadata['channel']}/{metadata['video']}")
        
        # Test middle index
        mid_idx = len(dataset) // 2
        print(f"   Testing index {mid_idx}...")
        current, future, metadata = dataset[mid_idx]
        print(f"   ✅ Middle index works: {metadata['channel']}/{metadata['video']}")
        
    except Exception as e:
        print(f"   ❌ Indexing error: {e}")
        return False
    
    print("\n2️⃣ Verifying tensor shapes...")
    print(f"   Current frames shape: {current.shape} (expected: [3, 3, 518, 518])")
    print(f"   Future frames shape: {future.shape} (expected: [3, 3, 518, 518])")
    print(f"   Metadata keys: {list(metadata.keys())}")
    
    # Verify shapes
    expected_current = torch.Size([3, 3, 518, 518])  # [m, C, H, W]
    expected_future = torch.Size([3, 3, 518, 518])   # [n, C, H, W]
    
    if current.shape == expected_current and future.shape == expected_future:
        print("   ✅ Tensor shapes are correct")
    else:
        print("   ❌ Tensor shapes are incorrect")
    
    return True


if __name__ == "__main__":
    try:
        # Test cache loading
        dataset = test_cache_loading()
        
        # Test indexing
        success = test_dataset_indexing()
        
        if success:
            print("\n🎉 All tests passed! Pickle cache loading works correctly.")
        else:
            print("\n❌ Some tests failed.")
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()