#!/usr/bin/env python3
"""
Benchmark YouTube S3 Dataset loading performance.
Measures time to load 1, 100, 1000, and 10000 samples.
"""

import time
import numpy as np
from utils.youtube_s3_dataset import YouTubeS3Dataset
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os

def benchmark_single_sample(dataset, num_samples=10):
    """Benchmark loading single samples."""
    print(f"\n🔍 Benchmarking single sample loading (average over {num_samples} samples)...")
    
    times = []
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        
        start_time = time.time()
        sample = dataset[idx]
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        if i == 0:
            # Print sample info
            X, y = sample
            print(f"   Sample shape: X={X.shape}, y={y.shape}")
            print(f"   Sample dtype: X={X.dtype}, y={y.dtype}")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"   ✅ Single sample: {avg_time:.3f}s ± {std_time:.3f}s")
    return avg_time

def benchmark_batch_loading(dataset, batch_size, num_batches, num_workers=0):
    """Benchmark loading multiple samples using DataLoader."""
    total_samples = batch_size * num_batches
    print(f"\n🔍 Benchmarking {total_samples} samples (batch_size={batch_size}, num_workers={num_workers})...")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Warm up
    print("   Warming up DataLoader...")
    warmup_iter = iter(dataloader)
    for _ in range(min(3, num_batches)):
        try:
            _ = next(warmup_iter)
        except StopIteration:
            break
    
    # Actual benchmark
    start_time = time.time()
    
    batch_times = []
    data_iter = iter(dataloader)
    
    for i in tqdm(range(num_batches), desc="   Loading batches"):
        batch_start = time.time()
        try:
            batch = next(data_iter)
        except StopIteration:
            print(f"   ⚠️ Dataset exhausted after {i} batches")
            break
        batch_end = time.time()
        
        batch_times.append(batch_end - batch_start)
        
        # Force GPU sync if using CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    actual_samples = len(batch_times) * batch_size
    avg_batch_time = np.mean(batch_times)
    avg_sample_time = total_time / actual_samples
    
    print(f"   ✅ Total time: {total_time:.2f}s")
    print(f"   ✅ Average per batch: {avg_batch_time:.3f}s")
    print(f"   ✅ Average per sample: {avg_sample_time:.3f}s")
    print(f"   ✅ Throughput: {actual_samples/total_time:.1f} samples/sec")
    
    return total_time, avg_sample_time, actual_samples

def main():
    parser = argparse.ArgumentParser(description='Benchmark YouTube S3 Dataset')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--skip-large', action='store_true', help='Skip 10000 sample test')
    args = parser.parse_args()
    
    print("🚀 YouTube S3 Dataset Benchmark")
    print("=" * 60)
    
    # Load config
    from yacs.config import CfgNode as CN
    from config.defaults import get_cfg_defaults
    
    cfg = get_cfg_defaults()
    if os.path.exists(args.config):
        cfg.merge_from_file(args.config)
    
    # Force YouTube dataset
    cfg.DATASET.USE_YOUTUBE = True
    cfg.DATASET.USE_S3 = False
    
    print(f"📊 Configuration:")
    print(f"   S3 Bucket: {cfg.DATASET.S3_BUCKET}")
    print(f"   YouTube Root: {cfg.DATASET.YOUTUBE_ROOT_PREFIX}")
    print(f"   Cache Dir: {cfg.DATASET.YOUTUBE_CACHE_DIR}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Num Workers: {args.num_workers}")
    
    # Create dataset
    print(f"\n📂 Creating YouTube S3 Dataset...")
    dataset_start = time.time()
    
    dataset = YouTubeS3Dataset(
        bucket_name=cfg.DATASET.S3_BUCKET,
        root_prefix=cfg.DATASET.YOUTUBE_ROOT_PREFIX,
        m=cfg.MODEL.M,
        n=cfg.MODEL.N,
        transform=None,  # No preprocessing for timing
        region_name=cfg.DATASET.get('S3_REGION', 'us-phoenix-1'),
        cache_dir=cfg.DATASET.YOUTUBE_CACHE_DIR,
        refresh_cache=cfg.DATASET.get('YOUTUBE_REFRESH_CACHE', False),
        min_sequence_length=cfg.DATASET.get('YOUTUBE_MIN_SEQUENCE_LENGTH', 50),
        skip_frames=cfg.DATASET.get('YOUTUBE_SKIP_FRAMES', 300),
        max_workers=cfg.DATASET.get('YOUTUBE_MAX_WORKERS', 32),
        verbose=True,
        use_async=False
    )
    
    dataset_end = time.time()
    print(f"   Dataset creation time: {dataset_end - dataset_start:.2f}s")
    print(f"   Total samples available: {len(dataset):,}")
    
    # Run benchmarks
    results = {}
    
    # 1. Single sample benchmark
    avg_single_time = benchmark_single_sample(dataset, num_samples=10)
    results['1_sample'] = avg_single_time
    
    # 2. 100 samples
    if len(dataset) >= 100:
        total_time, avg_time, actual = benchmark_batch_loading(
            dataset, args.batch_size, 100 // args.batch_size, args.num_workers
        )
        results['100_samples'] = {
            'total_time': total_time,
            'avg_per_sample': avg_time,
            'actual_samples': actual
        }
    
    # 3. 1000 samples
    if len(dataset) >= 1000:
        total_time, avg_time, actual = benchmark_batch_loading(
            dataset, args.batch_size, 1000 // args.batch_size, args.num_workers
        )
        results['1000_samples'] = {
            'total_time': total_time,
            'avg_per_sample': avg_time,
            'actual_samples': actual
        }
    
    # 4. 10000 samples (optional)
    if len(dataset) >= 10000 and not args.skip_large:
        total_time, avg_time, actual = benchmark_batch_loading(
            dataset, args.batch_size, 10000 // args.batch_size, args.num_workers
        )
        results['10000_samples'] = {
            'total_time': total_time,
            'avg_per_sample': avg_time,
            'actual_samples': actual
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    print(f"\n1 sample (average): {results['1_sample']:.3f}s")
    
    if '100_samples' in results:
        r = results['100_samples']
        print(f"\n100 samples:")
        print(f"  - Total time: {r['total_time']:.2f}s")
        print(f"  - Per sample: {r['avg_per_sample']:.3f}s")
        print(f"  - Throughput: {r['actual_samples']/r['total_time']:.1f} samples/sec")
    
    if '1000_samples' in results:
        r = results['1000_samples']
        print(f"\n1000 samples:")
        print(f"  - Total time: {r['total_time']:.2f}s")
        print(f"  - Per sample: {r['avg_per_sample']:.3f}s")
        print(f"  - Throughput: {r['actual_samples']/r['total_time']:.1f} samples/sec")
    
    if '10000_samples' in results:
        r = results['10000_samples']
        print(f"\n10000 samples:")
        print(f"  - Total time: {r['total_time']:.2f}s")
        print(f"  - Per sample: {r['avg_per_sample']:.3f}s")
        print(f"  - Throughput: {r['actual_samples']/r['total_time']:.1f} samples/sec")
    
    # Performance tips
    print("\n💡 Performance Tips:")
    if args.num_workers == 0:
        print("  - Try increasing --num-workers for better throughput")
    if args.batch_size == 1:
        print("  - Try increasing --batch-size for better GPU utilization")
    print("  - Ensure good network connectivity to S3")
    print("  - Consider using local caching for frequently accessed data")

if __name__ == "__main__":
    main()