#!/usr/bin/env python3
"""
Simple benchmark for YouTube S3 Dataset loading performance.
"""

import time
import numpy as np
from utils.youtube_s3_dataset import YouTubeS3Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def benchmark_samples(dataset, num_samples, batch_size=1, num_workers=0, pin_memory=None, prefetch_factor=None):
    """Benchmark loading N samples."""
    # Set defaults
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if prefetch_factor is None:
        prefetch_factor = 2 if num_workers > 0 else None
    
    print(f"\n🔍 Benchmarking {num_samples} samples (batch={batch_size}, workers={num_workers}, pin_mem={pin_memory}, prefetch={prefetch_factor})...")
    
    # For single sample
    if num_samples == 1:
        times = []
        for _ in range(10):  # Average over 10 runs
            idx = np.random.randint(0, min(len(dataset), 1000))
            start = time.time()
            sample = dataset[idx]
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        print(f"   ✅ Single sample: {avg_time:.3f}s (avg over 10 runs)")
        return avg_time
    
    # For multiple samples using DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False
    )
    
    # Warmup
    print("   Warming up...")
    for i, _ in enumerate(dataloader):
        if i >= 2:
            break
    
    # Actual timing
    start_time = time.time()
    samples_loaded = 0
    
    pbar = tqdm(total=num_samples, desc="   Loading")
    for batch in dataloader:
        samples_loaded += batch[0].shape[0]
        pbar.update(batch[0].shape[0])
        
        if samples_loaded >= num_samples:
            break
    pbar.close()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_per_sample = total_time / samples_loaded
    
    print(f"   ✅ Total time: {total_time:.2f}s")
    print(f"   ✅ Per sample: {avg_per_sample:.3f}s")
    print(f"   ✅ Throughput: {samples_loaded/total_time:.1f} samples/sec")
    
    return total_time, avg_per_sample

def main():
    print("🚀 YouTube S3 Dataset Simple Benchmark")
    print("=" * 60)
    
    # Create dataset with minimal config
    print("📂 Creating dataset...")
    start = time.time()
    
    dataset = YouTubeS3Dataset(
        bucket_name="research-datasets",
        root_prefix="openDV-YouTube/full_images/",
        m=3,
        n=3,
        cache_dir="./youtube_cache",
        skip_frames=300,
        max_workers=16,
        verbose=True
    )
    
    print(f"✅ Dataset created in {time.time() - start:.2f}s")
    print(f"📊 Total samples: {len(dataset):,}")
    
    if len(dataset) == 0:
        print("❌ No samples found in dataset!")
        return
    
    # Run benchmarks
    results = {}
    
    # Test different configurations: (samples, batch_size, workers, pin_memory, prefetch_factor)
    configs = [
        (1, 1, 0, False, None),      # Single sample baseline
        (100, 1, 0, False, None),    # 100 samples, no workers
        (100, 1, 4, True, 2),        # 100 samples, 4 workers, with pin_memory
        
        # Different prefetch factors
        # (1000, 8, 4, True, 1),      # prefetch=1
        # (1000, 8, 4, False, 1),      # prefetch=1

        # (1000, 8, 4, True, 4),      # prefetch=4
        # (1000, 8, 4, True, 8),      # prefetch=8
        
    ]
    
    for config in configs:
        num_samples, batch_size, num_workers, pin_memory, prefetch_factor = config
        
        if num_samples > len(dataset):
            print(f"\n⚠️ Skipping {num_samples} samples (dataset only has {len(dataset)})")
            continue
            
        result = benchmark_samples(dataset, num_samples, batch_size, num_workers, pin_memory, prefetch_factor)
        
        config_key = f"{num_samples}_b{batch_size}_w{num_workers}_pin{pin_memory}_pf{prefetch_factor}"
        results[config_key] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    # Group results by sample count
    single_sample_results = {}
    batch_results = {}
    
    for config, result in results.items():
        if config.startswith("1_"):
            single_sample_results[config] = result
        else:
            batch_results[config] = result
    
    # Single sample results
    if single_sample_results:
        print("\n🔍 Single Sample Performance:")
        for config, result in single_sample_results.items():
            print(f"  {config}: {result:.3f}s")
    
    # Batch results - sort by throughput
    if batch_results:
        print("\n🚀 Batch Loading Performance (sorted by throughput):")
        batch_perf = []
        for config, result in batch_results.items():
            if isinstance(result, tuple):
                total_time, avg_time = result
                # Extract sample count from config name
                sample_count = int(config.split('_')[0])
                throughput = sample_count / total_time
                batch_perf.append((throughput, config, total_time, avg_time, sample_count))
        
        # Sort by throughput (highest first)
        batch_perf.sort(reverse=True)
        
        print("\n" + "=" * 80)
        print(f"{'Rank':<4} {'Config':<35} {'Throughput':<12} {'Per Sample':<12} {'Total Time':<10}")
        print("=" * 80)
        
        for i, (throughput, config, total_time, avg_time, sample_count) in enumerate(batch_perf):
            rank = i + 1
            config_short = config.replace(f"{sample_count}_", "")
            print(f"{rank:<4} {config_short:<35} {throughput:>8.1f} s/s  {avg_time:>8.3f}s     {total_time:>6.2f}s")
        
        # Best configuration
        if batch_perf:
            best_throughput, best_config, _, _, _ = batch_perf[0]
            print(f"\n🏆 Best configuration: {best_config}")
            print(f"   Throughput: {best_throughput:.1f} samples/sec")
            
            # Performance recommendations
            print(f"\n💡 Performance Insights:")
            batch_16_results = [p for p in batch_perf if "b16_w" in p[1]]
            if len(batch_16_results) >= 2:
                print("   - Batch size 16 seems to be a good sweet spot")
            
            pin_true = [p for p in batch_perf if "pinTrue" in p[1]]
            pin_false = [p for p in batch_perf if "pinFalse" in p[1]]
            if pin_true and pin_false:
                avg_pin_true = np.mean([p[0] for p in pin_true])
                avg_pin_false = np.mean([p[0] for p in pin_false])
                if avg_pin_true > avg_pin_false * 1.1:
                    print("   - pin_memory=True provides significant speedup")
                    
            worker_results = {}
            for throughput, config, _, _, _ in batch_perf:
                if "w2_" in config or "w4_" in config or "w8_" in config:
                    workers = config.split("w")[1].split("_")[0]
                    if workers not in worker_results:
                        worker_results[workers] = []
                    worker_results[workers].append(throughput)
            
            if len(worker_results) >= 2:
                best_worker_count = max(worker_results.keys(), 
                                      key=lambda k: np.mean(worker_results[k]))
                print(f"   - Optimal worker count appears to be {best_worker_count}")

if __name__ == "__main__":
    main()