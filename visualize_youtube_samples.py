#!/usr/bin/env python3
"""Visualize samples from the YouTube dataset"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.youtube_s3_dataset import YouTubeS3Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import random


def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor for visualization"""
    # tensor shape: [C, H, W]
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def tensor_to_image(tensor):
    """Convert tensor to numpy image for matplotlib"""
    # tensor shape: [C, H, W]
    denorm = denormalize_tensor(tensor)
    image = denorm.permute(1, 2, 0).numpy()  # [H, W, C]
    return np.clip(image, 0, 1)


def visualize_single_sample(dataset, sample_idx=None):
    """Visualize a single training sample with current and future frames"""
    
    if sample_idx is None:
        sample_idx = random.randint(0, len(dataset) - 1)
    
    print(f"📸 Loading sample {sample_idx:,} / {len(dataset):,}")
    
    try:
        current_frames, future_frames, metadata = dataset[sample_idx]
        
        print(f"   Channel: {metadata['channel']}")
        print(f"   Video: {metadata['video']}")
        print(f"   Start frame: {metadata['start_frame_idx']}")
        print(f"   Current shape: {current_frames.shape}")
        print(f"   Future shape: {future_frames.shape}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"YouTube Sample: {metadata['channel']}/{metadata['video']}\n"
                    f"Sample {sample_idx:,} - Start Frame {metadata['start_frame_idx']}", 
                    fontsize=14)
        
        # Plot current frames (top row)
        for i in range(3):
            ax = axes[0, i]
            image = tensor_to_image(current_frames[i])
            ax.imshow(image)
            ax.set_title(f"Current Frame {i+1}")
            ax.axis('off')
        
        # Plot future frames (bottom row)
        for i in range(3):
            ax = axes[1, i]
            image = tensor_to_image(future_frames[i])
            ax.imshow(image)
            ax.set_title(f"Future Frame {i+1}")
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        save_path = f"youtube_sample_visualization_{sample_idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved visualization to: {save_path}")
        plt.show()
        
        return metadata
        
    except Exception as e:
        print(f"   ❌ Error loading sample {sample_idx}: {e}")
        return None


def visualize_random_samples(dataset, num_samples=3):
    """Visualize multiple random samples"""
    print(f"🎲 Visualizing {num_samples} random samples...")
    print("=" * 60)
    
    for i in range(num_samples):
        print(f"\n📸 Sample {i+1}/{num_samples}")
        metadata = visualize_single_sample(dataset)
        if metadata is None:
            continue


def visualize_channel_samples(dataset, channel_name, num_samples=2):
    """Visualize samples from a specific channel"""
    print(f"📺 Visualizing samples from channel: {channel_name}")
    print("=" * 60)
    
    # Find sequences from this channel
    channel_sequences = [i for i, seq in enumerate(dataset.sequences) if seq['channel'] == channel_name]
    
    if not channel_sequences:
        print(f"❌ No sequences found for channel: {channel_name}")
        return
    
    print(f"Found {len(channel_sequences)} videos from {channel_name}")
    
    # Get sample indices from this channel
    samples_visualized = 0
    for seq_idx in channel_sequences[:num_samples]:
        seq = dataset.sequences[seq_idx]
        print(f"\n🎬 Video: {seq['video']} ({len(seq['frames'])} frames)")
        
        # Calculate the starting index for this sequence in the dataset
        start_idx = 0
        for i in range(seq_idx):
            prev_seq = dataset.sequences[i]
            start_idx += len(prev_seq['frames']) - dataset.total_frames + 1
        
        # Pick a random sample from this video
        seq_length = len(seq['frames']) - dataset.total_frames + 1
        if seq_length > 0:
            local_idx = random.randint(0, seq_length - 1)
            global_idx = start_idx + local_idx
            
            metadata = visualize_single_sample(dataset, global_idx)
            samples_visualized += 1
            
            if samples_visualized >= num_samples:
                break


def show_dataset_overview(dataset):
    """Show overview of available channels and videos"""
    print("📊 Dataset Overview")
    print("=" * 60)
    
    from collections import defaultdict
    channel_stats = defaultdict(list)
    
    for seq in dataset.sequences:
        channel_stats[seq['channel']].append({
            'video': seq['video'],
            'frames': len(seq['frames'])
        })
    
    print(f"📺 Available Channels ({len(channel_stats)} total):")
    for channel, videos in sorted(channel_stats.items(), key=lambda x: len(x[1]), reverse=True):
        total_frames = sum(v['frames'] for v in videos)
        print(f"   - {channel}: {len(videos)} videos, {total_frames:,} frames")
    
    return list(channel_stats.keys())


def main():
    """Main visualization function"""
    print("🎬 YouTube Dataset Visualization Tool")
    print("=" * 60)
    
    # Load dataset
    print("📁 Loading dataset...")
    dataset = YouTubeS3Dataset(
        bucket_name="research-datasets",
        root_prefix="openDV-YouTube/full_images/",
        m=3,
        n=3,
        cache_dir="./youtube_cache",
        refresh_cache=False,
        skip_frames=300,
        min_sequence_length=50,
        verbose=False
    )
    
    print(f"✅ Dataset loaded: {len(dataset):,} training samples")
    
    # Show overview
    available_channels = show_dataset_overview(dataset)
    
    # Visualization options
    while True:
        print("\n" + "=" * 60)
        print("🎨 Visualization Options:")
        print("1. Visualize random samples")
        print("2. Visualize specific channel")
        print("3. Visualize specific sample index")
        print("4. Show dataset overview")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            num_samples = int(input("Number of random samples (default 3): ") or "3")
            visualize_random_samples(dataset, num_samples)
            
        elif choice == "2":
            print(f"\nAvailable channels: {', '.join(available_channels[:10])}")
            if len(available_channels) > 10:
                print(f"... and {len(available_channels) - 10} more")
            
            channel = input("Enter channel name: ").strip()
            num_samples = int(input("Number of samples from this channel (default 2): ") or "2")
            visualize_channel_samples(dataset, channel, num_samples)
            
        elif choice == "3":
            max_idx = len(dataset) - 1
            sample_idx = int(input(f"Enter sample index (0 to {max_idx:,}): "))
            if 0 <= sample_idx <= max_idx:
                visualize_single_sample(dataset, sample_idx)
            else:
                print(f"❌ Index out of range. Must be 0 to {max_idx:,}")
                
        elif choice == "4":
            show_dataset_overview(dataset)
            
        elif choice == "5":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    main()