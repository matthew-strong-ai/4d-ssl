#!/usr/bin/env python3
"""
Script to save sample images from your datasets for visualization.
"""

import os
import argparse
import sys

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SpaTrackerV2"))

from consecutive_images_dataset import ConsecutiveImagesDataset, get_default_transforms
from SpaTrackerV2.ssl_image_dataset import SequenceLearningDataset

def save_consecutive_samples(image_dir: str, output_dir: str = "./consecutive_samples"):
    """Save samples from ConsecutiveImagesDataset"""
    print(f"🔍 Loading ConsecutiveImagesDataset from: {image_dir}")
    
    dataset = ConsecutiveImagesDataset(
        image_dir=image_dir,
        batch_size=6,  # 6 consecutive frames per batch
        transform=get_default_transforms(),
        start_frame_idx=300  # Skip first 300 frames
    )
    
    print(f"📊 Dataset info:")
    info = dataset.get_image_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Save sample batches
    dataset.save_sample_images(
        output_dir=output_dir,
        num_batches=3,  # Save 3 batches
        denormalize=False
    )
    
    return output_dir

def save_sequence_samples(root_dir: str, output_dir: str = "./sequence_samples"):
    """Save samples from SequenceLearningDataset"""
    print(f"🔍 Loading SequenceLearningDataset from: {root_dir}")
    
    # Find all subdirectories
    image_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                  if os.path.isdir(os.path.join(root_dir, d))]
    
    print(f"📁 Found {len(image_dirs)} subdirectories:")
    for d in image_dirs[:5]:  # Show first 5
        print(f"   {d}")
    if len(image_dirs) > 5:
        print(f"   ... and {len(image_dirs) - 5} more")
    
    dataset = SequenceLearningDataset(
        image_dirs=image_dirs,
        m=3,  # 3 input frames
        n=3,  # 3 target frames
        transform=get_default_transforms()
    )
    
    print(f"📊 Total sequences: {len(dataset)}")
    
    # Save sample sequences
    dataset.save_sample_sequences(
        output_dir=output_dir,
        num_sequences=3,  # Save 3 sequences
        denormalize=True
    )
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Save sample images from datasets")
    parser.add_argument("--dataset_type", type=str, choices=["consecutive", "sequence"], 
                       required=True, help="Type of dataset to sample from")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to data (single dir for consecutive, root dir for sequence)")
    parser.add_argument("--output_dir", type=str, default="./sample_images",
                       help="Output directory for saved images")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"❌ Data path does not exist: {args.data_path}")
        return
    
    try:
        if args.dataset_type == "consecutive":
            output_dir = save_consecutive_samples(args.data_path, args.output_dir)
        else:  # sequence
            output_dir = save_sequence_samples(args.data_path, args.output_dir)
        
        print(f"\n🎉 Success! Sample images saved to: {output_dir}")
        print(f"\nTo view the images:")
        print(f"   ls -la {output_dir}")
        print(f"   find {output_dir} -name '*.png' | head -10")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()