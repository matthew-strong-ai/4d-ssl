import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np
import random

from simple_s3_dataset import S3Dataset


class S3SequenceLearningDataset(Dataset):
    """
    Dataset that wraps S3Dataset for sequence learning with Pi3.
    Each item returns (X, y) where:
    - X: first m frames from a sequence (input frames)
    - y: last n frames from the same sequence (target frames)
    
    This matches the interface expected by train_pi3.py while using S3 as the backend.
    
    Args:
        s3_bucket: S3 bucket name
        sequence_prefixes: List of S3 prefixes for image sequences  
        m: Number of input frames (first m frames)
        n: Number of target frames (last n frames)
        image_extension: Image file extension (default: ".jpg")
        transform: Optional transform to apply to images
        aws_access_key_id: AWS access key ID (optional)
        aws_secret_access_key: AWS secret access key (optional) 
        region_name: AWS region name (default: "us-east-1")
        preload_bytes: Whether to preload all bytes for multiprocessing (default: False)
        
    Example:
        dataset = S3SequenceLearningDataset(
            s3_bucket="research-datasets",
            sequence_prefixes=["autonomy_youtube/sf_day/", "autonomy_youtube/smoky_mountains/"],
            m=3,  # 3 input frames
            n=3,  # 3 target frames 
            image_extension=".png"
        )
        
        # Use with Pi3 training
        X, y = dataset[0]  # X: (3, C, H, W), y: (3, C, H, W)
    """
    
    def __init__(
        self,
        s3_bucket: str,
        sequence_prefixes: List[str],
        m: int,  # Input frames
        n: int,  # Target frames
        image_extension: str = ".jpg",
        transform: Optional[Callable] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1",
        preload_bytes: bool = False
    ):
        self.m = m
        self.n = n
        self.total_frames = m + n
        
        print(f"🚀 Creating S3SequenceLearningDataset:")
        print(f"   Bucket: {s3_bucket}")
        print(f"   Sequences: {len(sequence_prefixes)}")
        print(f"   Input frames (m): {m}")
        print(f"   Target frames (n): {n}")
        print(f"   Total frames per sample: {m + n}")
        
        # Create underlying S3Dataset
        self.s3_dataset = S3Dataset(
            bucket_name=s3_bucket,
            sequence_prefixes=sequence_prefixes,
            m=m,
            n=n,
            image_extension=image_extension,
            transform=transform,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            preload_bytes=preload_bytes
        )
        
        print(f"✅ S3SequenceLearningDataset ready with {len(self.s3_dataset)} samples")

    def __len__(self):
        return len(self.s3_dataset)

    def __getitem__(self, idx):
        """
        Get a sequence sample split into input and target frames.
        
        Args:
            idx: Sample index
            
        Returns:
            tuple: (X, y) where:
                - X: Input frames tensor of shape (m, C, H, W)
                - y: Target frames tensor of shape (n, C, H, W)
        """
        # S3Dataset already returns (current_frames, future_frames) which is exactly (X, y)
        X, y = self.s3_dataset[idx]
        
        # Verify shapes match expected m, n
        assert X.shape[0] == self.m, f"Expected {self.m} input frames, got {X.shape[0]}"
        assert y.shape[0] == self.n, f"Expected {self.n} target frames, got {y.shape[0]}"
        
        return X, y
    
    def get_sequence_info(self, idx: int) -> dict:
        """Get metadata about a specific sample"""
        return self.s3_dataset.get_sequence_info(idx)
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Visualize a sample showing input and target frames"""
        self.s3_dataset.visualize_sample(idx, save_path, figsize)
    
    def visualize_random_samples(self, num_samples: int = 6, save_path: Optional[str] = None, figsize: Tuple[int, int] = (20, 12)) -> None:
        """Visualize random samples from the dataset"""
        self.s3_dataset.visualize_random_samples(num_samples, save_path, figsize)
    
    def visualize_sequence_learning_samples(self, num_samples: int = 4, save_path: Optional[str] = None) -> None:
        """
        Visualize random samples specifically for sequence learning.
        Shows clear separation between input frames (X) and target frames (y).
        """
        try:
            import matplotlib.pyplot as plt
            import random
        except ImportError:
            print("matplotlib is required for visualization. Install with: pip install matplotlib")
            return
        
        if len(self) == 0:
            print("Dataset is empty")
            return
        
        # Get random sample indices
        random_indices = random.sample(range(len(self)), min(num_samples, len(self)))
        actual_samples = len(random_indices)
        
        print(f"📊 Visualizing {actual_samples} sequence learning samples (indices: {random_indices})")
        
        # Create figure
        fig = plt.figure(figsize=(20, actual_samples * 4))
        
        # Denormalize function (matches S3Dataset)
        def denormalize(tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return tensor * std + mean
        
        for row, sample_idx in enumerate(random_indices):
            try:
                # Get sample
                X, y = self[sample_idx]  # X: input frames, y: target frames
                info = self.get_sequence_info(sample_idx)
                
                # Plot input frames (X)
                for frame_idx in range(self.m):
                    ax = plt.subplot(actual_samples * 2, max(self.m, self.n), 
                                   row * 2 * max(self.m, self.n) + frame_idx + 1)
                    
                    img_tensor = denormalize(X[frame_idx]).clamp(0, 1)
                    img_array = img_tensor.permute(1, 2, 0).numpy()
                    
                    ax.imshow(img_array)
                    if frame_idx == 0:
                        ax.set_title(f'Sample {sample_idx}\nINPUT Frame {frame_idx+1}', fontsize=10, color='blue')
                    else:
                        ax.set_title(f'INPUT Frame {frame_idx+1}', fontsize=10, color='blue')
                    ax.axis('off')
                    
                    # Add blue border for input frames
                    for spine in ax.spines.values():
                        spine.set_edgecolor('blue')
                        spine.set_linewidth(2)
                
                # Plot target frames (y)
                for frame_idx in range(self.n):
                    ax = plt.subplot(actual_samples * 2, max(self.m, self.n),
                                   row * 2 * max(self.m, self.n) + max(self.m, self.n) + frame_idx + 1)
                    
                    img_tensor = denormalize(y[frame_idx]).clamp(0, 1)
                    img_array = img_tensor.permute(1, 2, 0).numpy()
                    
                    ax.imshow(img_array)
                    ax.set_title(f'TARGET Frame {frame_idx+1}', fontsize=10, color='red')
                    ax.axis('off')
                    
                    # Add red border for target frames
                    for spine in ax.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(2)
                        
            except Exception as e:
                print(f"Error visualizing sample {sample_idx}: {e}")
                continue
        
        plt.suptitle(f'S3 Sequence Learning Dataset\n'
                     f'Blue = Input Frames (X), Red = Target Frames (y)\n'
                     f'{actual_samples} samples from {len(self)} total | Sequence: m={self.m}, n={self.n}', 
                     fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📸 Sequence learning visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


def create_s3_sequence_dataset(
    s3_bucket: str,
    sequence_prefixes: List[str],
    m: int = 3,
    n: int = 3,
    **kwargs
) -> S3SequenceLearningDataset:
    """
    Convenience function to create S3SequenceLearningDataset.
    
    Args:
        s3_bucket: S3 bucket name
        sequence_prefixes: List of S3 prefixes
        m: Number of input frames
        n: Number of target frames
        **kwargs: Additional arguments for S3SequenceLearningDataset
    
    Returns:
        S3SequenceLearningDataset: Ready-to-use dataset
    """
    return S3SequenceLearningDataset(
        s3_bucket=s3_bucket,
        sequence_prefixes=sequence_prefixes,
        m=m,
        n=n,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    bucket_name = "research-datasets"
    sequence_prefixes = [
        "autonomy_youtube/sf_day/",
        "autonomy_youtube/smoky_mountains/",
    ]
    
    try:
        print("🧪 Testing S3SequenceLearningDataset...")
        
        # Create dataset
        dataset = S3SequenceLearningDataset(
            s3_bucket=bucket_name,
            sequence_prefixes=sequence_prefixes,
            m=3,  # 3 input frames
            n=3,  # 3 target frames
            image_extension=".png",
            preload_bytes=False  # Set to True for multiprocessing
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test sample loading
            X, y = dataset[0]
            print(f"Sample shapes - X (input): {X.shape}, y (target): {y.shape}")
            
            # Get sample info
            info = dataset.get_sequence_info(0)
            print(f"Sample info: {info}")
            
            # Test visualization
            print("\nVisualizing sequence learning samples...")
            dataset.visualize_sequence_learning_samples(num_samples=2, save_path="s3_sequence_learning_samples.png")
            
            # Test with DataLoader (Pi3 training style)
            from torch.utils.data import DataLoader
            
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=True,
                num_workers=0,  # Set to 0 for S3 compatibility
                prefetch_factor=2 if torch.cuda.is_available() else None,
                pin_memory=torch.cuda.is_available()
            )
            
            print(f"\nTesting DataLoader...")
            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                print(f"Batch {batch_idx}: X.shape={X_batch.shape}, y.shape={y_batch.shape}")
                if batch_idx >= 1:  # Test first 2 batches
                    break
                    
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use this dataset:")
        print("1. Update bucket_name and sequence_prefixes")
        print("2. Ensure AWS credentials are configured")
        print("3. Verify S3 bucket structure matches expected format")