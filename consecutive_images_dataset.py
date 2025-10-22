import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ConsecutiveImagesDataset(Dataset):
    """
    PyTorch Dataset that loads batches of n consecutive images from a directory.
    
    Args:
        image_dir (str): Path to directory containing images
        batch_size (int): Number of consecutive images to return in each batch
        transform (callable, optional): Optional transform to be applied to each image
        image_extensions (tuple): Valid image file extensions
        sort_key (callable, optional): Function to sort image filenames (default: natural sort)
        start_frame_idx (int): Starting frame index (default: 0). Only images from this index onwards will be used.
    """
    
    def __init__(
        self,
        image_dir: str,
        batch_size: int,
        transform: Optional[Callable] = None,
        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
        sort_key: Optional[Callable] = None,
        start_frame_idx: int = 300
    ):
        

        self.image_dir = image_dir
        self.batch_size = batch_size
        self.start_frame_idx = start_frame_idx
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
        self.image_extensions = image_extensions
        
        # Get all image files
        self.image_files = self._get_image_files()
        
        # Sort files (default: natural sort for numbered sequences)
        if sort_key is None:
            self.image_files = self._natural_sort(self.image_files)
        else:
            self.image_files.sort(key=sort_key)
        
        # If no images, make dataset empty and return early
        if len(self.image_files) == 0:
            self._empty = True
            return
        else:
            self._empty = False
        # Validate start_frame_idx
        if start_frame_idx < 0:
            raise ValueError(f"start_frame_idx must be non-negative, got {start_frame_idx}")
        if start_frame_idx >= len(self.image_files):
            raise ValueError(f"start_frame_idx ({start_frame_idx}) must be less than total number of images ({len(self.image_files)})")
        # Check if we have enough images from start_frame_idx
        available_frames = len(self.image_files) - start_frame_idx
        if available_frames < batch_size:
            raise ValueError(f"Not enough images from start_frame_idx {start_frame_idx}. Found {available_frames}, need at least {batch_size}")

    def _get_image_files(self) -> List[str]:
        """Get all valid image files from the directory."""
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Directory does not exist: {self.image_dir}")
            
        image_files = []
        for filename in os.listdir(self.image_dir):
            if filename.lower().endswith(self.image_extensions):
                image_files.append(filename)
        
        if not image_files:
            print(f"No valid image files found in {self.image_dir}")
            return []

        return image_files
    
    def _natural_sort(self, file_list: List[str]) -> List[str]:
        """
        Sort filenames naturally (e.g., img1.jpg, img2.jpg, img10.jpg)
        instead of lexicographically (img1.jpg, img10.jpg, img2.jpg)
        """
        import re
        
        def natural_key(filename):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', filename)]
        
        return sorted(file_list, key=natural_key)
    
    def __len__(self) -> int:
        if hasattr(self, '_empty') and self._empty:
            return 0
        return len(self.image_files) - self.start_frame_idx - self.batch_size + 1
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if hasattr(self, '_empty') and self._empty:
            raise IndexError("Empty dataset: no images available.")
        
        """
        Get a batch of consecutive images starting from index idx.
        
        Args:
            idx (int): Starting index for the batch
            
        Returns:
            torch.Tensor: Tensor of shape (batch_size, C, H, W) containing consecutive images
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
        
        # Get consecutive image filenames starting from start_frame_idx
        actual_idx = self.start_frame_idx + idx
        batch_files = self.image_files[actual_idx:actual_idx + self.batch_size]
        
        # Load and process images
        images = []
        for filename in batch_files:
            image_path = os.path.join(self.image_dir, filename)
            
            try:
                # Load image using OpenCV (faster than PIL)
                image_bgr = cv2.imread(image_path)
                if image_bgr is None:
                    # Fallback to PIL if OpenCV fails
                    image = Image.open(image_path).convert('RGB')
                else:
                    # Convert BGR to RGB (OpenCV loads as BGR by default)
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image for transform compatibility
                    image = Image.fromarray(image_rgb)
                
                # Apply transform
                if self.transform:
                    image = self.transform(image)
                
                images.append(image)
                
            except Exception as e:
                raise RuntimeError(f"Error loading image {image_path}: {e}")
        
        # Stack images into a batch tensor
        batch_tensor = torch.stack(images)  # Shape: (batch_size, C, H, W)
        
        return batch_tensor
    
    def get_image_info(self) -> dict:
        """Get information about the dataset."""
        return {
            'total_images': len(self.image_files),
            'start_frame_idx': self.start_frame_idx,
            'available_images': len(self.image_files) - self.start_frame_idx,
            'batch_size': self.batch_size,
            'num_batches': len(self),
            'image_dir': self.image_dir,
            'sample_files': self.image_files[self.start_frame_idx:self.start_frame_idx + 5]  # Show first 5 files from start_frame_idx
        }
    
    def save_sample_images(self, output_dir: str = "./sample_images", num_batches: int = 3, denormalize: bool = True):
        """
        Save sample input images to disk for visualization.
        
        Args:
            output_dir: Directory to save sample images
            num_batches: Number of batches to save
            denormalize: Whether to denormalize images (reverse ImageNet normalization)
        """
        import os
        import random
        from PIL import Image as PILImage
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🖼️ Saving {num_batches} random sample batches to {output_dir}")
        
        # ImageNet normalization values for denormalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Generate random batch indices
        max_batches = min(num_batches, len(self))
        random_indices = random.sample(range(len(self)), max_batches)
        print(f"📊 Selected random batch indices: {random_indices}")
        
        for i, batch_idx in enumerate(random_indices):
            try:
                # Get a batch of images
                batch_tensor = self[batch_idx]  # Shape: (batch_size, C, H, W)
                
                # Create batch directory with sample number for clarity
                batch_dir = os.path.join(output_dir, f"sample_{i:03d}_batch_{batch_idx:03d}")
                os.makedirs(batch_dir, exist_ok=True)
                
                print(f"  📁 Saving sample {i} (batch {batch_idx}) - {batch_tensor.shape[0]} images")
                
                for img_idx in range(batch_tensor.shape[0]):
                    # Get single image tensor
                    img_tensor = batch_tensor[img_idx]  # Shape: (C, H, W)
                    
                    # Convert to numpy and transpose to (H, W, C)
                    img_array = img_tensor.cpu().numpy().transpose(1, 2, 0)
                    
                    # Denormalize if requested
                    if denormalize:
                        img_array = img_array * std + mean
                    
                    # Clip values to [0, 1] and convert to [0, 255]
                    img_array = np.clip(img_array, 0, 1)
                    img_array = (img_array * 255).astype(np.uint8)
                    
                    # Convert to PIL Image and save
                    pil_image = PILImage.fromarray(img_array)
                    
                    # Get original filename for reference
                    actual_idx = self.start_frame_idx + batch_idx
                    original_filename = self.image_files[actual_idx + img_idx]
                    
                    # Save with descriptive filename
                    save_filename = f"frame_{img_idx:02d}_{original_filename}"
                    save_path = os.path.join(batch_dir, save_filename)
                    pil_image.save(save_path)
                
                print(f"    ✅ Saved {batch_tensor.shape[0]} images to {batch_dir}")
                
            except Exception as e:
                print(f"    ❌ Error saving sample {i} (batch {batch_idx}): {e}")
                continue
        
        print(f"🎉 Sample images saved to {output_dir}")
        print(f"    View with: ls -la {output_dir}/*/")
        
        return output_dir


# Example usage and utility functions
def create_dataloader(
    image_dir: str,
    batch_size: int,
    dataloader_batch_size: int = 1,
    transform: Optional[Callable] = None,
    shuffle: bool = False,
    num_workers: int = 0,
    start_frame_idx: int = 0
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the ConsecutiveImagesDataset.
    
    Args:
        image_dir: Directory containing images
        batch_size: Number of consecutive images per sample
        dataloader_batch_size: Number of samples per DataLoader batch
        transform: Optional image transform
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for loading
        start_frame_idx: Starting frame index (default: 0)
    
    Returns:
        torch.utils.data.DataLoader
    """
    dataset = ConsecutiveImagesDataset(
        image_dir=image_dir,
        batch_size=batch_size,
        transform=transform,
        start_frame_idx=start_frame_idx
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=dataloader_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


# Example transforms for different use cases
def get_default_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get default image transforms for training."""
    return transforms.Compose([
        # transforms.Resize(image_size),
        transforms.ToTensor(),
    ])


def get_video_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Get transforms optimized for video/temporal data."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # Normalize per frame for video data
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ConsecutiveImagesDataset")
    parser.add_argument("--image_dir", type=str, default='/home/matthew_strong/Desktop/autonomy-ssl/video_frames/smoky_mountains', help="Directory containing images")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of consecutive images per batch")
    parser.add_argument("--dataloader_batch_size", type=int, default=2, help="DataLoader batch size")
    parser.add_argument("--start_frame_idx", type=int, default=300, help="Starting frame index")
    
    args = parser.parse_args()

    # Create dataset
    dataset = ConsecutiveImagesDataset(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        transform=get_default_transforms(),
        start_frame_idx=args.start_frame_idx
    )

    
    # Print dataset info
    print("Dataset Info:")
    info = dataset.get_image_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create dataloader
    dataloader = create_dataloader(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        dataloader_batch_size=args.dataloader_batch_size,
        shuffle=True,
        num_workers=2,
        start_frame_idx=args.start_frame_idx
    )
    
    # Test loading a few batches
    print(f"\nTesting DataLoader:")
    for i, batch in enumerate(dataloader):
        print(f"  Batch {i}: shape {batch.shape}")
        if i >= 2:  # Only show first 3 batches
            break
    
    # Visualize a few batches using matplotlib
    import matplotlib.pyplot as plt

    def show_batch(batch_tensor, title=None, max_cols=5):
        # batch_tensor: (batch_size, C, H, W) or (batch_size, H, W)
        if isinstance(batch_tensor, torch.Tensor):
            batch_tensor = batch_tensor.detach().cpu()
        batch_size = batch_tensor.size(0)
        ncols = min(max_cols, batch_size)
        nrows = (batch_size + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        if title:
            fig.suptitle(title)
        axes = np.array(axes).reshape(nrows, ncols)
        for i in range(batch_size):
            row, col = divmod(i, ncols)
            ax = axes[row, col]
            img = batch_tensor[i]
            if img.ndim == 3:
                if img.shape[0] == 1:
                    img = img.squeeze(0)
                    img = np.stack([img.numpy()]*3, axis=-1)
                elif img.shape[0] == 3:
                    img = img.numpy().transpose(1, 2, 0)
                    img = img
                    img = np.clip(img, 0, 1)
                else:
                    img = img[0].numpy()
                    img = np.stack([img]*3, axis=-1)
            elif img.ndim == 2:
                img = np.stack([img.numpy()]*3, axis=-1)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Frame {i}')
        # Hide unused axes
        for i in range(batch_size, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row, col].axis('off')
        plt.tight_layout()
        plt.show()

    # Visualize a few batches
    print("\nVisualizing a few batches (close the window to continue)...")
    for i, batch in enumerate(dataloader):
        # batch shape: (dataloader_batch_size, batch_size, C, H, W)
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if batch.ndim == 5:
            # (dataloader_batch_size, batch_size, C, H, W)
            for j in range(batch.shape[0]):
                show_batch(batch[j], title=f'Batch {i}, Sample {j}')
        else:
            show_batch(batch, title=f'Batch {i}')
        if i >= 2:
            break
    print("Visualization done.")
    
    print("\nDone!")