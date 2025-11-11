"""
PPGeo dataset for frame triplets (prev, curr, next) from YouTube S3 data.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np
from utils.youtube_s3_dataset import YouTubeS3Dataset
from typing import Tuple, Dict


class PPGeoDataset(Dataset):
    """
    Dataset that provides frame triplets for PPGeo self-supervised training.
    Wraps the existing YouTubeS3Dataset but returns consecutive frame triplets.
    """
    
    def __init__(
        self,
        root_prefix: str,
        cache_dir: str,
        img_size: Tuple[int, int] = (160, 320),
        is_train: bool = True,
        frame_sampling_rate: int = 1,
        max_samples: int = -1,
        augment: bool = True
    ):
        super().__init__()
        
        self.img_size = img_size
        self.is_train = is_train
        self.augment = augment and is_train
        
        # Use existing YouTube dataset to get video sequences
        self.base_dataset = YouTubeS3Dataset(
            root_prefix=root_prefix,
            cache_dir=cache_dir,
            sequence_length=3,  # We need at least 3 consecutive frames
            img_size=img_size,
            is_train=is_train,
            frame_sampling_rate=frame_sampling_rate,
            max_samples=max_samples
        )
        
        # Setup transforms
        self.setup_transforms()
        
    def setup_transforms(self):
        """Setup image transformations."""
        # Base transforms
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Crop to remove car hood (PPGeo style)
        self.crop_transform = T.Lambda(lambda x: x.crop((0, 10, self.img_size[1], self.img_size[0] + 10)))
        
        # Resize to target size
        self.resize = T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR)
        
        # Color augmentations (only for training)
        if self.augment:
            self.color_aug = T.Compose([
                T.RandomApply([
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
            ])
        else:
            self.color_aug = T.Lambda(lambda x: x)  # Identity transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def preprocess_image(self, image: Image.Image, apply_aug: bool = False) -> torch.Tensor:
        """Preprocess a single image."""
        # Crop to remove car hood
        image = self.crop_transform(image)
        
        # Resize
        image = self.resize(image)
        
        # Apply color augmentation if requested
        if apply_aug and self.augment:
            image = self.color_aug(image)
        
        # Convert to tensor and normalize
        image = self.to_tensor(image)
        # Note: Skip normalization for PPGeo as it expects [0,1] range
        
        return image
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get frame triplet for PPGeo training.
        
        Returns:
            Dictionary with:
            - 'images': [3, 3, H, W] tensor (prev, curr, next frames)
            - 'idx': sample index
        """
        # Get sequence from base dataset
        sequence_data = self.base_dataset[idx]
        images = sequence_data['images']  # [T, 3, H, W] where T >= 3
        
        # Extract first 3 frames as triplet
        img_tensors = []
        
        for i in range(3):
            # Convert tensor back to PIL for consistent preprocessing
            img_tensor = images[i]  # [3, H, W]
            img_pil = T.ToPILImage()(img_tensor)
            
            # Preprocess with augmentation for training
            processed_img = self.preprocess_image(img_pil, apply_aug=self.is_train)
            img_tensors.append(processed_img)
        
        # Stack into [3, 3, H, W] format
        frame_triplet = torch.stack(img_tensors, dim=0)
        
        return {
            'images': frame_triplet,  # [3, 3, H, W] - (prev, curr, next)
            'idx': idx
        }


class PPGeoDatasetSimple(Dataset):
    """
    Simplified PPGeo dataset that directly loads frame triplets from local files.
    Use this if you have a local copy of the YouTube data.
    """
    
    def __init__(
        self,
        data_root: str,
        meta_file: str,
        img_size: Tuple[int, int] = (160, 320),
        is_train: bool = True
    ):
        super().__init__()
        
        self.data_root = data_root
        self.img_size = img_size
        self.is_train = is_train
        
        # Load metadata (should contain triplet paths)
        self.load_metadata(meta_file)
        
        # Setup transforms
        self.setup_transforms()
    
    def load_metadata(self, meta_file: str):
        """Load frame triplet metadata."""
        # This should be implemented based on your metadata format
        # For now, create dummy data
        self.triplets = []
        
        # You would implement this based on your actual metadata format
        # Example format:
        # self.triplets = [
        #     {
        #         'prev': 'path/to/prev_frame.jpg',
        #         'curr': 'path/to/curr_frame.jpg', 
        #         'next': 'path/to/next_frame.jpg'
        #     },
        #     ...
        # ]
        
        print(f"⚠️ PPGeoDatasetSimple.load_metadata() needs implementation for {meta_file}")
    
    def setup_transforms(self):
        """Setup image transformations."""
        self.transform = T.Compose([
            T.Lambda(lambda x: x.crop((0, 10, 320, 170))),  # Remove car hood
            T.Resize(self.img_size),
            T.ToTensor()
        ])
        
        if self.is_train:
            self.color_aug = T.Compose([
                T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
            ])
        else:
            self.color_aug = T.Lambda(lambda x: x)
    
    def __len__(self):
        return len(self.triplets)
    
    def load_image(self, path: str) -> Image.Image:
        """Load image from path."""
        full_path = os.path.join(self.data_root, path)
        return Image.open(full_path).convert('RGB')
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get frame triplet."""
        triplet = self.triplets[idx]
        
        # Load images
        img_prev = self.load_image(triplet['prev'])
        img_curr = self.load_image(triplet['curr'])
        img_next = self.load_image(triplet['next'])
        
        # Apply transforms
        img_prev = self.transform(img_prev)
        img_curr = self.transform(img_curr)
        img_next = self.transform(img_next)
        
        # Apply augmentation to all frames consistently for training
        if self.is_train:
            # Apply same augmentation to all frames
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            img_prev = self.color_aug(T.ToPILImage()(img_prev))
            img_prev = T.ToTensor()(img_prev)
            
            torch.manual_seed(seed) 
            img_curr = self.color_aug(T.ToPILImage()(img_curr))
            img_curr = T.ToTensor()(img_curr)
            
            torch.manual_seed(seed)
            img_next = self.color_aug(T.ToPILImage()(img_next))
            img_next = T.ToTensor()(img_next)
        
        # Stack frames
        frame_triplet = torch.stack([img_prev, img_curr, img_next], dim=0)
        
        return {
            'images': frame_triplet,  # [3, 3, H, W]
            'idx': idx
        }