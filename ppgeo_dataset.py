"""
PPGeo dataset for frame triplets (prev, curr, next) from YouTube S3 data.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms as T
import numpy as np
from utils.youtube_s3_dataset import YouTubeS3Dataset
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


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
            bucket_name="research-datasets",
            root_prefix=root_prefix,
            m=3,  # Input frames
            n=0,  # Target frames (we just need sequences of 3 frames)
            transform=None,  # We'll handle transforms ourselves
            cache_dir=cache_dir,
            frame_sampling_rate=frame_sampling_rate,
            min_sequence_length=50,
            skip_frames=300,  # Skip first 300 frames like in train_cluster.py
            max_workers=8
        )
        
        # Limit dataset size if requested
        if max_samples > 0 and len(self.base_dataset) > max_samples:
            from torch.utils.data import Subset
            import random
            indices = list(range(len(self.base_dataset)))
            random.shuffle(indices)
            indices = indices[:max_samples]
            self.base_dataset = Subset(self.base_dataset, indices)
            print(f"🎯 Limited PPGeo dataset to {max_samples} samples")
        
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
    
    def preprocess_ppgeo_style(self, inputs, color_aug):
        """Preprocess images in PPGeo style with multi-scale resizing."""
        height, width = 160, 320
        num_scales = 4
        interp = T.InterpolationMode.BICUBIC
        to_tensor = T.ToTensor()
        
        # Create resize transforms for each scale
        resize_transforms = {}
        for i in range(num_scales):
            s = 2 ** i
            resize_transforms[i] = T.Resize((height // s, width // s), interpolation=interp)
        
        # Resize color images to the required scales
        for k in list(inputs.keys()):
            if "color" in k and k[2] == -1:  # Base scale images
                frame_id = k[1]
                base_image = inputs[k]
                
                # Create images at all scales
                for i in range(num_scales):
                    inputs[("color", frame_id, i)] = resize_transforms[i](base_image)
        
        # Convert to tensors and apply augmentation
        for k in list(inputs.keys()):
            if "color" in k and k[2] >= 0:  # Scaled images
                frame_id, scale = k[1], k[2]
                image = inputs[k]
                
                # Convert to tensor
                inputs[("color", frame_id, scale)] = to_tensor(image)
                
                # Apply augmentation and convert to tensor
                inputs[("color_aug", frame_id, scale)] = to_tensor(color_aug(image))
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get frame triplet for PPGeo training in the original PPGeo format.
        
        Returns:
            Dictionary with PPGeo-style keys:
            - ("color", frame_id, scale): PIL image for frame_id at scale
            - ("color_aug", frame_id, scale): Augmented PIL image for frame_id at scale
        """
        # Get sequence from base dataset
        cur_frames, future_frames, metadata = self.base_dataset[idx]
        images = cur_frames  # [T, 3, H, W] where T >= 3
        
        # PPGeo constants
        frame_ids = [-1, 0, 1]  # prev, curr, next
        num_scales = len([0, 1, 2, 3])
        
        inputs = {}
        
        # Convert frames back to PIL images and apply initial preprocessing
        pil_images = {}
        for i, frame_id in enumerate(frame_ids):
            # Convert tensor back to PIL for consistent preprocessing
            img_tensor = images[i]  # [3, H, W]
            img_pil = T.ToPILImage()(img_tensor)
            
            # Just resize to target size without cropping
            img_pil = img_pil.resize((320, 160), Image.BILINEAR)
                
            pil_images[frame_id] = img_pil
            
            # Store base image at scale -1 (will be processed later)
            inputs[("color", frame_id, -1)] = img_pil
        
        # Setup color augmentation (matching original PPGeo exactly)
        if self.is_train:
            color_aug = T.Compose([
                T.RandomApply([
                    T.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), 
                                saturation=(0.8, 1.2), hue=(-0.1, 0.1))
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur([.1, 2.])], p=0.5)
            ])
        else:
            color_aug = lambda x: x
        

        # Process for all scales like original PPGeo
        self.preprocess_ppgeo_style(inputs, color_aug)

        # Remove the scale -1 images (original PPGeo does this)
        for frame_id in frame_ids:
            del inputs[("color", frame_id, -1)]
            if ("color_aug", frame_id, -1) in inputs:
                del inputs[("color_aug", frame_id, -1)]
        
        return inputs


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