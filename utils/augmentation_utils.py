#!/usr/bin/env python3
"""
Data augmentation utilities for training.

This module contains functions for applying various augmentations to images
and other data during training to improve model robustness.
"""

import torch
import random
import torchvision.transforms.functional as TF


def apply_random_augmentations(images, training=True):
    """
    Apply random augmentations to images with different amounts per image.
    
    Args:
        images: Tensor of shape (T, C, H, W) where T is number of frames
        training: Whether to apply augmentations (only during training)
        
    Returns:
        Augmented images tensor of same shape
    """
    if not training:
        return images
        
    T, C, H, W = images.shape
    augmented_images = []
    
    for t in range(T):
        img = images[t]  # (C, H, W)
        
        # Random color jittering with different amounts per image
        brightness_factor = random.uniform(0.7, 1.3)
        contrast_factor = random.uniform(0.7, 1.3) 
        saturation_factor = random.uniform(0.7, 1.3)
        hue_factor = random.uniform(-0.1, 0.1)
        
        # Apply color jittering
        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)
        img = TF.adjust_saturation(img, saturation_factor)
        img = TF.adjust_hue(img, hue_factor)
        
        # Random Gaussian blur (20% chance)
        if random.random() < 0.2:
            sigma = random.uniform(0.1, 1.0)
            img = TF.gaussian_blur(img, kernel_size=3, sigma=sigma)
        
        # Random grayscale (10% chance)
        if random.random() < 0.1:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
        
        # Clamp values to valid range
        img = torch.clamp(img, 0.0, 1.0)
        
        augmented_images.append(img)
    
    return torch.stack(augmented_images, dim=0)


def apply_spatial_augmentations(images, intrinsics=None, training=True):
    """
    Apply spatial augmentations (rotation, scaling, cropping) to images.
    
    Args:
        images: Tensor of shape (T, C, H, W)
        intrinsics: Camera intrinsics tensor to adjust if spatial transforms are applied
        training: Whether to apply augmentations
        
    Returns:
        Tuple of (augmented_images, adjusted_intrinsics)
    """
    if not training:
        return images, intrinsics
        
    T, C, H, W = images.shape
    
    # Random horizontal flip (50% chance)
    if random.random() < 0.5:
        images = torch.flip(images, dims=[-1])  # Flip along width
        if intrinsics is not None:
            # Adjust intrinsics for horizontal flip
            intrinsics = intrinsics.clone()
            intrinsics[..., 0, 2] = W - 1 - intrinsics[..., 0, 2]  # cx = W - 1 - cx
    
    # Random rotation (small angles, ±5 degrees)
    if random.random() < 0.3:
        angle = random.uniform(-5, 5)
        images = TF.rotate(images, angle, fill=0)
        # Note: Intrinsics adjustment for rotation is more complex and omitted here
    
    # Random scaling (90-110%)
    if random.random() < 0.3:
        scale = random.uniform(0.9, 1.1)
        new_h, new_w = int(H * scale), int(W * scale)
        images = TF.resize(images, (new_h, new_w))
        
        # Center crop or pad to maintain original size
        if new_h > H or new_w > W:
            # Crop from center
            crop_h = min(new_h, H)
            crop_w = min(new_w, W)
            top = (new_h - crop_h) // 2
            left = (new_w - crop_w) // 2
            images = TF.crop(images, top, left, crop_h, crop_w)
        
        if images.shape[-2:] != (H, W):
            images = TF.resize(images, (H, W))
            
        if intrinsics is not None:
            # Adjust intrinsics for scaling
            intrinsics = intrinsics.clone()
            intrinsics[..., :2, :2] *= scale  # Scale fx, fy
            
    return images, intrinsics


def apply_temporal_augmentations(images, frame_skip_prob=0.1):
    """
    Apply temporal augmentations like frame dropping/skipping.
    
    Args:
        images: Tensor of shape (T, C, H, W)
        frame_skip_prob: Probability of skipping a frame
        
    Returns:
        Augmented images tensor (may have fewer frames)
    """
    T, C, H, W = images.shape
    
    if T <= 2:  # Don't skip frames if we have too few
        return images
    
    keep_frames = []
    for t in range(T):
        if random.random() > frame_skip_prob:
            keep_frames.append(t)
    
    # Ensure we keep at least 2 frames
    if len(keep_frames) < 2:
        keep_frames = [0, T-1]
    
    return images[keep_frames]


def apply_noise_augmentations(images, noise_std=0.01, training=True):
    """
    Apply noise augmentations to images.
    
    Args:
        images: Tensor of shape (T, C, H, W)
        noise_std: Standard deviation of Gaussian noise
        training: Whether to apply augmentations
        
    Returns:
        Augmented images tensor
    """
    if not training:
        return images
        
    # Add Gaussian noise (30% chance)
    if random.random() < 0.3:
        noise = torch.randn_like(images) * noise_std
        images = images + noise
        images = torch.clamp(images, 0.0, 1.0)
    
    return images


def apply_all_augmentations(images, intrinsics=None, training=True, config=None):
    """
    Apply all augmentations in sequence.
    
    Args:
        images: Tensor of shape (T, C, H, W)
        intrinsics: Optional camera intrinsics
        training: Whether to apply augmentations
        config: Configuration object with augmentation settings
        
    Returns:
        Tuple of (augmented_images, adjusted_intrinsics)
    """
    if not training:
        return images, intrinsics
    
    # Get augmentation settings from config if available
    if config and hasattr(config, 'AUGMENTATION'):
        aug_cfg = config.AUGMENTATION
        apply_color = aug_cfg.get('COLOR', True)
        apply_spatial = aug_cfg.get('SPATIAL', True) 
        apply_temporal = aug_cfg.get('TEMPORAL', False)
        apply_noise = aug_cfg.get('NOISE', True)
        noise_std = aug_cfg.get('NOISE_STD', 0.01)
    else:
        # Default settings
        apply_color = True
        apply_spatial = True
        apply_temporal = False
        apply_noise = True
        noise_std = 0.01
    
    # Apply augmentations in order
    if apply_color:
        images = apply_random_augmentations(images, training)
    
    if apply_spatial:
        images, intrinsics = apply_spatial_augmentations(images, intrinsics, training)
    
    if apply_temporal:
        images = apply_temporal_augmentations(images)
    
    if apply_noise:
        images = apply_noise_augmentations(images, noise_std, training)
    
    return images, intrinsics