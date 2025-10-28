#!/usr/bin/env python3
"""
Image preprocessing utilities for model inference and training.

This module provides standalone preprocessing functions that can be used
without importing from deep nested paths.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as TF
from typing import Tuple, Union, Optional


def preprocess_image(img_tensor, mode="crop", target_size=528, keep_ratio=False, patch_size=16):
    """
    Preprocess image tensor(s) to target size with crop or pad mode.
    
    Args:
        img_tensor (torch.Tensor): Image tensor of shape (C, H, W) or (T, C, H, W), values in [0, 1]
        mode (str): 'crop' or 'pad'
        target_size (int): Target size for width/height
        keep_ratio (bool): If True and mode='crop', maintain full height without cropping
        patch_size (int): Ensure dimensions are divisible by patch_size
    
    Returns:
        torch.Tensor: Preprocessed image tensor(s), same batch dim as input
    """
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")
    
    if img_tensor.dim() == 3:
        tensors = [img_tensor]
        squeeze = True
    elif img_tensor.dim() == 4:
        tensors = list(img_tensor)
        squeeze = False
    else:
        raise ValueError("Input tensor must be (C, H, W) or (T, C, H, W)")
    
    processed = []
    
    for img in tensors:
        C, H, W = img.shape
        
        if mode == "pad":
            # Make the largest dimension target_size while maintaining aspect ratio
            if W >= H:
                new_W = target_size
                new_H = round(H * (new_W / W) / patch_size) * patch_size
            else:
                new_H = target_size
                new_W = round(W * (new_H / H) / patch_size) * patch_size
            
            # Resize
            out = torch.nn.functional.interpolate(
                img.unsqueeze(0), 
                size=(new_H, new_W), 
                mode="bicubic", 
                align_corners=False
            ).squeeze(0)
            
            # Pad to square
            h_padding = target_size - new_H
            w_padding = target_size - new_W
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            
            if h_padding > 0 or w_padding > 0:
                out = torch.nn.functional.pad(
                    out, 
                    (pad_left, pad_right, pad_top, pad_bottom), 
                    mode="constant", 
                    value=1.0
                )
        else:  # crop mode
            # Set width to target_size and maintain aspect ratio
            new_W = target_size
            new_H = round(H * (new_W / W) / patch_size) * patch_size
            
            # Resize
            out = torch.nn.functional.interpolate(
                img.unsqueeze(0), 
                size=(new_H, new_W), 
                mode="bicubic", 
                align_corners=False
            ).squeeze(0)
            
            # Center crop if height exceeds target_size (unless keep_ratio is True)
            if not keep_ratio and new_H > target_size:
                start_y = (new_H - target_size) // 2
                out = out[:, start_y : start_y + target_size, :]
        
        processed.append(out)
    
    result = torch.stack(processed)
    
    if squeeze:
        return result[0]
    return result