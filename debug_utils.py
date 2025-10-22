#!/usr/bin/env python3
"""
Debug utilities for training and model validation.

This module contains debugging functions for checking tensor values,
model parameters, and other validation utilities used across training scripts.
"""

import torch


def check_for_nans(value, name, step=None):
    """
    Check tensor or scalar value for NaN values and print detailed diagnostics.
    
    Args:
        value: Tensor, float, or int to check
        name: Name for logging
        step: Optional step number for context
    
    Returns:
        bool: True if NaNs found, False otherwise
    """
    import math
    
    step_info = f" at step {step}" if step is not None else ""
    
    # Handle scalar values (float, int)
    if isinstance(value, (float, int)):
        if math.isnan(value):
            print(f"❌ NaN detected in {name}{step_info}")
            print(f"   Value: {value}")
            return True
        elif math.isinf(value):
            print(f"⚠️ Inf detected in {name}{step_info}")
            print(f"   Value: {value}")
            return True
        return False
    
    # Handle tensors
    if torch.isnan(value).any():
        print(f"❌ NaN detected in {name}{step_info}")
        print(f"   Shape: {value.shape}")
        print(f"   Min: {value.min().item()}, Max: {value.max().item()}")
        print(f"   NaN count: {torch.isnan(value).sum().item()}")
        if value.numel() < 100:  # Only print small tensors
            print(f"   Values: {value}")
        return True
    
    # Also check for inf values
    if torch.isinf(value).any():
        print(f"⚠️ Inf detected in {name}{step_info}")
        print(f"   Shape: {value.shape}")
        print(f"   Min: {value.min().item()}, Max: {value.max().item()}")
        print(f"   Inf count: {torch.isinf(value).sum().item()}")
        return True
    
    return False


def check_model_parameters(model, name, step=None):
    """Check model parameters for NaN/Inf values."""
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            if check_for_nans(param.grad, f"{name}.{param_name}.grad", step):
                return True
        if check_for_nans(param, f"{name}.{param_name}", step):
            return True
    return False


def check_gradients(model, name="model", step=None, max_norm_threshold=10.0):
    """
    Check model gradients for issues and compute gradient statistics.
    
    Args:
        model: PyTorch model to check
        name: Name for logging
        step: Optional step number for context
        max_norm_threshold: Threshold for detecting large gradients
    
    Returns:
        dict: Gradient statistics
    """
    grad_stats = {
        'has_nans': False,
        'has_infs': False,
        'total_norm': 0.0,
        'max_grad': 0.0,
        'min_grad': 0.0,
        'num_params_with_grads': 0
    }
    
    total_norm = 0.0
    max_grad = float('-inf')
    min_grad = float('inf')
    num_params = 0
    
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            num_params += 1
            grad = param.grad
            
            # Check for NaN/Inf
            if torch.isnan(grad).any():
                grad_stats['has_nans'] = True
                print(f"❌ NaN in gradient: {name}.{param_name}")
            
            if torch.isinf(grad).any():
                grad_stats['has_infs'] = True
                print(f"⚠️ Inf in gradient: {name}.{param_name}")
            
            # Compute statistics
            param_norm = grad.norm().item()
            total_norm += param_norm ** 2
            
            grad_max = grad.max().item()
            grad_min = grad.min().item()
            
            max_grad = max(max_grad, grad_max)
            min_grad = min(min_grad, grad_min)
    
    total_norm = total_norm ** 0.5
    
    grad_stats.update({
        'total_norm': total_norm,
        'max_grad': max_grad if num_params > 0 else 0.0,
        'min_grad': min_grad if num_params > 0 else 0.0,
        'num_params_with_grads': num_params
    })
    
    # Warning for large gradients
    if total_norm > max_norm_threshold:
        step_info = f" at step {step}" if step is not None else ""
        print(f"⚠️ Large gradient norm in {name}{step_info}: {total_norm:.3f}")
    
    return grad_stats


def log_tensor_stats(tensor, name, step=None):
    """
    Log comprehensive tensor statistics.
    
    Args:
        tensor: Tensor to analyze
        name: Name for logging
        step: Optional step number for context
    """
    step_info = f" at step {step}" if step is not None else ""
    
    print(f"📊 {name}{step_info}:")
    print(f"   Shape: {tensor.shape}")
    print(f"   Dtype: {tensor.dtype}")
    print(f"   Device: {tensor.device}")
    print(f"   Min: {tensor.min().item():.6f}")
    print(f"   Max: {tensor.max().item():.6f}")
    print(f"   Mean: {tensor.mean().item():.6f}")
    print(f"   Std: {tensor.std().item():.6f}")
    print(f"   Norm: {tensor.norm().item():.6f}")
    
    # Check for special values
    num_nans = torch.isnan(tensor).sum().item()
    num_infs = torch.isinf(tensor).sum().item()
    num_zeros = (tensor == 0).sum().item()
    
    if num_nans > 0:
        print(f"   ❌ NaNs: {num_nans}")
    if num_infs > 0:
        print(f"   ⚠️ Infs: {num_infs}")
    if num_zeros > 0:
        print(f"   🔍 Zeros: {num_zeros}")


def validate_training_batch(batch, name="batch", step=None):
    """
    Validate a training batch for common issues.
    
    Args:
        batch: Batch data (can be tensor, dict, or list)
        name: Name for logging
        step: Optional step number for context
    
    Returns:
        bool: True if batch is valid, False if issues found
    """
    def check_tensor(tensor, tensor_name):
        if not isinstance(tensor, torch.Tensor):
            return True
            
        if check_for_nans(tensor, tensor_name, step):
            return False
            
        # Check for reasonable value ranges
        if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
            abs_max = tensor.abs().max().item()
            if abs_max > 1e6:
                print(f"⚠️ Large values in {tensor_name}: max abs = {abs_max}")
                
        return True
    
    is_valid = True
    
    if isinstance(batch, torch.Tensor):
        is_valid &= check_tensor(batch, name)
    elif isinstance(batch, dict):
        for key, value in batch.items():
            is_valid &= validate_training_batch(value, f"{name}.{key}", step)
    elif isinstance(batch, (list, tuple)):
        for i, item in enumerate(batch):
            is_valid &= validate_training_batch(item, f"{name}[{i}]", step)
    
    return is_valid