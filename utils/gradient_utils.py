"""Utilities for gradient flow verification and debugging."""

import torch


def check_gradient_flow_after_loss(model, loss):
    """
    Check gradient flow after backward pass.
    Call this after loss.backward() but before optimizer.step().
    
    Args:
        model: The model to check
        loss: The computed loss (after backward)
    """
    print("\n=== Gradient Flow Check ===")
    
    # Check autoregressive transformer specifically
    if hasattr(model, 'autoregressive_transformer'):
        ar_transformer = model.autoregressive_transformer
        
        print("Autoregressive Transformer components:")
        
        # Check temporal_pos_embed
        if hasattr(ar_transformer.temporal_pos_embed, 'temporal_pos_embed'):
            param = ar_transformer.temporal_pos_embed.temporal_pos_embed
            if param.grad is not None:
                print(f"  temporal_pos_embed: grad_norm={param.grad.norm().item():.6f}")
            else:
                print(f"  temporal_pos_embed: NO GRADIENT")
        
        # Check blocks
        for i, block in enumerate(ar_transformer.blocks):
            block_grads = []
            for name, param in block.named_parameters():
                if param.grad is not None:
                    block_grads.append(param.grad.norm().item())
            
            if block_grads:
                avg_grad = sum(block_grads) / len(block_grads)
                print(f"  block_{i}: avg_grad_norm={avg_grad:.6f}, num_params_with_grad={len(block_grads)}")
            else:
                print(f"  block_{i}: NO GRADIENTS")
        
        # Check norm
        if hasattr(ar_transformer, 'norm'):
            norm_grads = [p.grad.norm().item() for p in ar_transformer.norm.parameters() if p.grad is not None]
            if norm_grads:
                print(f"  norm: avg_grad_norm={sum(norm_grads)/len(norm_grads):.6f}")
            else:
                print(f"  norm: NO GRADIENTS")
        
        # Note: No token predictor - using transformer output directly
        print("\n  Using transformer output directly (no token predictor)")
    
    # Check decoder gradients (should be None if frozen)
    for name in ['point_decoder', 'conf_decoder', 'camera_decoder', 
                 'point_head', 'conf_head', 'camera_head']:
        if hasattr(model, name):
            module = getattr(model, name)
            has_grad = any(p.grad is not None for p in module.parameters())
            print(f"{name}: {'HAS GRADIENTS' if has_grad else 'NO GRADIENTS (frozen)'}")
    
    print("========================\n")


def visualize_gradient_magnitudes(model):
    """
    Visualize gradient magnitudes across different layers.
    Useful for identifying vanishing/exploding gradients.
    """
    gradient_info = {}
    
    if hasattr(model, 'autoregressive_transformer'):
        ar_transformer = model.autoregressive_transformer
        
        # Check each transformer block
        for i, block in enumerate(ar_transformer.blocks):
            block_grads = []
            for name, param in block.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    block_grads.append(grad_norm)
            
            if block_grads:
                gradient_info[f'ar_block_{i}'] = {
                    'mean': sum(block_grads) / len(block_grads),
                    'max': max(block_grads),
                    'min': min(block_grads)
                }
    
    return gradient_info