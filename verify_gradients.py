#!/usr/bin/env python3
"""Quick script to verify gradient flow in the model."""

import torch
from utils.model_factory import create_model
from config.defaults import get_cfg_defaults

# Load config
cfg = get_cfg_defaults()
cfg.MODEL.ARCHITECTURE = "AutoregressivePi3"
cfg.MODEL.FREEZE_DECODERS = True

# Create model
model = create_model(cfg)

# Call the verification method
if hasattr(model, 'verify_gradient_flow'):
    print("\n=== Model Gradient Status ===")
    model.verify_gradient_flow()

# Do a dummy forward/backward pass to check actual gradients
print("\n=== Testing Gradient Flow ===")
dummy_input = torch.randn(1, 3, 3, 518, 924)  # B=1, N=3, H=518, W=924
model.train()

# Forward pass
output = model(dummy_input)
dummy_loss = output['local_feats'].mean()  # Simple dummy loss

# Backward pass
dummy_loss.backward()

# Check actual gradients
print("\n=== Actual Gradient Check After Backward ===")
if hasattr(model, 'autoregressive_transformer'):
    ar_params_with_grad = 0
    ar_params_total = 0
    
    for name, param in model.autoregressive_transformer.named_parameters():
        ar_params_total += 1
        if param.grad is not None:
            ar_params_with_grad += 1
            print(f"✓ {name}: grad_norm = {param.grad.norm().item():.6f}")
        else:
            print(f"✗ {name}: NO GRADIENT")
    
    print(f"\nAutoregressive Transformer: {ar_params_with_grad}/{ar_params_total} parameters have gradients")

# Check frozen components
print("\n=== Checking Frozen Components ===")
for component_name in ['point_decoder', 'conf_decoder', 'camera_decoder',
                       'point_head', 'conf_head', 'camera_head']:
    if hasattr(model, component_name):
        component = getattr(model, component_name)
        has_any_grad = any(p.grad is not None for p in component.parameters())
        print(f"{component_name}: {'ERROR - HAS GRADIENTS!' if has_any_grad else '✓ Frozen (no gradients)'}")