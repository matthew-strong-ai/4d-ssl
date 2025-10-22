#!/usr/bin/env python3
"""Debug script to check if autoregressive transformer is being used."""

import torch
from utils.model_factory import create_model
from config.defaults import get_cfg_defaults

# Load config
cfg = get_cfg_defaults()
cfg.MODEL.ARCHITECTURE = "AutoregressivePi3"
cfg.MODEL.FREEZE_DECODERS = True

# Create model
model = create_model(cfg)
model.train()  # Important: set to train mode

# Create dummy input
dummy_input = torch.randn(1, 3, 3, 518, 924)  # B=1, N=3, H=518, W=924

# Add hook to token_predictor to verify it's being called
call_count = 0
def hook_fn(module, input, output):
    global call_count
    call_count += 1
    print(f"\n🔍 Token Predictor called (call #{call_count})!")
    print(f"   Input shape: {input[0].shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output requires_grad: {output.requires_grad}")

# Register hook
handle = model.autoregressive_transformer.token_predictor.register_forward_hook(hook_fn)

# Forward pass
print("Running forward pass...")
output = model(dummy_input)

print(f"\n✅ Token predictor was called {call_count} times")

# Check if we have future frames in output
if 'local_feats' in output:
    B, N, H, W, C = output['local_feats'].shape
    print(f"Output shape: B={B}, N={N}, H={H}, W={W}, C={C}")
    print(f"Expected N=6 (3 input + 3 future), got N={N}")

# Clean up
handle.remove()