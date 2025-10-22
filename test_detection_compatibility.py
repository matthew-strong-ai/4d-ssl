#!/usr/bin/env python3
"""
Test script to verify detection_loss and generate_detection_targets compatibility.
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detection_utils import generate_detection_targets, detection_loss, validate_detection_shapes

def test_detection_compatibility():
    """Test that target generation and loss computation are compatible."""
    print("🔍 Testing Detection Loss and Target Generation Compatibility\n")
    
    # Test parameters
    batch_size = 2
    num_frames = 3
    H, W = 224, 224
    patch_h, patch_w = H // 14, W // 14  # 16x16 patches for 224x224 image
    num_classes = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Test Setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Frames: {num_frames}")
    print(f"  Image size: {H}x{W}")
    print(f"  Patch grid: {patch_h}x{patch_w}")
    print(f"  Device: {device}")
    print()
    
    # Mock GroundingDINO results
    print("📦 Creating mock GroundingDINO results...")
    
    # Example detection: traffic light at (0.3, 0.2, 0.4, 0.35)
    mock_boxes = torch.tensor([
        [0.3, 0.2, 0.4, 0.35],  # traffic light
        [0.6, 0.7, 0.8, 0.9]    # road sign
    ])
    mock_logits = torch.tensor([0.8, 0.6])
    mock_phrases = ['traffic light', 'road sign']
    
    # Replicate for batch processing (same detections across batch/frames for simplicity)
    gdino_boxes_batch = [mock_boxes] * (batch_size * num_frames)
    gdino_logits_batch = [mock_logits] * (batch_size * num_frames)
    gdino_phrases_batch = [mock_phrases] * (batch_size * num_frames)
    
    print(f"  Mock boxes shape: {mock_boxes.shape}")
    print(f"  Mock logits: {mock_logits}")
    print(f"  Mock phrases: {mock_phrases}")
    print()
    
    # Generate detection targets
    print("🎯 Generating detection targets...")
    targets = generate_detection_targets(
        gdino_boxes_batch,
        gdino_logits_batch, 
        gdino_phrases_batch,
        image_shape=(H, W),
        patch_shape=(patch_h, patch_w),
        device=device,
        batch_size=batch_size,
        num_frames=num_frames,
        confidence_threshold=0.35
    )
    
    print(f"  Target shape: {targets.shape}")
    print(f"  Expected shape: [{batch_size}, {num_frames}, {patch_h}, {patch_w}, {num_classes + 4}]")
    
    # Check for positive targets
    positive_locs = (targets[..., :num_classes].sum(dim=-1) > 0).sum().item()
    print(f"  Positive target locations: {positive_locs}")
    print()
    
    # Create mock predictions with same shape as targets
    print("🤖 Creating mock predictions...")
    predictions = torch.randn_like(targets) * 0.1  # Small random values
    
    # Add some positive predictions at target locations to test regression loss
    pos_mask = targets[..., :num_classes].sum(dim=-1) > 0
    predictions[pos_mask, :num_classes] = torch.randn(pos_mask.sum(), num_classes) * 2  # Higher class scores
    
    print(f"  Prediction shape: {predictions.shape}")
    print()
    
    # Validate shapes
    print("✅ Validating shape compatibility...")
    is_valid = validate_detection_shapes(predictions, targets, verbose=True)
    print()
    
    if not is_valid:
        print("❌ Shape validation failed!")
        return False
    
    # Compute detection loss
    print("📊 Computing detection loss...")
    try:
        total_loss, cls_loss, reg_loss = detection_loss(
            pred_detections=predictions,
            target_detections=targets,
            detection_architecture='dense',
            confidence_threshold=0.35
        )
        
        print(f"  Total loss: {total_loss.item():.6f}")
        print(f"  Classification loss: {cls_loss.item():.6f}")
        print(f"  Regression loss: {reg_loss.item():.6f}")
        print()
        
        # Check that losses are finite and reasonable
        if torch.isfinite(total_loss) and torch.isfinite(cls_loss) and torch.isfinite(reg_loss):
            print("✅ All losses are finite!")
        else:
            print("❌ Some losses are not finite!")
            return False
            
        if total_loss > 0:
            print("✅ Total loss is positive (expected for random predictions)")
        else:
            print("⚠️  Total loss is zero or negative")
        
        return True
        
    except Exception as e:
        print(f"❌ Error computing detection loss: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases like empty detections."""
    print("\n🔬 Testing edge cases...\n")
    
    device = 'cpu'
    batch_size, num_frames = 1, 1
    H, W = 224, 224
    patch_h, patch_w = H // 14, W // 14
    
    # Test empty detections
    print("📭 Testing empty detections...")
    empty_targets = generate_detection_targets(
        gdino_boxes=[torch.empty(0, 4)],
        gdino_logits=[torch.empty(0)],
        gdino_phrases=[[]],
        image_shape=(H, W),
        patch_shape=(patch_h, patch_w),
        device=device,
        batch_size=batch_size,
        num_frames=num_frames
    )
    
    print(f"  Empty targets shape: {empty_targets.shape}")
    print(f"  Non-zero targets: {(empty_targets > 0).sum().item()}")
    
    # Test loss with empty targets
    empty_preds = torch.randn_like(empty_targets) * 0.1
    total_loss, cls_loss, reg_loss = detection_loss(
        pred_detections=empty_preds,
        target_detections=empty_targets,
        detection_architecture='dense'
    )
    
    print(f"  Loss with empty targets - Total: {total_loss.item():.6f}, Cls: {cls_loss.item():.6f}, Reg: {reg_loss.item():.6f}")
    print("✅ Empty detection case handled successfully")
    
    return True

if __name__ == "__main__":
    success = True
    
    try:
        success &= test_detection_compatibility()
        success &= test_edge_cases()
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    if success:
        print("\n🎉 All tests passed! Detection loss and target generation are compatible.")
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        sys.exit(1)