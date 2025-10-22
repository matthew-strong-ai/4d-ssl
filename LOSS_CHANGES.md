# Loss Function Changes - Future Frame Weighting

## Overview
Added support for controlling whether to include future frames in loss computation by using the `FUTURE_FRAME_WEIGHT` configuration parameter. Setting this to `0.0` will compute losses only on current frames, while positive values (e.g., `3.0`) will emphasize future frames.

## Changes Made

### 1. Configuration Files

#### `config.yaml` and `config_mapanything.yaml`
- Updated `FUTURE_FRAME_WEIGHT` documentation:
  ```yaml
  FUTURE_FRAME_WEIGHT: 3.0  # Weight multiplier for future frame supervision (>1.0 emphasizes future frames, 0.0 = current frames only)
  ```

### 2. Loss Functions in `losses.py`

#### A. Camera Pose Loss (`official_pi3_camera_pose_loss`)
- **Added parameters**: `m_frames=3, future_frame_weight=1.0`
- **Changes**:
  - Added future frame weighting logic for relative pose losses
  - Frame pairs involving future frames (where either frame index >= m_frames) get weighted by `future_frame_weight`
  - Updated to compute per-pair losses when weighting is needed
  - Modified `rot_ang_loss` to support `reduction='none'` parameter for per-sample losses

#### B. Confidence Loss (`confidence_loss`)
- **Added parameters**: `m_frames=3, future_frame_weight=1.0`
- **Changes**:
  - Added future frame weighting for BCE loss component
  - Added future frame weighting for gradient loss component
  - Properly handles temporal dimensions (4D tensors with shape [B, N, H, W])
  - When `future_frame_weight != 1.0`, applies frame-specific weights

#### C. Point Cloud Loss (`official_pi3_point_loss`)
- **Already supported**: This function already had future frame weighting implemented
- No changes were needed

#### D. Segmentation Loss (`segmentation_bce_loss`)
- **Added parameters**: `m_frames=3, future_frame_weight=1.0`
- **Changes**:
  - Added future frame weighting for both multi-class (cross-entropy) and binary (BCE) losses
  - For multi-class segmentation: applies weighting to focal loss and standard CE loss
  - For binary segmentation: applies weighting to BCE loss
  - When `future_frame_weight != 1.0`, creates per-frame weights and applies them element-wise

### 3. Function Call Updates

Updated all calls to the modified loss functions to pass the new parameters:
- `official_pi3_camera_pose_loss`: Now receives `m_frames` and `future_frame_weight`
- `confidence_loss`: Now receives `m_frames` and `future_frame_weight`
- `segmentation_bce_loss`: Now receives `m_frames` and `future_frame_weight`

## Usage

To compute losses only on current frames (excluding future frames):
```yaml
LOSS:
  FUTURE_FRAME_WEIGHT: 0.0  # Excludes future frames from loss computation
```

To emphasize future frames (default behavior):
```yaml
LOSS:
  FUTURE_FRAME_WEIGHT: 3.0  # Future frames get 3x weight in loss
```

To treat all frames equally:
```yaml
LOSS:
  FUTURE_FRAME_WEIGHT: 1.0  # All frames have equal weight
```

## Implementation Details

1. **Frame Indexing**: Frames 0 to m_frames-1 are considered "current frames", frames m_frames and beyond are "future frames"
2. **Weight Application**: When `future_frame_weight` is different from 1.0, per-element losses are computed and weighted before averaging
3. **Backward Compatibility**: Default values maintain existing behavior (future_frame_weight=1.0 for most functions)

## Files Modified
- `/home/matthew_strong/Desktop/autonomy-wild/config.yaml`
- `/home/matthew_strong/Desktop/autonomy-wild/config_mapanything.yaml`
- `/home/matthew_strong/Desktop/autonomy-wild/losses.py`