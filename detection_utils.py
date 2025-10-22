"""
Detection utilities for GroundingDINO supervision of Pi3 detection head.
"""
import torch
import torch.nn.functional as F


def focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal Loss for addressing class imbalance in object detection.
    Uses binary_cross_entropy_with_logits for autocast safety.
    
    Args:
        pred: [N, num_classes] predicted class logits (NOT probabilities)
        target: [N, num_classes] target class probabilities (can be soft)
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum' or 'none'
    
    Returns:
        Focal loss value
    """
    # Reshape to handle any input dimensions
    original_shape = pred.shape
    pred_flat = pred.reshape(-1, original_shape[-1])  # [N, num_classes]
    target_flat = target.reshape(-1, original_shape[-1])  # [N, num_classes]
    
    # Compute focal loss for each class
    focal_losses = []
    for c in range(original_shape[-1]):
        # Get predictions and targets for this class
        pred_logits_c = pred_flat[:, c]  # [N] - logits
        target_c = target_flat[:, c]  # [N] - targets
        
        # Compute probabilities for focal weight calculation
        pred_prob_c = torch.sigmoid(pred_logits_c)
        
        # Compute binary cross entropy with logits (autocast-safe)
        bce = F.binary_cross_entropy_with_logits(pred_logits_c, target_c, reduction='none')
        
        # Compute focal weight: (1 - p_t)^gamma where p_t is model confidence on true class
        p_t = torch.where(target_c == 1, pred_prob_c, 1 - pred_prob_c)
        focal_weight = (1 - p_t) ** gamma
        
        # Apply alpha weighting
        alpha_t = torch.where(target_c == 1, alpha, 1 - alpha)
        
        # Compute focal loss
        focal_loss_c = alpha_t * focal_weight * bce
        focal_losses.append(focal_loss_c)
    
    # Combine losses across classes
    focal_loss_total = torch.stack(focal_losses, dim=1).sum(dim=1)  # [N]
    
    if reduction == 'mean':
        return focal_loss_total.mean()
    elif reduction == 'sum':
        return focal_loss_total.sum()
    else:
        return focal_loss_total.reshape(original_shape[:-1])  # Restore original shape minus last dim


def iou_loss(pred_boxes, target_boxes, loss_type='iou', eps=1e-6):
    """
    IoU-based loss for bounding box regression.
    
    Args:
        pred_boxes: [N, 4] predicted boxes (x, y, w, h)
        target_boxes: [N, 4] target boxes (x, y, w, h)
        loss_type: 'iou', 'giou', or 'diou'
        eps: Small constant for numerical stability
    
    Returns:
        IoU loss value
    """
    if pred_boxes.numel() == 0:
        return torch.tensor(0.0, device=pred_boxes.device)
    
    # Convert (x, y, w, h) to (x1, y1, x2, y2) format
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    
    target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
    
    # Compute intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    # Compute union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area + eps
    
    # Compute IoU
    iou = inter_area / union_area
    
    if loss_type == 'iou':
        return 1 - iou.mean()
    
    elif loss_type == 'giou':
        # Generalized IoU
        # Find enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + eps
        giou = iou - (enclose_area - union_area) / enclose_area
        return 1 - giou.mean()
    
    elif loss_type == 'diou':
        # Distance IoU
        # Center distances
        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        target_center_x = (target_x1 + target_x2) / 2
        target_center_y = (target_y1 + target_y2) / 2
        
        center_dist_sq = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
        
        # Diagonal of enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        diagonal_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps
        
        diou = iou - center_dist_sq / diagonal_sq
        return 1 - diou.mean()
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def generate_detection_targets(gdino_boxes, gdino_logits, gdino_phrases, image_shape, patch_shape, device, 
                             batch_size=1, num_frames=1, confidence_threshold=0.35):
    """
    Generate detection targets from GroundingDINO results for Pi3 detection head supervision.
    Uses IoU-based assignment to handle overlapping boxes properly.
    
    Args:
        gdino_boxes: [num_detections, 4] normalized bbox coordinates (x1, y1, x2, y2) 
                    OR list of [num_detections, 4] for batch processing
        gdino_logits: [num_detections] confidence scores
                     OR list of [num_detections] for batch processing  
        gdino_phrases: List of detection phrases 
                      OR list of lists for batch processing
        image_shape: (H, W) original image dimensions
        patch_shape: (patch_h, patch_w) patch grid dimensions
        device: torch device
        batch_size: Number of sequences in batch
        num_frames: Number of frames per sequence  
        confidence_threshold: Minimum confidence threshold for valid detections
        
    Returns:
        detection_targets: [B, N, patch_h, patch_w, num_classes + 4] detection targets
                          (classes + bbox)
    """
    H, W = image_shape
    patch_h, patch_w = patch_shape
    num_classes = 2  # traffic light, road sign
    
    # Initialize targets: [B, N, patch_h, patch_w, num_classes + 4]
    # First num_classes channels: classification probabilities
    # Next 4 channels: bbox regression (x, y, w, h)
    targets = torch.zeros(batch_size, num_frames, patch_h, patch_w, num_classes + 4, device=device)
    
    # Class mapping from phrases to indices
    class_mapping = {
        'traffic light': 0,
        'road sign': 1
    }
    
    # Handle single input (convert to batch format)
    if not isinstance(gdino_boxes, list):
        # Single image case: replicate across batch and frames
        gdino_boxes = [gdino_boxes] * batch_size * num_frames
        gdino_logits = [gdino_logits] * batch_size * num_frames  
        gdino_phrases = [gdino_phrases] * batch_size * num_frames
    elif len(gdino_boxes) == batch_size:
        # Batch case but single frame per batch: replicate across frames
        expanded_boxes = []
        expanded_logits = []
        expanded_phrases = []
        for b in range(batch_size):
            for n in range(num_frames):
                expanded_boxes.append(gdino_boxes[b])
                expanded_logits.append(gdino_logits[b])
                expanded_phrases.append(gdino_phrases[b])
        gdino_boxes = expanded_boxes
        gdino_logits = expanded_logits
        gdino_phrases = expanded_phrases
    
    # Process each batch and frame
    for b in range(batch_size):
        for n in range(num_frames):
            frame_idx = b * num_frames + n
            
            # Get data for this frame
            if frame_idx >= len(gdino_boxes):
                continue  # No more data available
                
            frame_boxes = gdino_boxes[frame_idx]
            frame_logits = gdino_logits[frame_idx] 
            frame_phrases = gdino_phrases[frame_idx]
            
            # Skip if no detections
            if len(frame_boxes) == 0:
                continue
            
            # Convert normalized boxes to patch coordinates
            patch_w_size = W / patch_w
            patch_h_size = H / patch_h
            
            for i, (box, logit, phrase) in enumerate(zip(frame_boxes, frame_logits, frame_phrases)):
                # Get class index
                class_idx = None
                for key, idx in class_mapping.items():
                    if key in phrase.lower():
                        class_idx = idx
                        break
                
                if class_idx is None or logit <= confidence_threshold:  # Skip unknown/low-confidence
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                x1, y1, x2, y2 = box
                x1_pix, y1_pix = x1 * W, y1 * H
                x2_pix, y2_pix = x2 * W, y2 * H
                box_center_x = (x1_pix + x2_pix) / 2
                box_center_y = (y1_pix + y2_pix) / 2
                box_w_pix = x2_pix - x1_pix
                box_h_pix = y2_pix - y1_pix
                
                # Find the single closest patch to assign this object to
                best_distance = float('inf')
                best_patch = None
                
                for py in range(patch_h):
                    for px in range(patch_w):
                        # Patch center coordinates
                        patch_center_x = (px + 0.5) * patch_w_size
                        patch_center_y = (py + 0.5) * patch_h_size
                        
                        # Distance from patch center to box center
                        dist_x = abs(box_center_x - patch_center_x) / patch_w_size
                        dist_y = abs(box_center_y - patch_center_y) / patch_h_size
                        distance = (dist_x**2 + dist_y**2)**0.5
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_patch = (px, py, patch_center_x, patch_center_y)
                
                # Assign object to the single best patch
                if best_patch is not None:
                    px, py, patch_center_x, patch_center_y = best_patch
                    
                    # Compute centerness score for quality weighting
                    left = max(0, box_center_x - x1_pix)
                    right = max(0, x2_pix - box_center_x)
                    top = max(0, box_center_y - y1_pix) 
                    bottom = max(0, y2_pix - box_center_y)
                    
                    if left > 0 and right > 0 and top > 0 and bottom > 0:
                        centerness = ((min(left, right) / max(left, right)) * 
                                     (min(top, bottom) / max(top, bottom)))**0.5
                    else:
                        centerness = max(0, 1.0 - best_distance * 2)  # Distance-based fallback
                    
                    # Weight by confidence and centerness
                    target_weight = logit * centerness
                    
                    # Assign target to this patch (no competition since it's single assignment)
                    targets[b, n, py, px, class_idx] = target_weight
                    
                    # Bbox regression targets (relative to patch center)
                    rel_x = (box_center_x - patch_center_x) / patch_w_size
                    rel_y = (box_center_y - patch_center_y) / patch_h_size
                    rel_w = box_w_pix / patch_w_size
                    rel_h = box_h_pix / patch_h_size
                    
                    targets[b, n, py, px, num_classes:num_classes+4] = torch.tensor(
                        [rel_x, rel_y, rel_w, rel_h], device=device
                    )
    
    return targets


def detection_loss(pred_detections, target_detections=None, gdino_boxes=None, gdino_logits=None, 
                  gdino_phrases=None, image_shape=None, detection_architecture='dense', pos_weight=2.0,
                  confidence_threshold=0.35):
    """
    Compute detection loss for Pi3 detection head using GroundingDINO supervision.
    Routes between dense grid loss and DETR loss based on architecture.
    
    Args:
        pred_detections: Predictions (format depends on architecture)
                        Dense: [B, N, patch_h, patch_w, num_classes + 4] 
                        DETR: Dict with 'class_logits', 'bbox_preds'
        target_detections: [B, N, patch_h, patch_w, num_classes + 4] targets (dense only)
        gdino_boxes: [num_detections, 4] GroundingDINO boxes (DETR only)
        gdino_logits: [num_detections] GroundingDINO confidences (DETR only)
        gdino_phrases: List of detection phrases (DETR only)
        image_shape: (H, W) image dimensions (DETR only)
        detection_architecture: 'dense' or 'detr'
        pos_weight: Weight for positive samples in classification loss (dense only)
        confidence_threshold: Minimum confidence threshold for valid detections
        
    Returns:
        total_loss: Combined detection loss
        cls_loss: Classification loss component  
        reg_loss: Regression loss component
    """
    if pred_detections is None:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    
    if detection_architecture == 'detr':
        # DETR-style loss with Hungarian matching
        from detr_losses import detr_detection_loss
        
        if gdino_boxes is None or len(gdino_boxes) == 0:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        
        total_loss, loss_dict = detr_detection_loss(
            pred_detections, gdino_boxes, gdino_logits, gdino_phrases, image_shape
        )
        
        # Extract individual components
        cls_loss = loss_dict.get('loss_ce', torch.tensor(0.0))
        reg_loss = loss_dict.get('loss_bbox', torch.tensor(0.0)) + loss_dict.get('loss_giou', torch.tensor(0.0))
        
        return total_loss, cls_loss, reg_loss
        
    else:
        # Dense grid loss with Focal Loss and IoU Loss
        if target_detections is None:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        
        # Validate shapes
        if not validate_detection_shapes(pred_detections, target_detections, verbose=False):
            print("❌ Detection shape validation failed! Returning zero loss.")
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        
        B, N, patch_h, patch_w, channels = pred_detections.shape
        num_classes = 2
        
        # Split predictions and targets into classification and regression
        pred_cls = pred_detections[..., :num_classes]  
        pred_reg = pred_detections[..., num_classes:]  
        target_cls = target_detections[..., :num_classes]  
        target_reg = target_detections[..., num_classes:]
        
        # Focal Loss for classification (addresses class imbalance)
        cls_loss = focal_loss(pred_cls, target_cls, alpha=0.25, gamma=2.0)
        
        # Positive sample mask based on classification targets
        positive_mask = (target_cls.sum(dim=-1) > 0)  # [B, N, patch_h, patch_w]
        
        # Regression loss - IoU Loss for better localization
        if positive_mask.sum() > 0:
            # Get positive samples
            pos_pred_reg = pred_reg[positive_mask]  # [num_pos, 4]
            pos_target_reg = target_reg[positive_mask]  # [num_pos, 4]
            
            # Convert relative coordinates to absolute boxes for IoU computation
            # This is approximate since we don't have exact patch positions
            reg_loss = iou_loss(pos_pred_reg, pos_target_reg, loss_type='giou')
        else:
            reg_loss = torch.tensor(0.0, device=pred_cls.device)
        
        # Combine losses with appropriate weighting
        total_loss = cls_loss + 2.0 * reg_loss
        
        return total_loss, cls_loss, reg_loss


def store_gdino_results_for_batch(gdino_results_list, batch_size, num_frames):
    """
    Store GroundingDINO results for a batch with proper indexing.
    
    Args:
        gdino_results_list: List to store results
        batch_size: Number of sequences in batch
        num_frames: Number of frames per sequence
        
    Returns:
        Dictionary structure for storing frame-wise GDINO results
    """
    batch_results = {}
    for b in range(batch_size):
        batch_results[b] = {}
        for t in range(num_frames):
            batch_results[b][t] = {
                'boxes': [],
                'logits': [],
                'phrases': [],
                'processed': False
            }
    return batch_results


def validate_detection_shapes(pred_detections, target_detections, verbose=False):
    """
    Validate that prediction and target shapes are compatible for dense detection loss.
    
    Args:
        pred_detections: [B, N, patch_h, patch_w, num_classes + 4] predictions
        target_detections: [B, N, patch_h, patch_w, num_classes + 4] targets
        verbose: Whether to print detailed shape information
        
    Returns:
        bool: True if shapes are compatible, False otherwise
    """
    if pred_detections is None or target_detections is None:
        if verbose:
            print("⚠️  One of pred_detections or target_detections is None")
        return False
    
    pred_shape = pred_detections.shape
    target_shape = target_detections.shape
    
    if pred_shape != target_shape:
        if verbose:
            print(f"❌ Shape mismatch:")
            print(f"   Predictions: {pred_shape}")
            print(f"   Targets:     {target_shape}")
        return False
    
    if len(pred_shape) != 5:
        if verbose:
            print(f"❌ Expected 5D tensor [B, N, patch_h, patch_w, channels], got {len(pred_shape)}D")
        return False
    
    B, N, patch_h, patch_w, channels = pred_shape
    expected_channels = 6  # 2 classes + 4 bbox coords
    
    if channels != expected_channels:
        if verbose:
            print(f"❌ Expected {expected_channels} channels (2 classes + 4 bbox), got {channels}")
        return False
    
    if verbose:
        print(f"✅ Shape validation passed:")
        print(f"   Shape: [B={B}, N={N}, patch_h={patch_h}, patch_w={patch_w}, channels={channels}]")
        print(f"   Total spatial locations: {B * N * patch_h * patch_w}")
        
        # Check for non-zero targets
        positive_targets = (target_detections[..., :2].sum(dim=-1) > 0).sum().item()
        print(f"   Positive target locations: {positive_targets}")
    
    return True