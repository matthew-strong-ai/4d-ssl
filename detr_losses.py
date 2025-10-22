"""
DETR-style losses with Hungarian matching for Pi3 detection head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import List, Dict, Tuple


class HungarianMatcher:
    """
    Hungarian matching for DETR-style detection.
    Computes optimal bipartite matching between predictions and ground truth.
    """
    
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        """
        Args:
            cost_class: Weight for classification cost
            cost_bbox: Weight for L1 bounding box cost
            cost_giou: Weight for GIoU cost
        """
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox 
        self.cost_giou = cost_giou
        
    def forward(self, outputs: Dict, targets: List[Dict]) -> List[Tuple]:
        """
        Compute optimal matching between predictions and targets.
        
        Args:
            outputs: Dict with:
                'class_logits': [B, num_queries, num_classes + 1] 
                'bbox_preds': [B, num_queries, 4]
            targets: List of B target dicts, each with:
                'labels': [num_targets] class indices
                'boxes': [num_targets, 4] normalized bbox coordinates
                
        Returns:
            List of (pred_indices, target_indices) for each batch item
        """
        B, num_queries = outputs['class_logits'].shape[:2]
        
        # Get predictions
        pred_class = outputs['class_logits'].flatten(0, 1).softmax(-1)  # [B*num_queries, num_classes+1]
        pred_bbox = outputs['bbox_preds'].flatten(0, 1)  # [B*num_queries, 4]
        
        # Compute costs for each batch item
        batch_indices = []
        for i, target in enumerate(targets):
            if len(target['labels']) == 0:
                # No targets, assign all queries to no-object
                batch_indices.append(([], []))
                continue
                
            # Classification cost (use negative log probability)
            tgt_ids = target['labels']  # [num_targets]
            cost_class = -pred_class[i*num_queries:(i+1)*num_queries, tgt_ids]  # [num_queries, num_targets]
            
            # L1 bounding box cost
            tgt_bbox = target['boxes']  # [num_targets, 4]
            cost_bbox = torch.cdist(pred_bbox[i*num_queries:(i+1)*num_queries], tgt_bbox, p=1)  # [num_queries, num_targets]
            
            # GIoU cost
            cost_giou = -self._generalized_box_iou(
                pred_bbox[i*num_queries:(i+1)*num_queries], tgt_bbox
            )  # [num_queries, num_targets]
            
            # Combined cost matrix
            cost_matrix = (self.cost_class * cost_class + 
                          self.cost_bbox * cost_bbox + 
                          self.cost_giou * cost_giou)
            
            # Hungarian algorithm
            cost_matrix_np = cost_matrix.detach().cpu().numpy()
            pred_indices, tgt_indices = linear_sum_assignment(cost_matrix_np)
            
            batch_indices.append((pred_indices, tgt_indices))
            
        return batch_indices
    
    def _generalized_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Generalized IoU from https://giou.stanford.edu/
        
        Args:
            boxes1: [N, 4] in (x, y, w, h) format
            boxes2: [M, 4] in (x, y, w, h) format
            
        Returns:
            GIoU matrix [N, M]
        """
        # Convert to (x1, y1, x2, y2) format
        boxes1_xyxy = torch.cat([
            boxes1[:, :2] - boxes1[:, 2:] / 2,  # x1, y1
            boxes1[:, :2] + boxes1[:, 2:] / 2   # x2, y2
        ], dim=-1)
        
        boxes2_xyxy = torch.cat([
            boxes2[:, :2] - boxes2[:, 2:] / 2,  # x1, y1  
            boxes2[:, :2] + boxes2[:, 2:] / 2   # x2, y2
        ], dim=-1)
        
        return self._generalized_box_iou_xyxy(boxes1_xyxy, boxes2_xyxy)
    
    def _generalized_box_iou_xyxy(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """GIoU for boxes in (x1, y1, x2, y2) format"""
        # Intersection area
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
        
        # Union area
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
        union = area1[:, None] + area2 - inter  # [N, M]
        
        # IoU
        iou = inter / union
        
        # Enclosing box
        lt_enc = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb_enc = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
        wh_enc = (rb_enc - lt_enc).clamp(min=0)  # [N, M, 2]
        enclose = wh_enc[:, :, 0] * wh_enc[:, :, 1]  # [N, M]
        
        # GIoU
        giou = iou - (enclose - union) / enclose
        return giou


class DETRLoss:
    """
    DETR loss combining classification and bounding box losses.
    """
    
    def __init__(self, num_classes: int, matcher: HungarianMatcher, 
                 weight_dict: Dict[str, float] = None):
        """
        Args:
            num_classes: Number of object classes (without background)
            matcher: Hungarian matcher
            weight_dict: Weights for different loss components
        """
        self.num_classes = num_classes
        self.matcher = matcher
        
        if weight_dict is None:
            weight_dict = {
                'loss_ce': 1.0,      # Classification loss
                'loss_bbox': 5.0,    # L1 bbox loss
                'loss_giou': 2.0     # GIoU loss
            }
        self.weight_dict = weight_dict
        
        # Calculate class weights (higher weight for positive classes)
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = 0.1  # Lower weight for "no object" class
        self.empty_weight = empty_weight

    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Compute DETR losses.
        
        Args:
            outputs: Model outputs with 'class_logits' and 'bbox_preds'
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        # Get matching indices
        indices = self.matcher.forward(outputs, targets)
        
        # Calculate losses
        losses = {}
        losses.update(self._loss_labels(outputs, targets, indices))
        losses.update(self._loss_boxes(outputs, targets, indices))
        
        return losses
    
    def _loss_labels(self, outputs: Dict, targets: List[Dict], indices: List[Tuple]) -> Dict[str, torch.Tensor]:
        """Classification loss"""
        src_logits = outputs['class_logits']  # [B, num_queries, num_classes+1]
        
        # Create target classes
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, 
                                   dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        # Cross entropy loss with class weights
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, 
                                 weight=self.empty_weight.to(src_logits.device))
        
        return {'loss_ce': loss_ce}
    
    def _loss_boxes(self, outputs: Dict, targets: List[Dict], indices: List[Tuple]) -> Dict[str, torch.Tensor]:
        """Bounding box losses (L1 + GIoU)"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['bbox_preds'][idx]  # [num_matched, 4]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / len(target_boxes) if len(target_boxes) > 0 else torch.tensor(0.0, device=src_boxes.device)
        
        # GIoU loss
        if len(target_boxes) > 0:
            giou_matrix = HungarianMatcher()._generalized_box_iou(src_boxes, target_boxes)
            loss_giou = 1 - torch.diag(giou_matrix)
            loss_giou = loss_giou.mean()
        else:
            loss_giou = torch.tensor(0.0, device=src_boxes.device)
        
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}
    
    def _get_src_permutation_idx(self, indices: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get permutation indices for source (predictions)"""
        batch_idx = torch.cat([torch.full_like(torch.tensor(src), i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([torch.tensor(src) for (src, _) in indices])
        return batch_idx, src_idx


def detr_detection_loss(pred_detections: Dict, gdino_boxes: torch.Tensor, gdino_logits: torch.Tensor, 
                       gdino_phrases: List[str], image_shape: Tuple[int, int],
                       num_classes: int = 2) -> Tuple[torch.Tensor, Dict]:
    """
    Compute DETR detection loss using GroundingDINO supervision.
    
    Args:
        pred_detections: Dict with 'class_logits' [B, N, num_queries, num_classes+1] 
                        and 'bbox_preds' [B, N, num_queries, 4]
        gdino_boxes: [num_detections, 4] normalized bbox coordinates
        gdino_logits: [num_detections] confidence scores  
        gdino_phrases: List of detection phrases
        image_shape: (H, W) image dimensions
        num_classes: Number of detection classes
        
    Returns:
        total_loss: Combined DETR loss
        loss_dict: Individual loss components
    """
    if pred_detections is None:
        return torch.tensor(0.0), {}
    
    B, N = pred_detections['class_logits'].shape[:2]
    
    # Class mapping
    class_mapping = {'traffic light': 0, 'road sign': 1}
    
    # Process targets for each batch and frame
    all_outputs = {'class_logits': [], 'bbox_preds': []}
    all_targets = []
    
    for b in range(B):
        for n in range(N):
            # Get predictions for this frame
            frame_class_logits = pred_detections['class_logits'][b, n]  # [num_queries, num_classes+1]
            frame_bbox_preds = pred_detections['bbox_preds'][b, n]      # [num_queries, 4]
            
            all_outputs['class_logits'].append(frame_class_logits)
            all_outputs['bbox_preds'].append(frame_bbox_preds)
            
            # Create targets from GroundingDINO results
            target_labels = []
            target_boxes = []
            
            for i, (box, logit, phrase) in enumerate(zip(gdino_boxes, gdino_logits, gdino_phrases)):
                # Map phrase to class index
                class_idx = None
                for key, idx in class_mapping.items():
                    if key in phrase.lower():
                        class_idx = idx
                        break
                
                if class_idx is not None and logit > 0.5:  # High confidence threshold
                    target_labels.append(class_idx)
                    # Box is already normalized
                    target_boxes.append(box.tolist())
            
            # Create target dict (ensure proper device placement)
            device = pred_detections['class_logits'].device
            target = {
                'labels': torch.tensor(target_labels, dtype=torch.long, device=device),
                'boxes': torch.tensor(target_boxes, dtype=torch.float32, device=device) if target_boxes else torch.empty(0, 4, device=device)
            }
            all_targets.append(target)
    
    # Stack outputs
    stacked_outputs = {
        'class_logits': torch.stack(all_outputs['class_logits']),  # [B*N, num_queries, num_classes+1]
        'bbox_preds': torch.stack(all_outputs['bbox_preds'])       # [B*N, num_queries, 4]
    }
    
    # Create DETR loss
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    criterion = DETRLoss(num_classes=num_classes, matcher=matcher)
    
    # Compute loss
    loss_dict = criterion.forward(stacked_outputs, all_targets)
    
    # Combine losses
    total_loss = sum(loss_dict[k] * criterion.weight_dict.get(k, 1.0) for k in loss_dict.keys())
    
    return total_loss, loss_dict