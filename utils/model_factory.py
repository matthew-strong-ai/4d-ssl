#!/usr/bin/env python3
"""
Model factory for creating different model architectures.

This module provides a centralized way to create and configure different
model architectures like Pi3, MapAnything, etc.
"""

import torch
import sys
import os

# from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_model

# Add Pi3 to path
pi3_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Pi3")
if pi3_path not in sys.path:
    sys.path.append(pi3_path)

from pi3.models.pi3 import AutonomyPi3, AutoregressivePi3


def create_model(cfg, dinov3_local_path=None):
    """
    Create a model based on the configuration.
    
    Args:
        cfg: YACS configuration object
        dinov3_local_path: Path to DINOv3 weights (for Pi3 models)
        
    Returns:
        model: The created model instance
    """
    architecture = cfg.MODEL.ARCHITECTURE.lower()
    
    if architecture == "pi3" or "pi3" in architecture:
        return create_pi3_model(cfg, dinov3_local_path)
    elif architecture == "mapanything":
        return create_mapanything_model(cfg)
    else:
        raise ValueError(f"Unknown model architecture: {cfg.MODEL.ARCHITECTURE}")


def create_pi3_model(cfg, dinov3_local_path=None):
    """
    Create a Pi3 model with the given configuration.
    
    Args:
        cfg: YACS configuration object
        dinov3_local_path: Path to DINOv3 weights
        
    Returns:
        AutonomyPi3 or AutoregressivePi3: Configured Pi3 model
    """
    # Check if we should use AutoregressivePi3
    if hasattr(cfg.MODEL, 'ARCHITECTURE') and cfg.MODEL.ARCHITECTURE == "AutoregressivePi3":
        model = AutoregressivePi3(
            n_future_frames=cfg.MODEL.N,
            ar_n_heads=cfg.MODEL.AR_N_HEADS,
            ar_n_layers=cfg.MODEL.AR_N_LAYERS,
            ar_dropout=cfg.MODEL.AR_DROPOUT,
            encoder_name=cfg.MODEL.ENCODER_NAME.lower(),
            freeze_decoders=getattr(cfg.MODEL, 'FREEZE_DECODERS', False),
            use_segmentation_head=cfg.MODEL.USE_SEGMENTATION_HEAD,
            segmentation_num_classes=cfg.MODEL.SEGMENTATION_NUM_CLASSES,
            use_motion_head=cfg.MODEL.USE_MOTION_HEAD,
            use_flow_head=cfg.MODEL.USE_FLOW_HEAD
        )
        
        print(f"✅ Created AutoregressivePi3 model with:")
        print(f"   - Encoder: {cfg.MODEL.ENCODER_NAME}")
        print(f"   - Input frames (M): {cfg.MODEL.M}")
        print(f"   - Future frames (N): {cfg.MODEL.N}")
        print(f"   - AR heads: {cfg.MODEL.AR_N_HEADS}")
        print(f"   - AR layers: {cfg.MODEL.AR_N_LAYERS}")
        print(f"   - AR dropout: {cfg.MODEL.AR_DROPOUT}")
        print(f"   - Segmentation head: {cfg.MODEL.USE_SEGMENTATION_HEAD}")
        print(f"   - Freeze decoders: {getattr(cfg.MODEL, 'FREEZE_DECODERS', False)}")
        
    else:
        # Default to AutonomyPi3
        model = AutonomyPi3(
            full_N=cfg.MODEL.M + cfg.MODEL.N,
            extra_tokens=cfg.MODEL.N,
            encoder_name=cfg.MODEL.ENCODER_NAME,
            dinov3_checkpoint_path=dinov3_local_path,
            use_motion_head=cfg.MODEL.USE_MOTION_HEAD,
            use_segmentation_head=cfg.MODEL.USE_SEGMENTATION_HEAD,
            segmentation_num_classes=cfg.MODEL.SEGMENTATION_NUM_CLASSES,
            use_detection_head=cfg.MODEL.USE_DETECTION_HEAD,
            num_detection_classes=cfg.MODEL.NUM_DETECTION_CLASSES,
            detection_architecture=cfg.MODEL.DETECTION_ARCHITECTURE,
            num_object_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
            detr_hidden_dim=cfg.MODEL.DETR_HIDDEN_DIM,
            detr_num_heads=cfg.MODEL.DETR_NUM_HEADS,
            detr_num_layers=cfg.MODEL.DETR_NUM_LAYERS,
            freeze_decoders=getattr(cfg.MODEL, 'FREEZE_DECODERS', False)
        )
        
        print(f"✅ Created AutonomyPi3 model with:")
        print(f"   - Encoder: {cfg.MODEL.ENCODER_NAME}")
        print(f"   - Input frames (M): {cfg.MODEL.M}")
        print(f"   - Target frames (N): {cfg.MODEL.N}")
        print(f"   - Motion head: {cfg.MODEL.USE_MOTION_HEAD}")
        print(f"   - Segmentation head: {cfg.MODEL.USE_SEGMENTATION_HEAD}")
        print(f"   - Detection head: {cfg.MODEL.USE_DETECTION_HEAD}")
        print(f"   - Freeze decoders: {getattr(cfg.MODEL, 'FREEZE_DECODERS', False)}")
    
    return model


def create_mapanything_model(cfg):
    """
    Create a MapAnything model with the given configuration.
    
    Args:
        cfg: YACS configuration object
        
    Returns:
        MapAnything: Configured MapAnything model
    """
    # Build configuration from YAML settings instead of hardcoded values
    config_overrides = [
        f"machine={cfg.MODEL.MAPANYTHING.MACHINE}",
        "model=mapanything",
        f"model/task={cfg.MODEL.MAPANYTHING.TASK}",
        f"model.encoder.uses_torch_hub={str(cfg.MODEL.MAPANYTHING.USE_TORCH_HUB).lower()}",
    ]
    
    high_level_config = {
        "path": cfg.MODEL.MAPANYTHING.CONFIG_PATH,
        "hf_model_name": cfg.MODEL.MAPANYTHING.HF_MODEL_NAME,
        "model_str": "mapanything",
        "config_overrides": config_overrides,
        "checkpoint_name": cfg.MODEL.MAPANYTHING.CHECKPOINT_NAME,
        "config_name": cfg.MODEL.MAPANYTHING.CONFIG_NAME,
        "trained_with_amp": cfg.MODEL.MAPANYTHING.TRAINED_WITH_AMP,
        "trained_with_amp_dtype": cfg.MODEL.MAPANYTHING.AMP_DTYPE,
        "data_norm_type": cfg.MODEL.MAPANYTHING.DATA_NORM_TYPE,
        "patch_size": cfg.MODEL.MAPANYTHING.PATCH_SIZE,
        "resolution": cfg.MODEL.MAPANYTHING.RESOLUTION,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"✅ Creating MapAnything model with configuration:")
    print(f"   - HF Model: {cfg.MODEL.MAPANYTHING.HF_MODEL_NAME}")
    print(f"   - Backbone: {cfg.MODEL.MAPANYTHING.BACKBONE}")
    print(f"   - Task: {cfg.MODEL.MAPANYTHING.TASK}")
    print(f"   - Resolution: {cfg.MODEL.MAPANYTHING.RESOLUTION}")
    print(f"   - Patch size: {cfg.MODEL.MAPANYTHING.PATCH_SIZE}")
    print(f"   - AMP: {cfg.MODEL.MAPANYTHING.TRAINED_WITH_AMP} ({cfg.MODEL.MAPANYTHING.AMP_DTYPE})")
    print(f"   - Data norm: {cfg.MODEL.MAPANYTHING.DATA_NORM_TYPE}")
    
    model = initialize_mapanything_model(high_level_config, device)
    
    print(f"✅ MapAnything model initialized successfully")
    return model


def get_model_info(cfg):
    """
    Get information about the model configuration.
    
    Args:
        cfg: YACS configuration object
        
    Returns:
        dict: Model information
    """
    architecture = cfg.MODEL.ARCHITECTURE.lower()
    
    info = {
        "architecture": cfg.MODEL.ARCHITECTURE,
        "input_frames": cfg.MODEL.M,
        "target_frames": cfg.MODEL.N,
        "encoder": cfg.MODEL.ENCODER_NAME,
    }
    
    if architecture in ["pi3", "autoregressivepi3"]:
        if architecture == "autoregressivepi3":
            info.update({
                "ar_heads": cfg.MODEL.AR_N_HEADS,
                "ar_layers": cfg.MODEL.AR_N_LAYERS,
                "ar_dropout": cfg.MODEL.AR_DROPOUT,
            })
        else:
            info.update({
                "motion_head": cfg.MODEL.USE_MOTION_HEAD,
                "segmentation_head": cfg.MODEL.USE_SEGMENTATION_HEAD,
                "detection_head": cfg.MODEL.USE_DETECTION_HEAD,
            })
    elif architecture == "mapanything":
        info.update({
            "backbone": cfg.MODEL.MAPANYTHING.BACKBONE,
            "decoder_dim": cfg.MODEL.MAPANYTHING.DECODER_DIM,
            "num_layers": cfg.MODEL.MAPANYTHING.NUM_LAYERS,
            "temporal": cfg.MODEL.MAPANYTHING.USE_TEMPORAL,
        })
    
    return info


def validate_model_config(cfg):
    """
    Validate model configuration for consistency.
    
    Args:
        cfg: YACS configuration object
        
    Raises:
        ValueError: If configuration is invalid
    """
    architecture = cfg.MODEL.ARCHITECTURE.lower()
    
    if architecture not in ["pi3", "mapanything", "autoregressivepi3"]:
        raise ValueError(f"Unsupported model architecture: {cfg.MODEL.ARCHITECTURE}")
    
    if cfg.MODEL.M <= 0 or cfg.MODEL.N <= 0:
        raise ValueError("Input frames (M) and target frames (N) must be positive")
    
    if architecture in ["pi3", "autoregressivepi3"]:
        valid_encoders = ["dinov2", "dinov3"]
        if cfg.MODEL.ENCODER_NAME not in valid_encoders:
            raise ValueError(f"Pi3/AutoregressivePi3 encoder must be one of {valid_encoders}, got {cfg.MODEL.ENCODER_NAME}")
        
        # AutoregressivePi3 specific validation
        if architecture == "autoregressivepi3":
            if not hasattr(cfg.MODEL, 'AR_N_HEADS') or cfg.MODEL.AR_N_HEADS <= 0:
                raise ValueError("AutoregressivePi3 requires positive AR_N_HEADS")
            if not hasattr(cfg.MODEL, 'AR_N_LAYERS') or cfg.MODEL.AR_N_LAYERS <= 0:
                raise ValueError("AutoregressivePi3 requires positive AR_N_LAYERS")
            if not hasattr(cfg.MODEL, 'AR_DROPOUT') or cfg.MODEL.AR_DROPOUT < 0 or cfg.MODEL.AR_DROPOUT >= 1:
                raise ValueError("AutoregressivePi3 requires AR_DROPOUT in [0, 1)")
    
    elif architecture == "mapanything":
        if cfg.MODEL.MAPANYTHING.DECODER_DIM <= 0:
            raise ValueError("MapAnything decoder dimension must be positive")
        if cfg.MODEL.MAPANYTHING.NUM_LAYERS <= 0:
            raise ValueError("MapAnything number of layers must be positive")
    
    print(f"✅ Model configuration validation passed for {cfg.MODEL.ARCHITECTURE}")