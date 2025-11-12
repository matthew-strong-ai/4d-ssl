"""
Default configuration for Pi3 training.
"""

import os
from yacs.config import CfgNode as CN

# Create the config node
_C = CN()

# Dataset parameters
_C.DATASET = CN()
# Local dataset (legacy support)
_C.DATASET.ROOT_DIR = ""                    # Root directory containing subfolders of images

# Image dimensions
_C.DATASET.IMG_HEIGHT = 224                 # Input image height
_C.DATASET.IMG_WIDTH = 224                  # Input image width

# S3 dataset configuration
_C.DATASET.USE_S3 = False                   # Use S3 dataset instead of local files
_C.DATASET.S3_BUCKET = ""                   # S3 bucket name
_C.DATASET.S3_SEQUENCE_PREFIXES = []        # List of S3 prefixes for image sequences
_C.DATASET.S3_IMAGE_EXTENSION = ".jpg"      # Image file extension in S3
_C.DATASET.S3_REGION = "us-east-1"         # AWS region
_C.DATASET.S3_PRELOAD_BYTES = False        # Preload all bytes for multiprocessing

# YouTube S3 dataset configuration
_C.DATASET.USE_YOUTUBE = False              # Use YouTube S3 dataset
_C.DATASET.YOUTUBE_ROOT_PREFIX = "openDV-YouTube/full_images/"  # Root prefix for YouTube dataset
_C.DATASET.YOUTUBE_CACHE_DIR = "./youtube_cache"                # Cache directory for metadata
_C.DATASET.YOUTUBE_REFRESH_CACHE = False    # Force refresh cache
_C.DATASET.YOUTUBE_SKIP_FRAMES = 300        # Skip first N frames per video
_C.DATASET.YOUTUBE_MIN_SEQUENCE_LENGTH = 50 # Minimum video length
_C.DATASET.YOUTUBE_MAX_WORKERS = 8          # Parallel workers for S3 indexing
_C.DATASET.MAX_SAMPLES = -1                 # Limit dataset size (-1 for all samples)
_C.DATASET.FRAME_SAMPLING_RATE = 1         # Sample every Nth frame (1=10Hz, 5=2Hz) from source 10Hz videos

_C.DATASET.BATCH_SIZE = 20                  # Batch size for training
_C.DATASET.VAL_SPLIT = 0.1                 # Fraction of data to use for validation

# Model parameters
_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = "Pi3"              # Model architecture: "Pi3", "AutonomyPi3", "AutoregressivePi3", or "MapAnything"
_C.MODEL.M = 3                             # Number of input frames
_C.MODEL.N = 3                             # Number of target frames
_C.MODEL.GRID_SIZE = 10                    # Grid size for query points
_C.MODEL.ENCODER_NAME = "dinov3"           # Encoder type: "dinov2" or "dinov3"

# AutoregressivePi3 specific parameters
_C.MODEL.AR_N_HEADS = 16                   # Number of attention heads for autoregressive transformer
_C.MODEL.AR_N_LAYERS = 8                   # Number of layers for autoregressive transformer
_C.MODEL.AR_DROPOUT = 0.1                  # Dropout rate for autoregressive transformer

# Distilled ViT parameters
_C.MODEL.USE_DISTILLED_VIT = False         # Enable distilled ViT training
_C.MODEL.DISTILLED_VIT = CN()
_C.MODEL.DISTILLED_VIT.EMBED_DIM = 768     # Student embedding dimension
_C.MODEL.DISTILLED_VIT.DEPTH = 12          # Number of transformer layers
_C.MODEL.DISTILLED_VIT.NUM_HEADS = 12      # Number of attention heads
_C.MODEL.DISTILLED_VIT.DISTILL_TOKENS = ['point_features', 'camera_features', 'autonomy_features']  # Features to distill
_C.MODEL.DISTILLED_VIT.TEMPERATURE = 4.0   # Distillation temperature
_C.MODEL.DISTILLED_VIT.USE_COSINE_SIMILARITY = True  # Use cosine similarity loss
_C.MODEL.DISTILLED_VIT.LOAD_PRETRAINED = True  # Load pretrained DINOv2 weights for initialization

_C.MODEL.USE_DETECTION_HEAD = False        # Enable optional detection head for traffic lights/road signs
_C.MODEL.NUM_DETECTION_CLASSES = 2         # Number of detection classes (traffic light, road sign)

# Detection head architecture options
_C.MODEL.DETECTION_ARCHITECTURE = "dense"  # Detection architecture: "dense" or "detr"
_C.MODEL.NUM_OBJECT_QUERIES = 100          # Number of object queries for DETR (only used if DETECTION_ARCHITECTURE="detr")
_C.MODEL.DETR_HIDDEN_DIM = 256             # DETR decoder hidden dimension
_C.MODEL.DETR_NUM_HEADS = 8                # Number of attention heads in DETR decoder
_C.MODEL.DETR_NUM_LAYERS = 6               # Number of DETR decoder layers
_C.MODEL.USE_MOTION_HEAD = False           # Enable motion head
_C.MODEL.USE_FLOW_HEAD = False             # Enable optical flow head
_C.MODEL.USE_SEGMENTATION_HEAD = True       # Enable segmentation head
_C.MODEL.SEGMENTATION_MODEL = "gsam2"   # Segmentation model: "gsam2", "segformer", "deeplabv3"
_C.MODEL.SEGMENTATION_NUM_CLASSES = 7       # Number of segmentation classes (7 for Cityscapes, 6 for GSAM2)
_C.MODEL.FREEZE_DECODERS = False            # Freeze point, conf, and camera decoders/heads
_C.MODEL.USE_FROZEN_DECODER_SUPERVISION = False  # Use frozen model's decoder features as supervision for autoregressive transformer

# MapAnything specific parameters (only used when ARCHITECTURE="MapAnything")
_C.MODEL.MAPANYTHING = CN()
_C.MODEL.MAPANYTHING.BACKBONE = "dinov2"    # Backbone for MapAnything: "dinov3", "sam", etc.
_C.MODEL.MAPANYTHING.DECODER_DIM = 512      # MapAnything decoder dimension
_C.MODEL.MAPANYTHING.NUM_LAYERS = 6         # Number of decoder layers
_C.MODEL.MAPANYTHING.USE_TEMPORAL = True    # Enable temporal processing
_C.MODEL.MAPANYTHING.TEMPORAL_WINDOW = 3    # Temporal window size

# PPGeo specific parameters
_C.PPGEO = CN()
_C.PPGEO.STAGE = 1                          # PPGeo stage: 1 (depth+pose), 2 (structure+motion)
_C.PPGEO.FRAME_IDS = [-1, 0, 1]            # Frame indices for temporal consistency
_C.PPGEO.SCALES = [0, 1, 2, 3]             # Multi-scale supervision scales
_C.PPGEO.MIN_DEPTH = 0.1                   # Minimum depth value
_C.PPGEO.MAX_DEPTH = 100.0                 # Maximum depth value
_C.PPGEO.ENCODER = "dinov3"                 # Encoder type: "dinov3", "dinov2", "vit", or "resnet"
_C.PPGEO.RESNET_LAYERS = 18                 # Number of ResNet layers (18, 34, 50, 101, 152)
_C.STAGE1_CHECKPOINT = ""                   # Path to Stage 1 checkpoint for Stage 2 training
_C.DEPTHANYTHING_CHECKPOINT = "/home/matthew_strong/Desktop/autonomy-wild/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth"  # Pre-trained DepthAnything checkpoint

# MapAnything HuggingFace configuration
_C.MODEL.MAPANYTHING.HF_MODEL_NAME = "facebook/map-anything"  # HuggingFace model name
_C.MODEL.MAPANYTHING.CONFIG_PATH = "/home/matthew_strong/Desktop/autonomy-wild/map-anything/configs/train.yaml"       # Path to MapAnything config
_C.MODEL.MAPANYTHING.CHECKPOINT_NAME = "model.safetensors"    # Checkpoint file name
_C.MODEL.MAPANYTHING.CONFIG_NAME = "config.json"             # Config file name
_C.MODEL.MAPANYTHING.MACHINE = "aws"                          # Machine type
_C.MODEL.MAPANYTHING.TASK = "images_only"                     # Task type
_C.MODEL.MAPANYTHING.USE_TORCH_HUB = False                    # Whether encoder uses torch hub
_C.MODEL.MAPANYTHING.TRAINED_WITH_AMP = True                  # Whether trained with AMP
_C.MODEL.MAPANYTHING.AMP_DTYPE = "bf16"                       # AMP data type
_C.MODEL.MAPANYTHING.DATA_NORM_TYPE = "dinov2"                # Data normalization type
_C.MODEL.MAPANYTHING.PATCH_SIZE = 14                          # Patch size
_C.MODEL.MAPANYTHING.RESOLUTION = 518                         # Input resolution

_C.POINT_MOTION = CN()
_C.POINT_MOTION.TRAIN_MOTION = False       # Train point motion prediction head
_C.POINT_MOTION.MOTION_THRESHOLD = 0.1     # Threshold for converting motion magnitude to binary mask (meters)

# Motion Detection parameters (using optical flow)
_C.MOTION_DETECTION = CN()
_C.MOTION_DETECTION.ENABLE = False                    # Enable flow-based motion detection
_C.MOTION_DETECTION.MOTION_THRESHOLD = 0.1           # 3D motion threshold in meters per frame
_C.MOTION_DETECTION.TEMPORAL_WINDOW = 5              # Number of frames for object motion classification
_C.MOTION_DETECTION.CONSISTENCY_THRESHOLD = 0.6      # Fraction of frames that must be dynamic for classification
_C.MOTION_DETECTION.CAMERA_MOTION_COMPENSATION = True # Remove camera motion from object motion
_C.MOTION_DETECTION.SAVE_MOTION_VISUALIZATIONS = True # Save motion field and dynamic object visualizations
_C.MOTION_DETECTION.MAX_MOTION_DISPLAY = 1.0         # Maximum motion magnitude for visualization scaling
_C.MOTION_DETECTION.VISUALIZATION_SUBSAMPLE = 16     # Arrow plot subsampling factor

# GroundingDINO Detection parameters
_C.DETECTION = CN()
_C.DETECTION.USE_GROUNDING_DINO = False     # Enable GroundingDINO for detection
_C.DETECTION.GDINO_CONFIG_PATH = ""         # Path to GroundingDINO config file (.py)
_C.DETECTION.GDINO_WEIGHTS_PATH = ""        # Path to GroundingDINO weights (.pth)
_C.DETECTION.TEXT_PROMPT = "traffic light . road sign"  # Detection text prompt
_C.DETECTION.BOX_THRESHOLD = 0.3            # Detection confidence threshold
_C.DETECTION.TEXT_THRESHOLD = 0.25          # Text-image similarity threshold

# Training parameters
_C.TRAINING = CN()
_C.TRAINING.NUM_EPOCHS = 5                 # Number of training epochs
_C.TRAINING.LEARNING_RATE = 1e-5          # Learning rate (reduced for stability)
_C.TRAINING.WARMUP_STEPS = 500            # Number of warmup steps for learning rate scheduler
_C.TRAINING.WARMUP_START_FACTOR = 0.1     # Starting factor for warmup
_C.TRAINING.GRAD_ACCUM_STEPS = 1          # Number of gradient accumulation steps
_C.TRAINING.MAX_GRAD_NORM = 0.5           # Max gradient norm for clipping (reduced for stability)
_C.TRAINING.USE_FP32_FOR_LOSSES = True    # Use FP32 for loss computations (more stable than BF16)
_C.TRAINING.DETECT_NANS = True            # Enable NaN detection and recovery

# Loss weights
_C.LOSS = CN()
_C.LOSS.PC_LOSS_WEIGHT = 0.1              # Point cloud loss weight
_C.LOSS.POSE_LOSS_WEIGHT = 0.9            # Camera pose loss weight
_C.LOSS.CONF_LOSS_WEIGHT = 0.5            # Confidence loss weight
_C.LOSS.NORMAL_LOSS_WEIGHT = 0.0          # Normal loss weight
_C.LOSS.DETECTION_LOSS_WEIGHT = 0.0       # Detection loss weight (GroundingDINO supervision)
_C.LOSS.SEGMENTATION_LOSS_WEIGHT = 0.1     # Segmentation loss weight (composite mask supervision)
_C.LOSS.SEGMENTATION_USE_FOCAL_LOSS = True  # Use focal loss for class imbalance (vs weighted CE)
_C.LOSS.MOTION_LOSS_WEIGHT = 0.1           # Motion loss weight (3D motion vector supervision)
_C.LOSS.FLOW_LOSS_WEIGHT = 0.1             # Optical flow loss weight (2D flow supervision)
_C.LOSS.FUTURE_FRAME_WEIGHT = 1.0         # Weight multiplier for future frame supervision
_C.LOSS.FROZEN_DECODER_SUPERVISION_WEIGHT = 1.0  # Weight for frozen model decoder supervision loss
_C.LOSS.DISTILLATION_LOSS_WEIGHT = 1.0        # Weight for distilled ViT loss
_C.LOSS.DISTILLATION_POINT_FEATURES_WEIGHT = 1.0   # Weight for point_features distillation
_C.LOSS.DISTILLATION_CAMERA_FEATURES_WEIGHT = 1.0  # Weight for camera_features distillation
_C.LOSS.DISTILLATION_AUTONOMY_FEATURES_WEIGHT = 1.0 # Weight for autonomy_features distillation

# PPGeo loss weights
_C.LOSS.PHOTOMETRIC_WEIGHT = 1.0          # Photometric consistency loss weight
_C.LOSS.SMOOTHNESS_WEIGHT = 0.001         # Depth smoothness loss weight

# Confidence-weighted point loss parameters
_C.LOSS.USE_CONF_WEIGHTED_POINTS = True  # Use confidence-weighted point loss instead of scale-invariant loss
_C.LOSS.CONF_GAMMA = 1.0                  # Weight for confidence-weighted reconstruction loss
_C.LOSS.CONF_ALPHA = 0.1                  # Weight for confidence regularization term
_C.LOSS.GRADIENT_WEIGHT = 0.1             # Weight for gradient comparison term in confidence loss

# Validation parameters
_C.VALIDATION = CN()
_C.VALIDATION.VAL_FREQ = 1000             # Validate every N steps
_C.VALIDATION.VAL_SAMPLES = 50            # Number of validation samples to use (-1 for all)
_C.VALIDATION.EARLY_STOPPING_PATIENCE = 10  # Early stopping patience

# Logging parameters
_C.LOGGING = CN()
_C.LOGGING.LOG_FREQ = 50                  # Log every N steps
_C.LOGGING.SAVE_FREQ = 10000              # Check for best model every N steps
_C.LOGGING.N_VISUALIZE = 3                # Number of random batches to visualize before training
_C.LOGGING.SAVE_IMAGES = True             # Save sample images to PNG during training
_C.LOGGING.SAVE_IMAGES_STEPS = 5          # Save images for first N training steps
_C.LOGGING.SAVE_IMAGES_DIR = "./sample_images"  # Directory to save sample images

# Output parameters
_C.OUTPUT = CN()
_C.OUTPUT.CHECKPOINT_DIR = "checkpoints"   # Directory to save checkpoints
_C.OUTPUT.SAVE_NPZ = False                # Save npz files
_C.OUTPUT.SAVE_DEPTHS = False             # Save depths files and visualizations
_C.OUTPUT.SAVE_SEGMENTATION = True        # Save segmentation visualizations (PNG + WandB)

# S3 checkpoint upload parameters
_C.OUTPUT.UPLOAD_TO_S3 = True             # Upload checkpoints to S3
_C.OUTPUT.S3_BUCKET = "research-datasets"  # S3 bucket for checkpoints
_C.OUTPUT.S3_PREFIX = "autonomy_checkpoints"  # S3 prefix/folder for checkpoints

# Augmentation parameters
_C.AUGMENTATION = CN()
_C.AUGMENTATION.USE_AUGMENTATIONS = True  # Apply random augmentations during training

# W&B parameters
_C.WANDB = CN()
_C.WANDB.PROJECT = "pi3-cluster-training"  # Weights & Biases project name
_C.WANDB.USE_WANDB = True                 # Enable Weights & Biases logging
_C.WANDB.RUN_NAME = ""                    # Custom run name (empty = auto-generated)

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

def update_config(cfg, args=None):
    """Update config with command line arguments"""
    cfg.defrost()

    
    # Always try to load from default config.yaml first
    default_config_path = "config.yaml"
    if os.path.exists(default_config_path):
        cfg.merge_from_file(default_config_path)
        print(f"✅ Loaded default configuration from {default_config_path}")
    elif os.path.exists("config/config.yaml"):
        cfg.merge_from_file("config/config.yaml")
        print(f"✅ Loaded default configuration from config/config.yaml")
    
    # cfg.merge_from_file("config_mapanything.yaml")

    # Update from args if they exist
    if args:
        if hasattr(args, 'config') and args.config:
            cfg.merge_from_file(args.config)
            print(f"✅ Overridden with configuration from {args.config}")
        
        if hasattr(args, 'opts') and args.opts:
            cfg.merge_from_list(args.opts)
            print("✅ Applied command line overrides")
    
    cfg.freeze()
    return cfg