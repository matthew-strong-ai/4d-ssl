# Copyright (c) 2025 Matt Strong. Created for Self Supervised Learning from In the Wild Driving Videos
# Fixed version with GPU synchronization improvements to prevent cluster freezing

import argparse
import os
import torch
import signal
import time
from functools import wraps
import threading
from contextlib import contextmanager

from accelerate import Accelerator
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb
import torchvision.transforms.functional as TF
import random

# Configuration management
from yacs.config import CfgNode as CN
from config.defaults import get_cfg_defaults, update_config

from utils.geometry_torch import recover_focal_shift
import utils3d

# all imports for spatracker.
from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image, get_default_transforms
from SpaTrackerV2.ssl_image_dataset import SequenceLearningDataset
from simple_s3_dataset import S3Dataset

# add to path where pi3 is located (one folder deep relative to this file)
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pi3"))

from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3, AutonomyPi3, AutoregressivePi3

# import pi3 losses
from losses import Pi3Losses

#################################################################################################

from rich import print
import random
import numpy as np

import datetime
import subprocess
from io import BytesIO
import boto3
import json

# ================================================================================================
# GPU Synchronization and Timeout Utilities
# ================================================================================================

class TimeoutHandler:
    """Handle timeouts for distributed operations."""
    
    def __init__(self, timeout_seconds=300):
        self.timeout_seconds = timeout_seconds
        self.timed_out = False
    
    def timeout_handler(self, signum, frame):
        self.timed_out = True
        raise TimeoutError(f"Operation timed out after {self.timeout_seconds} seconds")
    
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.timeout_seconds)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
        if exc_type is TimeoutError:
            print(f"⏰ Timeout occurred after {self.timeout_seconds} seconds")
            return False

def distributed_barrier_with_timeout(accelerator, timeout=300, description="Synchronization"):
    """Safe distributed barrier with timeout and error handling."""
    if not accelerator.use_distributed:
        return True
    
    try:
        print(f"🔄 {description} - waiting for all processes...")
        
        with TimeoutHandler(timeout):
            # Use accelerator's built-in barrier with timeout handling
            start_time = time.time()
            accelerator.wait_for_everyone()
            elapsed = time.time() - start_time
            
            if accelerator.is_main_process:
                print(f"✅ {description} completed in {elapsed:.2f}s")
            return True
            
    except TimeoutError:
        if accelerator.is_main_process:
            print(f"⏰ {description} timed out after {timeout}s")
        return False
    except Exception as e:
        if accelerator.is_main_process:
            print(f"❌ {description} failed: {e}")
        return False

@contextmanager
def gpu_memory_guard(accelerator, description="Operation"):
    """Context manager for safe GPU memory operations."""
    try:
        if accelerator.is_main_process:
            print(f"🔒 Starting {description}")
        
        # Clear cache before operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        yield
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"💥 GPU OOM during {description}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            raise
        else:
            raise
    finally:
        # Always clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if accelerator.is_main_process:
                print(f"🧹 Cleaned up after {description}")

def safe_distributed_operation(func, accelerator, max_retries=3, timeout=300, description="Distributed operation"):
    """Wrapper for safe distributed operations with retries."""
    for attempt in range(max_retries):
        try:
            # Barrier before operation
            if not distributed_barrier_with_timeout(accelerator, timeout, f"{description} - pre-barrier"):
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Pre-barrier timeout for {description}")
                continue
            
            # Execute the operation
            result = func()
            
            # Barrier after operation
            if not distributed_barrier_with_timeout(accelerator, timeout, f"{description} - post-barrier"):
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Post-barrier timeout for {description}")
                continue
                
            return result
            
        except Exception as e:
            if accelerator.is_main_process:
                print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed for {description}: {e}")
            
            if attempt == max_retries - 1:
                raise
            
            # Wait before retry
            time.sleep(2 ** attempt)  # Exponential backoff
            
            # Clear GPU memory before retry
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

# ================================================================================================
# Enhanced Validation Function with Distributed Safety
# ================================================================================================

def run_validation_safe(train_model, frozen_model, val_dataloader, cfg, accelerator, dtype=torch.bfloat16):
    """
    Run validation with distributed safety and timeout handling.
    """
    if val_dataloader is None:
        return None
    
    def validation_operation():
        train_model.eval()
        
        total_val_loss = 0.0
        total_point_loss = 0.0
        total_camera_loss = 0.0
        total_conf_loss = 0.0
        total_unweighted_l1_points = 0.0
        total_unweighted_pose_loss = 0.0
        num_batches = 0
        max_batches = cfg.VALIDATION.VAL_SAMPLES if cfg.VALIDATION.VAL_SAMPLES > 0 else len(val_dataloader)
        
        if accelerator.is_main_process:
            print(f"🔍 Running validation on {min(max_batches, len(val_dataloader))} samples...")
        
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                if step >= max_batches:
                    break
                
                try:
                    with gpu_memory_guard(accelerator, f"Validation step {step}"):
                        # Process batch
                        X = batch[0]  # (B, m, C, H, W) - current frames
                        y = batch[1]  # (B, n, C, H, W) - future frames
                        X_all = torch.cat([X, y], dim=1)  # (B, T, C, H, W) where T = m + n
                        
                        # For validation, both models use unaugmented data
                        batch_size = X_all.shape[0]
                        if batch_size == 1:
                            video_tensor = preprocess_image(X_all[0]).unsqueeze(0)  # (1, T, C, H, W)
                        else:
                            video_tensors = []
                            for b in range(batch_size):
                                sample = X_all[b]  # (T, C, H, W)
                                processed_sample = preprocess_image(sample)  # (T, C, H, W)
                                video_tensors.append(processed_sample)
                            video_tensor = torch.stack(video_tensors, dim=0)
                        
                        subset_video_tensor = video_tensor[:, :cfg.MODEL.M]  # (B, m, C, H, W)
                        
                        # Get ground truth from frozen model
                        with torch.amp.autocast('cuda', dtype=dtype):
                            pseudo_gt = frozen_model(video_tensor)
                        
                        # Get predictions from training model
                        with torch.amp.autocast('cuda', dtype=dtype):
                            predictions = train_model(subset_video_tensor)
                        
                        # Compute validation loss
                        if cfg.LOSS.USE_CONF_WEIGHTED_POINTS:
                            point_map_loss, camera_pose_loss, conf_loss = Pi3Losses.pi3_loss_with_confidence_weighting(
                                predictions, pseudo_gt, m_frames=cfg.MODEL.M, future_frame_weight=cfg.LOSS.FUTURE_FRAME_WEIGHT,
                                gamma=cfg.LOSS.CONF_GAMMA, alpha=cfg.LOSS.CONF_ALPHA, use_conf_weighted_points=True, gradient_weight=cfg.LOSS.GRADIENT_WEIGHT
                            )
                        else:
                            point_map_loss, camera_pose_loss, conf_loss = Pi3Losses.pi3_loss(
                                predictions, pseudo_gt, m_frames=cfg.MODEL.M, future_frame_weight=cfg.LOSS.FUTURE_FRAME_WEIGHT, gradient_weight=cfg.LOSS.GRADIENT_WEIGHT
                            )
                        
                        val_loss = (
                            cfg.LOSS.PC_LOSS_WEIGHT * point_map_loss
                            + cfg.LOSS.POSE_LOSS_WEIGHT * camera_pose_loss
                            + cfg.LOSS.CONF_LOSS_WEIGHT * conf_loss
                        )
                        
                        # Compute unweighted validation losses
                        pred_points = predictions['points']  # [B, T, H, W, 3]
                        gt_points = pseudo_gt['points']      # [B, T, H, W, 3]
                        unweighted_l1_points = torch.nn.functional.l1_loss(pred_points, gt_points, reduction='mean')
                        
                        pred_poses = predictions['camera_poses']  # [B, T, 4, 4]
                        gt_poses = pseudo_gt['camera_poses']      # [B, T, 4, 4]
                        unweighted_pose_loss = torch.nn.functional.l1_loss(pred_poses, gt_poses, reduction='mean')
                        
                        # Accumulate losses
                        total_val_loss += val_loss.item()
                        total_point_loss += point_map_loss.item()
                        total_camera_loss += camera_pose_loss.item()
                        total_conf_loss += conf_loss.item() if torch.is_tensor(conf_loss) else conf_loss
                        total_unweighted_l1_points += unweighted_l1_points.item()
                        total_unweighted_pose_loss += unweighted_pose_loss.item()
                        num_batches += 1
                        
                except Exception as e:
                    if accelerator.is_main_process:
                        print(f"⚠️ Validation step {step} failed: {e}")
                    continue
        
        # Calculate averages
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
        avg_point_loss = total_point_loss / num_batches if num_batches > 0 else float('inf')
        avg_camera_loss = total_camera_loss / num_batches if num_batches > 0 else float('inf')
        avg_conf_loss = total_conf_loss / num_batches if num_batches > 0 else float('inf')
        avg_unweighted_l1_points = total_unweighted_l1_points / num_batches if num_batches > 0 else float('inf')
        avg_unweighted_pose_loss = total_unweighted_pose_loss / num_batches if num_batches > 0 else float('inf')
        
        train_model.train()  # Back to training mode
        
        return {
            'val_loss': avg_val_loss,
            'val_point_loss': avg_point_loss,
            'val_camera_loss': avg_camera_loss,
            'val_conf_loss': avg_conf_loss,
            'val_unweighted_l1_points': avg_unweighted_l1_points,
            'val_unweighted_pose_loss': avg_unweighted_pose_loss,
            'num_samples': num_batches
        }
    
    # Execute validation with distributed safety
    return safe_distributed_operation(
        validation_operation, 
        accelerator, 
        max_retries=2, 
        timeout=600,  # 10 minutes for validation
        description="Validation"
    )

# ================================================================================================
# Original utility functions (unchanged but added for completeness)
# ================================================================================================

def check_for_nans(tensor, name, step=None):
    """Check tensor for NaN values and print detailed diagnostics."""
    if torch.isnan(tensor).any():
        step_info = f" at step {step}" if step is not None else ""
        print(f"❌ NaN detected in {name}{step_info}")
        print(f"   Shape: {tensor.shape}")
        print(f"   Min: {tensor.min().item()}, Max: {tensor.max().item()}")
        print(f"   NaN count: {torch.isnan(tensor).sum().item()}")
        if tensor.numel() < 100:
            print(f"   Values: {tensor}")
        return True
    
    if torch.isinf(tensor).any():
        step_info = f" at step {step}" if step is not None else ""
        print(f"⚠️ Inf detected in {name}{step_info}")
        print(f"   Shape: {tensor.shape}")
        print(f"   Min: {tensor.min().item()}, Max: {tensor.max().item()}")
        print(f"   Inf count: {torch.isinf(tensor).sum().item()}")
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

def apply_random_augmentations(images, training=True):
    """Apply random augmentations to images with different amounts per image."""
    if not training:
        return images
        
    T, C, H, W = images.shape
    augmented_images = []
    
    for t in range(T):
        img = images[t]  # (C, H, W)
        
        # Random color jittering with different amounts per image
        brightness_factor = random.uniform(0.7, 1.3)
        contrast_factor = random.uniform(0.7, 1.3) 
        saturation_factor = random.uniform(0.7, 1.3)
        hue_factor = random.uniform(-0.1, 0.1)
        
        # Apply color jittering
        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)
        img = TF.adjust_saturation(img, saturation_factor)
        img = TF.adjust_hue(img, hue_factor)
        
        # Random Gaussian blur (20% chance)
        if random.random() < 0.2:
            sigma = random.uniform(0.1, 1.0)
            img = TF.gaussian_blur(img, kernel_size=3, sigma=sigma)
        
        # Random grayscale (10% chance)
        if random.random() < 0.1:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
        
        # Clamp values to valid range
        img = torch.clamp(img, 0.0, 1.0)
        
        augmented_images.append(img)
    
    return torch.stack(augmented_images, dim=0)

def save_batch_images_to_png(X_all, step, cfg, max_batches=3, max_frames_per_batch=6):
    """Save images from X_all tensor to PNG files."""
    import os
    from PIL import Image as PILImage
    
    # Check if image saving is enabled
    if not cfg.LOGGING.SAVE_IMAGES:
        return
    
    # Only save images for the first few steps to avoid filling disk
    if step > cfg.LOGGING.SAVE_IMAGES_STEPS:
        return
    
    output_dir = cfg.LOGGING.SAVE_IMAGES_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    B, T, C, H, W = X_all.shape
    print(f"🖼️ Saving images from step {step}: {B} batches, {T} frames each")
    
    # ImageNet normalization values for denormalization (if needed)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Limit number of batches and frames to save
    num_batches_to_save = min(B, max_batches)
    num_frames_to_save = min(T, max_frames_per_batch)
    
    for b in range(num_batches_to_save):
        for t in range(num_frames_to_save):
            try:
                # Get single image: (C, H, W)
                img_tensor = X_all[b, t]  # Shape: (C, H, W)
                
                # Convert to numpy and transpose to (H, W, C)
                img_array = img_tensor.cpu().numpy().transpose(1, 2, 0)
                
                # Check if image is normalized (values in [0,1] or [-1,1] range)
                if img_array.min() >= 0 and img_array.max() <= 1:
                    # Assume it's in [0,1] range, convert to [0,255]
                    img_array = (img_array * 255).astype(np.uint8)
                elif img_array.min() >= -1 and img_array.max() <= 1:
                    # Assume it's normalized [-1,1], convert to [0,255]
                    img_array = ((img_array + 1) * 127.5).astype(np.uint8)
                else:
                    # Try ImageNet denormalization
                    img_array = img_array * std + mean
                    img_array = np.clip(img_array, 0, 1)
                    img_array = (img_array * 255).astype(np.uint8)
                
                # Convert to PIL Image and save
                pil_image = PILImage.fromarray(img_array)
                
                # Create filename with step, batch, and frame info
                filename = f"step_{step:04d}_batch_{b:02d}_frame_{t:02d}.png"
                save_path = os.path.join(output_dir, filename)
                pil_image.save(save_path)
                
            except Exception as e:
                print(f"❌ Error saving image step {step}, batch {b}, frame {t}: {e}")
                continue
    
    print(f"✅ Saved {num_batches_to_save}x{num_frames_to_save} images to {output_dir}")

def save_state_dict_to_s3(state_dict, s3_path: str):
    """Save a PyTorch state dict to an S3 bucket."""
    assert s3_path.startswith("s3://"), "Not a valid S3 path"
    s3_path = s3_path[5:]  # remove 's3://'
    bucket, key = s3_path.split("/", 1)
    
    print(f"📤 Starting state dict upload to s3://{bucket}/{key}")
    
    # Serialize the state dict
    buffer = BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    
    # Upload to S3
    session = boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
        region_name=os.getenv("REGION"),
    )
    client = session.client("s3", endpoint_url=os.getenv("ENDPOINT_URL"))
    client.upload_fileobj(buffer, bucket, key)
    print(f"✅ Successfully saved state dict to s3://{bucket}/{key}")

def upload_file_to_s3(file_path, s3_bucket="research-datasets", s3_prefix="autonomy_checkpoints", wandb_run_name=None):
    """Upload any file to S3 using AWS CLI."""
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return False
    
    # Get base filename and modify it to include wandb run name
    base_filename = os.path.basename(file_path)
    
    if wandb_run_name:
        # Split filename and extension
        name, ext = os.path.splitext(base_filename)
        # Create new filename with wandb run name
        filename = f"{wandb_run_name}_{name}{ext}"
    else:
        filename = base_filename
    
    s3_path = f"s3://{s3_bucket}/{s3_prefix}/{filename}"
    
    try:
        print(f"📤 Uploading file to S3: {s3_path}")
        
        # Use AWS CLI to upload
        cmd = ["aws", "s3", "cp", file_path, s3_path]

        # Set AWS environment variables for checksum validation
        env = os.environ.copy()
        env["AWS_REQUEST_CHECKSUM_CALCULATION"] = "when_required"
        env["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "when_required"
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)  # 5 min timeout
        
        if result.returncode == 0:
            print(f"✅ Successfully uploaded file to S3: {s3_path}")
            return True
        else:
            print(f"❌ Failed to upload file to S3:")
            print(f"   Command: {' '.join(cmd)}")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ S3 upload timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error uploading checkpoint to S3: {e}")
        return False

def denormalize_intrinsics(intrinsics, image_shape):
    """More robust denormalization handling different tensor shapes."""
    H, W = image_shape[-2:]
    
    # Handle batch and temporal dimensions
    original_shape = intrinsics.shape
    intrinsics_flat = intrinsics.view(-1, *intrinsics.shape[-2:])
    
    if intrinsics.shape[-2:] == (3, 3):
        # Scale the intrinsics matrix
        scale_matrix = torch.tensor([
            [W, 1, W],
            [1, H, H], 
            [1, 1, 1]
        ], device=intrinsics.device, dtype=intrinsics.dtype)
        
        intrinsics_flat = intrinsics_flat * scale_matrix[None, ...]
    
    return intrinsics_flat.view(original_shape)

# ================================================================================================
# Enhanced Training Function with Distributed Safety
# ================================================================================================

def train_model(train_config=None, experiment_tracker=None):
    """
    Main training function with enhanced GPU synchronization and timeout handling.
    """
    
    # Get default config and load from config.yaml automatically
    cfg = get_cfg_defaults()
    cfg = update_config(cfg)  # This will automatically load config.yaml
    
    # If train_config is provided (e.g., from Ray Tune), update the config
    if train_config and isinstance(train_config, dict):
        cfg.defrost()
        # Map Ray Tune config to YACS config structure
        if 'learning_rate' in train_config:
            cfg.TRAINING.LEARNING_RATE = train_config['learning_rate']
        if 'num_epochs' in train_config:
            cfg.TRAINING.NUM_EPOCHS = train_config['num_epochs']
        if 'batch_size' in train_config:
            cfg.DATASET.BATCH_SIZE = train_config['batch_size']
        if 'future_frame_weight' in train_config:
            cfg.LOSS.FUTURE_FRAME_WEIGHT = train_config['future_frame_weight']
        if 'pc_loss_weight' in train_config:
            cfg.LOSS.PC_LOSS_WEIGHT = train_config['pc_loss_weight']
        if 'pose_loss_weight' in train_config:
            cfg.LOSS.POSE_LOSS_WEIGHT = train_config['pose_loss_weight']
        if 'conf_loss_weight' in train_config:
            cfg.LOSS.CONF_LOSS_WEIGHT = train_config['conf_loss_weight']
        cfg.freeze()
    
    # Initialize wandb
    if cfg.WANDB.USE_WANDB:
        wandb_kwargs = {
            "entity": "research-interns",
            "project": cfg.WANDB.PROJECT,
            "config": {
                "learning_rate": cfg.TRAINING.LEARNING_RATE,
                "batch_size": cfg.DATASET.BATCH_SIZE,
                "grad_accum_steps": cfg.TRAINING.GRAD_ACCUM_STEPS,
                "effective_batch_size": cfg.DATASET.BATCH_SIZE * cfg.TRAINING.GRAD_ACCUM_STEPS,
                "num_epochs": cfg.TRAINING.NUM_EPOCHS,
                "m_frames": cfg.MODEL.M,
                "n_frames": cfg.MODEL.N,
                "future_frame_weight": cfg.LOSS.FUTURE_FRAME_WEIGHT,
                "pc_loss_weight": cfg.LOSS.PC_LOSS_WEIGHT,
                "pose_loss_weight": cfg.LOSS.POSE_LOSS_WEIGHT,
                "conf_loss_weight": cfg.LOSS.CONF_LOSS_WEIGHT,
                "use_conf_weighted_points": cfg.LOSS.USE_CONF_WEIGHTED_POINTS,
                "conf_gamma": cfg.LOSS.CONF_GAMMA,
                "conf_alpha": cfg.LOSS.CONF_ALPHA,
                "max_grad_norm": cfg.TRAINING.MAX_GRAD_NORM,
                "architecture": "AutonomyPi3",
                "optimizer": "Adam",
                "scheduler": "Warmup+CosineAnnealingLR",
                "warmup_steps": cfg.TRAINING.WARMUP_STEPS,
                "warmup_start_factor": cfg.TRAINING.WARMUP_START_FACTOR,
                "mixed_precision": "bf16",
                "val_split": cfg.DATASET.VAL_SPLIT,
                "val_freq": cfg.VALIDATION.VAL_FREQ,
                "val_samples": cfg.VALIDATION.VAL_SAMPLES,
                "early_stopping_patience": cfg.VALIDATION.EARLY_STOPPING_PATIENCE,
                "dataset_type": "S3" if cfg.DATASET.USE_S3 else "Local",
                "s3_bucket": cfg.DATASET.S3_BUCKET if cfg.DATASET.USE_S3 else None,
                "s3_preload_bytes": cfg.DATASET.S3_PRELOAD_BYTES if cfg.DATASET.USE_S3 else None,
                "save_images": cfg.LOGGING.SAVE_IMAGES,
                "save_images_steps": cfg.LOGGING.SAVE_IMAGES_STEPS,
            },
            "tags": ["pi3", "ssl", "cluster-training-fixed", "s3" if cfg.DATASET.USE_S3 else "local"]
        }
        
        # Add custom run name if specified
        if cfg.WANDB.RUN_NAME:
            wandb_kwargs["name"] = cfg.WANDB.RUN_NAME
            
        run = wandb.init(**wandb_kwargs)
        # Get the actual run name (either custom or auto-generated)
        actual_run_name = run.name if run else None
    else:
        run = None
        actual_run_name = None
    
    print("[training.train_model] Using configuration:")
    print(cfg)

    # Enhanced Accelerator setup with distributed safety
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.TRAINING.GRAD_ACCUM_STEPS,
        mixed_precision='bf16'  # Use bfloat16 for better stability and performance
    )
    
    # Wait for all processes to initialize
    distributed_barrier_with_timeout(accelerator, 60, "Accelerator initialization")
    
    if accelerator.is_main_process:
        print(f"🚀 Distributed training setup:")
        print(f"   World size: {accelerator.num_processes}")
        print(f"   Local process index: {accelerator.local_process_index}")
        print(f"   Device: {accelerator.device}")
    
    # Create dataset based on configuration
    if cfg.DATASET.USE_S3:
        print(f"🚀 Using S3 dataset from bucket: {cfg.DATASET.S3_BUCKET}")
        print(f"   Sequence prefixes: {cfg.DATASET.S3_SEQUENCE_PREFIXES}")
        print(f"   Image extension: {cfg.DATASET.S3_IMAGE_EXTENSION}")
        print(f"   AWS region: {cfg.DATASET.S3_REGION}")
        print(f"   Preload bytes: {cfg.DATASET.S3_PRELOAD_BYTES}")
        print(f"   ⏭️ Skipping first 300 frames per sequence, using frames 300+")
        
        # Create S3 dataset with new implementation
        full_dataset = S3Dataset(
            bucket_name=cfg.DATASET.S3_BUCKET,
            sequence_prefixes=cfg.DATASET.S3_SEQUENCE_PREFIXES,
            m=cfg.MODEL.M,
            n=cfg.MODEL.N,
            image_extension=cfg.DATASET.S3_IMAGE_EXTENSION,
            transform=get_default_transforms(),
            region_name=cfg.DATASET.S3_REGION,
            preload_bytes=cfg.DATASET.S3_PRELOAD_BYTES
        )
    else:
        print(f"📁 Using local dataset from: {cfg.DATASET.ROOT_DIR}")
        
        # Find all subdirectories in root_dir
        image_dirs = [os.path.join(cfg.DATASET.ROOT_DIR, d) for d in os.listdir(cfg.DATASET.ROOT_DIR)
                      if os.path.isdir(os.path.join(cfg.DATASET.ROOT_DIR, d))]
        print(f"Found {len(image_dirs)} subfolders:")
        for d in image_dirs:
            print(f"  {d}")

        # Create local dataset
        full_dataset = SequenceLearningDataset(
            image_dirs=image_dirs,
            m=cfg.MODEL.M,
            n=cfg.MODEL.N,
            transform=get_default_transforms())

    # Split dataset into train and validation
    if cfg.DATASET.VAL_SPLIT > 0:
        total_size = len(full_dataset)
        val_size = int(total_size * cfg.DATASET.VAL_SPLIT)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible split
        )
        
        print(f"📊 Dataset split: {train_size} train, {val_size} validation samples")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print(f"📊 Using full dataset for training: {len(train_dataset)} samples")
    
    # Print augmentation status
    if cfg.AUGMENTATION.USE_AUGMENTATIONS:
        print("🎨 Random augmentations enabled: color jittering, Gaussian blur, grayscale")
    else:
        print("🚫 Random augmentations disabled")

    # Smart DataLoader configuration optimized for multi-GPU training
    num_gpus = accelerator.num_processes if hasattr(accelerator, 'num_processes') else 1
    
    if cfg.DATASET.USE_S3:
        if cfg.DATASET.S3_PRELOAD_BYTES:
            # Maximum performance: bytes preloaded, scale workers with GPUs
            num_workers = min(8, 2 * num_gpus)  # 2 workers per GPU, max 8
            prefetch_factor = 16  # Aggressive prefetching for multi-GPU
            persistent_workers = True
            pin_memory = True
            print(f"🚀 Multi-GPU high-performance mode: {num_workers} workers, prefetch_factor={prefetch_factor} (S3 bytes preloaded)")
        else:
            # Balanced: scale workers with GPUs but keep reasonable for S3 credentials
            num_workers = 4
            prefetch_factor = 8  # Good prefetching for multi-GPU
            persistent_workers = True
            pin_memory = True
            print(f"⚖️ Multi-GPU balanced mode: {num_workers} workers, prefetch_factor={prefetch_factor} (S3 on-demand)")
    else:
        # Local dataset: scale with GPUs for file I/O
        num_workers = min(6, 2 * num_gpus)  # 2 workers per GPU, max 6
        prefetch_factor = 8
        persistent_workers = True
        pin_memory = True
        print(f"💾 Multi-GPU local dataset mode: {num_workers} workers, prefetch_factor={prefetch_factor}")
    
    print(f"   🖥️ Detected GPUs: {num_gpus}")
    print(f"   📊 DataLoader config: workers={num_workers}, prefetch={prefetch_factor}, persistent={persistent_workers}")
    print(f"   ⚡ Memory optimizations: pin_memory={pin_memory}")
    
    # Calculate effective throughput
    effective_workers = num_workers * num_gpus if num_gpus > 1 else num_workers
    print(f"   🚀 Effective worker processes across all GPUs: {effective_workers}")

    # Create optimized dataloaders with enhanced configuration
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,  # Ensure consistent batch sizes for training stability
        timeout=600 if num_workers > 0 else 0  # 10 min timeout for S3 downloads (increased)
    )
    
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.DATASET.BATCH_SIZE,
            shuffle=False,  # Don't shuffle validation
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,  # Keep all validation samples
            timeout=600 if num_workers > 0 else 0
        )
        
        print(f"📊 DataLoaders created:")
        print(f"   🎯 Train: {len(train_dataloader)} batches of size {cfg.DATASET.BATCH_SIZE}")
        print(f"   ✅ Val: {len(val_dataloader)} batches of size {cfg.DATASET.VAL_SPLIT}")
    else:
        print(f"📊 Train DataLoader created: {len(train_dataloader)} batches of size {cfg.DATASET.BATCH_SIZE}")

    print("Initializing and loading Pi3 model...")
    frozen_model = Pi3.from_pretrained("yyfz233/Pi3")
    frozen_model = frozen_model.to(accelerator.device)
    frozen_model.requires_grad_(False)  # freeze parameters

    # Define training model based on configuration
    if cfg.MODEL.ARCHITECTURE == "AutoregressivePi3":
        train_model = AutoregressivePi3(
            n_future_frames=cfg.MODEL.N,
            ar_n_heads=cfg.MODEL.AR_N_HEADS,
            ar_n_layers=cfg.MODEL.AR_N_LAYERS,
            ar_dropout=cfg.MODEL.AR_DROPOUT,
            encoder_name=cfg.MODEL.ENCODER_NAME.lower()
        )
    else:  # Default to AutonomyPi3
        train_model = AutonomyPi3(full_N=cfg.MODEL.M + cfg.MODEL.N, extra_tokens=cfg.MODEL.N)

    # Load encoder and rope
    train_model.encoder.load_state_dict(frozen_model.encoder.state_dict())
    train_model.rope.load_state_dict(frozen_model.rope.state_dict())

    # Load decoders
    train_model.decoder.load_state_dict(frozen_model.decoder.state_dict())
    train_model.point_decoder.load_state_dict(frozen_model.point_decoder.state_dict())
    train_model.conf_decoder.load_state_dict(frozen_model.conf_decoder.state_dict())
    train_model.camera_decoder.load_state_dict(frozen_model.camera_decoder.state_dict())
    
    # Load register token
    train_model.register_token.data.copy_(frozen_model.register_token.data)
    
    # Load point and conf heads (compatible parts only)
    frozen_point_dict = frozen_model.point_head.state_dict()
    train_point_dict = train_model.point_head.state_dict()
    matched_point_dict = {
        k: v for k, v in frozen_point_dict.items()
        if k in train_point_dict and v.shape == train_point_dict[k].shape
    }
    train_point_dict.update(matched_point_dict)
    train_model.point_head.load_state_dict(train_point_dict)
    
    frozen_conf_dict = frozen_model.conf_head.state_dict()
    train_conf_dict = train_model.conf_head.state_dict()
    matched_conf_dict = {
        k: v for k, v in frozen_conf_dict.items()
        if k in train_conf_dict and v.shape == train_conf_dict[k].shape
    }
    train_conf_dict.update(matched_conf_dict)
    train_model.conf_head.load_state_dict(train_conf_dict)
    
    # Load camera head (compatible parts only)
    frozen_camera_dict = frozen_model.camera_head.state_dict()
    train_camera_dict = train_model.camera_head.state_dict()
    matched_camera_dict = {
        k: v for k, v in frozen_camera_dict.items()
        if k in train_camera_dict and v.shape == train_camera_dict[k].shape
    }
    train_camera_dict.update(matched_camera_dict)
    train_model.camera_head.load_state_dict(train_camera_dict)
    
    # Load image normalization buffers
    train_model.image_mean.data.copy_(frozen_model.image_mean.data)
    train_model.image_std.data.copy_(frozen_model.image_std.data)

    train_model.point_head.current_proj.load_state_dict(frozen_model.point_head.proj.state_dict())
    train_model.conf_head.current_proj.load_state_dict(frozen_model.conf_head.proj.state_dict())

    # Define optimizer for train_model
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=cfg.TRAINING.LEARNING_RATE)
    
    # Create warmup + cosine annealing scheduler
    total_steps = len(train_dataloader) * cfg.TRAINING.NUM_EPOCHS
    warmup_steps = min(cfg.TRAINING.WARMUP_STEPS, total_steps // 10)  # Cap warmup at 10% of total steps
    cosine_steps = total_steps - warmup_steps
    
    if warmup_steps > 0:
        # Create warmup scheduler (linear increase from start_factor to 1.0)
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=cfg.TRAINING.WARMUP_START_FACTOR, 
            end_factor=1.0,
            total_iters=warmup_steps
        )
        # Create cosine annealing scheduler for after warmup
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)
        # Combine them with SequentialLR
        scheduler = SequentialLR(
            optimizer, 
            [warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_steps]
        )
        print(f"🔥 Using warmup scheduler: {warmup_steps} warmup steps + {cosine_steps} cosine annealing steps")
    else:
        # Fallback to original cosine annealing if no warmup
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        print(f"📊 Using cosine annealing scheduler: {total_steps} total steps")

    # Wait for all processes before preparing models
    distributed_barrier_with_timeout(accelerator, 120, "Model initialization")

    # Prepare training components with Accelerator
    if val_dataloader is not None:
        train_model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
            train_model,
            optimizer,
            scheduler,
            train_dataloader,
            val_dataloader
        )
    else:
        train_model, optimizer, scheduler, train_dataloader = accelerator.prepare(
            train_model,
            optimizer, 
            scheduler,
            train_dataloader
        )

    # Move frozen model manually to accelerator.device (but do NOT prepare it if you don't train it)
    frozen_model.to(accelerator.device)
    device = accelerator.device

    # Wait for all processes after model preparation
    distributed_barrier_with_timeout(accelerator, 120, "Model preparation")

    # Create checkpoint directory
    if accelerator.is_main_process:
        os.makedirs(cfg.OUTPUT.CHECKPOINT_DIR, exist_ok=True)

    # TensorBoard SummaryWriter
    if accelerator.is_main_process:
        writer = SummaryWriter("runs/pi3_cluster_fixed")

    # Training loop
    global_step = 0
    total_step = 0
    best_loss = float('inf')
    best_val_loss = float('inf')
    running_loss = 0.0
    loss_history = []
    val_loss_history = []
    steps_without_improvement = 0

    # Save sample images at the start (only main process)
    if accelerator.is_main_process:
        print("🖼️ Saving sample training images...")
        try:
            from PIL import Image as PILImage
            
            sample_dir = "./training_samples"
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save a few random training samples (original and augmented)
            sample_batch = next(iter(train_dataloader))
            X_sample, y_sample = sample_batch
            X_all_sample = torch.cat([X_sample, y_sample], dim=1)  # (B, T, C, H, W)
            
            # Save original images first
            sample_tensor_orig = preprocess_image(X_all_sample[0]).cpu()  # (T, C, H, W)
            
            for frame_idx in range(min(6, sample_tensor_orig.shape[0])):  # Save first 6 frames
                img_tensor = sample_tensor_orig[frame_idx]  # (C, H, W)
                img_array = img_tensor.permute(1, 2, 0).numpy()  # (H, W, C)
                img_array = np.clip(img_array, 0, 1)
                img_array = (img_array * 255).astype(np.uint8)
                
                pil_image = PILImage.fromarray(img_array)
                frame_type = "input" if frame_idx < cfg.MODEL.M else "target"
                local_idx = frame_idx if frame_idx < cfg.MODEL.M else frame_idx - cfg.MODEL.M
                save_path = os.path.join(sample_dir, f"original_{frame_type}_frame_{local_idx:02d}.png")
                pil_image.save(save_path)
            
            # Save augmented images if augmentations are enabled
            if cfg.AUGMENTATION.USE_AUGMENTATIONS:
                # Apply augmentations to the sample
                X_all_sample_aug = apply_random_augmentations(X_all_sample[0], training=True)  # (T, C, H, W)
                sample_tensor_aug = preprocess_image(X_all_sample_aug).cpu()  # (T, C, H, W)
                
                for frame_idx in range(min(6, sample_tensor_aug.shape[0])):  # Save first 6 frames
                    img_tensor = sample_tensor_aug[frame_idx]  # (C, H, W)
                    img_array = img_tensor.permute(1, 2, 0).numpy()  # (H, W, C)
                    img_array = np.clip(img_array, 0, 1)
                    img_array = (img_array * 255).astype(np.uint8)
                    
                    pil_image = PILImage.fromarray(img_array)
                    frame_type = "input" if frame_idx < cfg.MODEL.M else "target"
                    local_idx = frame_idx if frame_idx < cfg.MODEL.M else frame_idx - cfg.MODEL.M
                    save_path = os.path.join(sample_dir, f"augmented_{frame_type}_frame_{local_idx:02d}.png")
                    pil_image.save(save_path)
            
            if cfg.AUGMENTATION.USE_AUGMENTATIONS:
                print(f"✅ Saved original and augmented sample images to {sample_dir}")
            else:
                print(f"✅ Saved original sample images to {sample_dir}")
            
        except Exception as e:
            print(f"⚠️ Could not save sample images: {e}")

    # Wait for all processes before starting training
    distributed_barrier_with_timeout(accelerator, 60, "Pre-training initialization")

    for epoch in range(cfg.TRAINING.NUM_EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch+1}/{cfg.TRAINING.NUM_EPOCHS}", 
            disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            if accelerator.is_main_process and step % 100 == 0:
                print(f"🔄 Step {step}, Global step {global_step}")
            
            try:
                with accelerator.accumulate(train_model):
                    with gpu_memory_guard(accelerator, f"Training step {global_step}"):
                        
                        # Process batch properly for any batch size
                        X = batch[0]  # (B, m, C, H, W) - current frames
                        y = batch[1]  # (B, n, C, H, W) - future frames
                        X_all = torch.cat([X, y], dim=1)  # (B, T, C, H, W) where T = m + n
                        
                        save_batch_images_to_png(X_all, global_step, cfg)
                        
                        # Create unaugmented tensor for frozen model (ground truth)
                        batch_size = X_all.shape[0]
                        if batch_size == 1:
                            # Optimize for batch_size=1 (most common case)
                            video_tensor_unaugmented = preprocess_image(X_all[0]).unsqueeze(0)  # (1, T, C, H, W)
                        else:
                            # Handle larger batch sizes
                            video_tensors_unaugmented = []
                            for b in range(batch_size):
                                # Get single sample: (T, C, H, W)
                                sample = X_all[b]  # (T, C, H, W)
                                # Preprocess this sample without augmentations
                                processed_sample = preprocess_image(sample)  # (T, C, H, W)
                                video_tensors_unaugmented.append(processed_sample)
                            
                            # Stack to create batch: (B, T, C, H, W)
                            video_tensor_unaugmented = torch.stack(video_tensors_unaugmented, dim=0)
                        
                        # Apply augmentations to each sample in the batch for training model
                        augmented_samples = []
                        for b in range(batch_size):
                            sample = X_all[b]  # (T, C, H, W)
                            # Apply random augmentations (different for each image in sequence)
                            augmented_sample = apply_random_augmentations(sample, training=cfg.AUGMENTATION.USE_AUGMENTATIONS)
                            augmented_samples.append(augmented_sample)
                        
                        # Stack back to batch format
                        X_all_augmented = torch.stack(augmented_samples, dim=0)  # (B, T, C, H, W)
                        
                        # Process augmented samples for training model
                        if batch_size == 1:
                            # Optimize for batch_size=1 (most common case)
                            video_tensor_augmented = preprocess_image(X_all_augmented[0]).unsqueeze(0)  # (1, T, C, H, W)
                        else:
                            # Handle larger batch sizes
                            video_tensors_augmented = []
                            for b in range(batch_size):
                                # Get single sample: (T, C, H, W)
                                sample = X_all_augmented[b]  # (T, C, H, W)
                                # Preprocess this augmented sample
                                processed_sample = preprocess_image(sample)  # (T, C, H, W)
                                video_tensors_augmented.append(processed_sample)
                            
                            # Stack to create batch: (B, T, C, H, W)
                            video_tensor_augmented = torch.stack(video_tensors_augmented, dim=0)
                        
                        subset_video_tensor = video_tensor_augmented[:, :cfg.MODEL.M]  # (B, m, C, H, W) - augmented for training
                        dtype = torch.bfloat16

                        with torch.no_grad():
                            with torch.amp.autocast('cuda', dtype=dtype):
                                pseudo_gt = frozen_model(video_tensor_unaugmented)  # Use unaugmented data for ground truth
                        
                        # Clear cache after frozen model inference
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                        # run inference on the training model
                        with torch.amp.autocast('cuda', dtype=dtype):
                            predictions = train_model(subset_video_tensor)
                    
                        # compute loss between prediction and pseudo_gt with optional confidence weighting
                        # Use FP32 for loss computation if enabled for better numerical stability
                        loss_dtype = torch.float32 if cfg.TRAINING.USE_FP32_FOR_LOSSES else dtype
                        
                        with torch.amp.autocast('cuda', dtype=loss_dtype):
                            if cfg.LOSS.USE_CONF_WEIGHTED_POINTS:
                                point_map_loss, camera_pose_loss, conf_loss = Pi3Losses.pi3_loss_with_confidence_weighting(
                                    predictions, pseudo_gt, m_frames=cfg.MODEL.M, future_frame_weight=cfg.LOSS.FUTURE_FRAME_WEIGHT,
                                    gamma=cfg.LOSS.CONF_GAMMA, alpha=cfg.LOSS.CONF_ALPHA, use_conf_weighted_points=True, gradient_weight=cfg.LOSS.GRADIENT_WEIGHT
                                )
                            else:
                                point_map_loss, camera_pose_loss, conf_loss = Pi3Losses.pi3_loss(
                                    predictions, pseudo_gt, m_frames=cfg.MODEL.M, future_frame_weight=cfg.LOSS.FUTURE_FRAME_WEIGHT, gradient_weight=cfg.LOSS.GRADIENT_WEIGHT
                                )
                        
                        # Check for NaNs in individual loss components (only if detection is enabled)
                        nan_detected = False
                        if cfg.TRAINING.DETECT_NANS:
                            nan_detected |= check_for_nans(point_map_loss, "point_map_loss", global_step)
                            nan_detected |= check_for_nans(camera_pose_loss, "camera_pose_loss", global_step)
                            if not cfg.LOSS.USE_CONF_WEIGHTED_POINTS:
                                nan_detected |= check_for_nans(conf_loss, "conf_loss", global_step)
                        
                        pi3_loss = (cfg.LOSS.PC_LOSS_WEIGHT * point_map_loss) + (cfg.LOSS.POSE_LOSS_WEIGHT * camera_pose_loss) + (cfg.LOSS.CONF_LOSS_WEIGHT * conf_loss)
                        
                        # Check final loss for NaNs
                        if cfg.TRAINING.DETECT_NANS:
                            nan_detected |= check_for_nans(pi3_loss, "pi3_loss", global_step)
                            
                            if nan_detected:
                                print(f"🚨 NaN detected at training step {global_step}! Skipping this batch...")
                                # Skip backward pass for this batch
                                continue

                        # Enhanced backward pass with synchronization safety
                        try:
                            accelerator.backward(pi3_loss)
                            
                            if accelerator.sync_gradients:
                                # Check for NaN gradients before clipping (only if detection enabled)
                                if cfg.TRAINING.DETECT_NANS and check_model_parameters(train_model, "train_model", global_step):
                                    print(f"🚨 NaN gradients detected at step {global_step}! Skipping optimizer step...")
                                    optimizer.zero_grad()  # Clear gradients and continue
                                    continue
                                
                                # Sync gradients safely with timeout
                                accelerator.clip_grad_norm_(train_model.parameters(), cfg.TRAINING.MAX_GRAD_NORM)
                                
                                # Check for NaN gradients after clipping (only if detection enabled)
                                if cfg.TRAINING.DETECT_NANS and check_model_parameters(train_model, "train_model_after_clip", global_step):
                                    print(f"🚨 NaN gradients after clipping at step {global_step}! Skipping optimizer step...")
                                    optimizer.zero_grad()  # Clear gradients and continue
                                    continue
                            
                            # Safe optimizer step
                            optimizer.step()
                            
                            # Check model parameters after optimizer step (only if detection enabled)
                            if cfg.TRAINING.DETECT_NANS and check_model_parameters(train_model, "train_model_after_step", global_step):
                                print(f"🚨 NaN model parameters after optimizer step {global_step}! This indicates a serious numerical issue...")
                            
                            scheduler.step()
                            optimizer.zero_grad()
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                print(f"💥 GPU OOM during backward pass at step {global_step}: {e}")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()
                                optimizer.zero_grad()  # Clear gradients
                                continue  # Skip this batch
                            elif "nccl" in str(e).lower() or "communication" in str(e).lower():
                                print(f"🔄 NCCL/Communication error at step {global_step}: {e}")
                                # Attempt to recover from communication failure
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                optimizer.zero_grad()
                                # Try to re-establish communication with barrier
                                if not distributed_barrier_with_timeout(accelerator, 30, "Recovery barrier"):
                                    print(f"⚠️ Failed to recover communication, continuing...")
                                continue  # Skip this batch but continue training
                            elif "deadlock" in str(e).lower() or "hang" in str(e).lower():
                                print(f"🚨 Potential deadlock detected at step {global_step}: {e}")
                                # Force synchronization and cleanup
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()
                                optimizer.zero_grad()
                                continue  # Skip this batch
                            else:
                                raise

                # Store loss value immediately and delete large tensors
                current_loss = pi3_loss.detach().item()
                
                # Aggressive memory cleanup after optimization
                if accelerator.sync_gradients and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                # Process additional outputs (unchanged from original)
                points = pseudo_gt["local_points"]
                masks = torch.sigmoid(pseudo_gt["conf"][..., 0]) > 0.1
                original_height, original_width = points.shape[-3:-1]
                aspect_ratio = original_width / original_height

                pseudo_gt['images'] = video_tensor_unaugmented.permute(0, 1, 3, 4, 2)
                pseudo_gt['conf'] = torch.sigmoid(pseudo_gt['conf'])
                edge = depth_edge(pseudo_gt['local_points'][..., 2], rtol=0.03)
                pseudo_gt['conf'][edge] = 0.0

                # Save NPZ files if enabled (only main process)
                if cfg.OUTPUT.SAVE_NPZ and accelerator.is_main_process:
                    # get copy and detach
                    scene_rgb = video_tensor_unaugmented.cpu().numpy()[0]
                    conf = pseudo_gt['conf'].cpu().numpy().copy()
                    # remove last dim
                    conf = conf[..., 0]

                    depths = pseudo_gt['local_points'][..., 2].cpu().numpy().copy()
                    intrinsics = pseudo_gt['intrinsics'].cpu().numpy().copy() if 'intrinsics' in pseudo_gt else None

                    data_npz_load = {}
                    # repat the first cam pose n times
                    cam_poses = pseudo_gt['camera_poses'].cpu().numpy().copy()
                    repeated_poses = np.tile(cam_poses[0], (20, 1, 1))

                    data_npz_load["extrinsics"] = repeated_poses
                    data_npz_load["intrinsics"] = intrinsics
                    # depth_save = points_map[:,2,...]
                    depths[conf<0.5] = 0
                    data_npz_load["depths"] = depths
                    data_npz_load["video"] = scene_rgb
                    # save the data_npz_load to a npz file
                    output_file = f"output_{total_step}.npz"
                    total_step += 1
                    np.savez(output_file, **data_npz_load)
                    print(f"Saved data to {output_file}")

                # Save depth visualizations if enabled (only main process)
                if cfg.OUTPUT.SAVE_DEPTHS and 'local_points' in predictions and global_step % 200 == 0 and accelerator.is_main_process:
                    # Convert tensors to numpy
                    for key in pseudo_gt.keys():
                        if isinstance(pseudo_gt[key], torch.Tensor):
                            pseudo_gt[key] = pseudo_gt[key].cpu().numpy().squeeze(0)  # remove batch dimension

                    for key in predictions.keys():
                        if isinstance(predictions[key], torch.Tensor):
                            predictions[key] = predictions[key].clone().detach().cpu().numpy().squeeze(0)  # remove batch dimension

                    import matplotlib.pyplot as plt
                    import imageio
                    
                    # === DEPTH VISUALIZATION ===
                    local_points = predictions['local_points']  # shape (T, H, W, 3)
                    depth_maps = local_points[..., 2]  # shape (T, H, W)
                    depth_images_for_wandb = []
                    
                    for t in range(depth_maps.shape[0]):
                        depth = depth_maps[t]
                        vmin, vmax = np.percentile(depth, 2), np.percentile(depth, 98)
                        norm_depth = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
                        # Apply viridis colormap
                        colored = plt.get_cmap('viridis')(norm_depth)[:, :, :3]  # shape (H, W, 3), drop alpha
                        colored_uint8 = (colored * 255).astype(np.uint8)
                        
                        # Save to disk
                        imageio.imwrite(f"depth_frame_{t}_viridis.png", colored_uint8)
                        print(f"Saved depth map for frame {t} to depth_frame_{t}_viridis.png")
                        
                        # Prepare for WandB
                        if cfg.WANDB.USE_WANDB:
                            from PIL import Image as PILImage
                            pil_image = PILImage.fromarray(colored_uint8)
                            depth_images_for_wandb.append(wandb.Image(pil_image, caption=f"Depth Frame {t}"))
                    
                    # === CONFIDENCE VISUALIZATION ===
                    confidence_images_for_wandb = []
                    if 'conf' in predictions:
                        conf_maps = predictions['conf']  # shape (T, H, W, 1) or (T, H, W)
                        if conf_maps.ndim == 4:  # (T, H, W, 1)
                            conf_maps = conf_maps.squeeze(-1)  # (T, H, W)
                        
                        for t in range(conf_maps.shape[0]):
                            conf = conf_maps[t]  # (H, W)
                            # Apply sigmoid if values are not in [0,1] range
                            if conf.min() < 0 or conf.max() > 1:
                                conf = 1 / (1 + np.exp(-conf))  # sigmoid
                            
                            # Apply hot colormap for confidence (red=high confidence, blue=low confidence)
                            colored = plt.get_cmap('hot')(conf)[:, :, :3]  # shape (H, W, 3), drop alpha
                            colored_uint8 = (colored * 255).astype(np.uint8)
                            
                            # Save to disk
                            imageio.imwrite(f"confidence_frame_{t}_hot.png", colored_uint8)
                            print(f"Saved confidence map for frame {t} to confidence_frame_{t}_hot.png")
                            
                            # Prepare for WandB
                            if cfg.WANDB.USE_WANDB:
                                from PIL import Image as PILImage
                                pil_image = PILImage.fromarray(colored_uint8)
                                confidence_images_for_wandb.append(wandb.Image(pil_image, caption=f"Confidence Frame {t}"))
                    
                    # === LOG TO WANDB ===
                    if cfg.WANDB.USE_WANDB:
                        wandb_log_dict = {}
                        
                        # Log depth images
                        if depth_images_for_wandb:
                            wandb_log_dict["visualizations/depth_maps"] = depth_images_for_wandb
                            print(f"🚀 Logged {len(depth_images_for_wandb)} depth maps to WandB")
                        
                        # Log confidence images  
                        if confidence_images_for_wandb:
                            wandb_log_dict["visualizations/confidence_maps"] = confidence_images_for_wandb
                            print(f"🚀 Logged {len(confidence_images_for_wandb)} confidence maps to WandB")
                        
                        # Send all visualizations to WandB
                        if wandb_log_dict:
                            run.log(wandb_log_dict, step=global_step)

                epoch_loss += current_loss
                running_loss += current_loss
                loss_history.append(current_loss)
                    
                # Enhanced logging with distributed safety
                if global_step % cfg.LOGGING.LOG_FREQ == 0 and accelerator.is_main_process:
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # TensorBoard logging
                    writer.add_scalar("Loss/Train", pi3_loss.item(), global_step)
                    writer.add_scalar("Learning_Rate", current_lr, global_step)
                    
                    # Weights & Biases logging
                    if cfg.WANDB.USE_WANDB:
                        log_dict = {
                            "train/total_loss": pi3_loss.item(),
                            "train/point_map_loss": point_map_loss.item(),
                            "train/camera_pose_loss": camera_pose_loss.item(),
                            "train/conf_loss": conf_loss.item() if torch.is_tensor(conf_loss) else conf_loss,
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch,
                            "train/best_loss": best_loss,
                            "train/step": global_step
                        }
                        # Add warmup phase indicator if using warmup
                        if warmup_steps > 0:
                            log_dict["train/warmup_phase"] = 1.0 if global_step < warmup_steps else 0.0
                        run.log(log_dict, step=global_step)
                    
                    postfix_dict = {
                        'loss': f'{pi3_loss.item():.6f}',
                        'lr': f'{current_lr:.2e}',
                        'best': f'{best_loss:.6f}',
                        'val_best': f'{best_val_loss:.6f}' if val_dataloader else 'N/A'
                    }
                    # Add warmup indicator if we're in warmup phase
                    if warmup_steps > 0 and global_step < warmup_steps:
                        postfix_dict['warmup'] = f'{global_step}/{warmup_steps}'
                    progress_bar.set_postfix(postfix_dict)
                
                # Enhanced validation with distributed safety
                if (val_dataloader is not None and 
                    global_step % cfg.VALIDATION.VAL_FREQ == 0 and 
                    global_step > 0 and 
                    accelerator.is_main_process):
                    
                    print(f"\n🔍 Running distributed-safe validation at step {global_step}...")
                    
                    # Use enhanced validation function
                    val_metrics = run_validation_safe(
                        train_model, frozen_model, val_dataloader, cfg, accelerator, dtype
                    )
                    
                    if val_metrics is not None:
                        val_loss_history.append(val_metrics['val_loss'])
                        
                        # Log validation metrics
                        writer.add_scalar("Loss/Validation", val_metrics['val_loss'], global_step)
                        writer.add_scalar("Loss/Val_Point", val_metrics['val_point_loss'], global_step)
                        writer.add_scalar("Loss/Val_Camera", val_metrics['val_camera_loss'], global_step)
                        writer.add_scalar("Loss/Val_Unweighted_L1_Points", val_metrics['val_unweighted_l1_points'], global_step)
                        writer.add_scalar("Loss/Val_Unweighted_Pose", val_metrics['val_unweighted_pose_loss'], global_step)
                        
                        if cfg.WANDB.USE_WANDB:
                            run.log({
                                "val/total_loss": val_metrics['val_loss'],
                                "val/point_map_loss": val_metrics['val_point_loss'],
                                "val/camera_pose_loss": val_metrics['val_camera_loss'],
                                "val/conf_loss": val_metrics['val_conf_loss'],
                                "val/unweighted_l1_points": val_metrics['val_unweighted_l1_points'],
                                "val/unweighted_pose_loss": val_metrics['val_unweighted_pose_loss'],
                                "val/num_samples": val_metrics['num_samples'],
                                "val/step": global_step
                            }, step=global_step)
                        
                        print(f"📊 Validation Results:")
                        print(f"   Total Loss (weighted): {val_metrics['val_loss']:.6f}")
                        print(f"   Point Loss (weighted): {val_metrics['val_point_loss']:.6f}")
                        print(f"   Camera Loss (weighted): {val_metrics['val_camera_loss']:.6f}")
                        print(f"   Conf Loss (weighted): {val_metrics['val_conf_loss']:.6f}")
                        print(f"   L1 Points (unweighted): {val_metrics['val_unweighted_l1_points']:.6f}")
                        print(f"   Pose Loss (unweighted): {val_metrics['val_unweighted_pose_loss']:.6f}")
                        print(f"   Samples: {val_metrics['num_samples']}")
                        
                        # Early stopping check
                        if val_metrics['val_loss'] < best_val_loss:
                            best_val_loss = val_metrics['val_loss']
                            steps_without_improvement = 0
                            
                            # Save best validation model
                            best_val_checkpoint = {
                                'epoch': epoch,
                                'global_step': global_step,
                                'model_state_dict': accelerator.unwrap_model(train_model).state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_val_loss': best_val_loss,
                                'val_metrics': val_metrics,
                                'loss_history': loss_history,
                                'val_loss_history': val_loss_history,
                                'config': cfg
                            }
                            
                            best_val_model_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, 'best_val_model.pt')
                            torch.save(best_val_checkpoint, best_val_model_path)
                            print(f"💾 New best validation model saved! Val Loss: {best_val_loss:.6f}")
                            
                            # Upload best validation model to S3
                            upload_success = False
                            if cfg.OUTPUT.UPLOAD_TO_S3:
                                try:
                                    s3_filename = f"{actual_run_name}_best_val_model.pt" if actual_run_name else "best_val_model.pt"
                                    s3_path = f"s3://{cfg.OUTPUT.S3_BUCKET}/{cfg.OUTPUT.S3_PREFIX}/{s3_filename}"
                                    save_state_dict_to_s3(best_val_checkpoint, s3_path)
                                    upload_success = True
                                except Exception as e:
                                    print(f"❌ Failed to upload best val model to S3: {e}")
                                    upload_success = False
                            
                            if cfg.WANDB.USE_WANDB:
                                run.log({
                                    "val/best_model_saved": True,
                                    "val/new_best_loss": best_val_loss,
                                    "val/s3_upload_success": upload_success
                                }, step=global_step)
                        else:
                            steps_without_improvement += 1
                            print(f"⚠️  No validation improvement. Steps without improvement: {steps_without_improvement}/{cfg.VALIDATION.EARLY_STOPPING_PATIENCE}")
                        
                        # Early stopping
                        if cfg.VALIDATION.EARLY_STOPPING_PATIENCE > 0 and steps_without_improvement >= cfg.VALIDATION.EARLY_STOPPING_PATIENCE:
                            print(f"🛑 Early stopping triggered after {steps_without_improvement} validation checks without improvement.")
                            if cfg.WANDB.USE_WANDB:
                                run.log({
                                    "training/early_stopped": True,
                                    "training/final_step": global_step,
                                    "training/final_val_loss": val_metrics['val_loss']
                                }, step=global_step)
                    else:
                        print("⚠️ Validation failed - continuing training")
                
                # Model saving based on best loss (only main process)
                if global_step % cfg.LOGGING.SAVE_FREQ == 0 and global_step != 0 and accelerator.is_main_process:
                    # Calculate average loss over the last save_freq steps
                    recent_loss = running_loss / cfg.LOGGING.SAVE_FREQ
                    
                    # Check if this is the best loss so far
                    if recent_loss < best_loss:
                        best_loss = recent_loss
                        
                        # Save the best model
                        checkpoint = {
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': accelerator.unwrap_model(train_model).state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_loss': best_loss,
                            'best_val_loss': best_val_loss,
                            'loss_history': loss_history,
                            'val_loss_history': val_loss_history,
                            'steps_without_improvement': steps_without_improvement,
                            'config': cfg
                        }
                        
                        best_model_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, 'best_model.pt')
                        torch.save(checkpoint, best_model_path)
                        print(f"💾 New best model saved! Loss: {best_loss:.6f} at step {global_step}")
                        
                        # Upload best training model to S3
                        upload_success = False
                        if cfg.OUTPUT.UPLOAD_TO_S3:
                            try:
                                s3_filename = f"{actual_run_name}_best_model.pt" if actual_run_name else "best_model.pt"
                                s3_path = f"s3://{cfg.OUTPUT.S3_BUCKET}/{cfg.OUTPUT.S3_PREFIX}/{s3_filename}"
                                save_state_dict_to_s3(checkpoint, s3_path)
                                upload_success = True
                            except Exception as e:
                                print(f"❌ Failed to upload best model to S3: {e}")
                                upload_success = False
                        
                        # Log best model to wandb
                        if cfg.WANDB.USE_WANDB:
                            run.log({
                                "train/best_model_saved": True,
                                "train/new_best_loss": best_loss,
                                "train/s3_upload_success": upload_success
                            }, step=global_step)
                    
                    # Always save a recent checkpoint
                    recent_checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': accelerator.unwrap_model(train_model).state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'current_loss': recent_loss,
                        'best_loss': best_loss,
                        'best_val_loss': best_val_loss,
                        'loss_history': loss_history,
                        'val_loss_history': val_loss_history,
                        'steps_without_improvement': steps_without_improvement,
                        'config': cfg
                    }
                    
                    recent_model_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, f'checkpoint_step_{global_step}.pt')
                    torch.save(recent_checkpoint, recent_model_path)
                    
                    # Keep only the last 3 checkpoints to save disk space
                    checkpoint_files = sorted([f for f in os.listdir(cfg.OUTPUT.CHECKPOINT_DIR) if f.startswith('checkpoint_step_')])
                    if len(checkpoint_files) > 3:
                        for old_checkpoint in checkpoint_files[:-3]:
                            os.remove(os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, old_checkpoint))
                    
                    # Reset running loss
                    running_loss = 0.0
                
                global_step += 1
                
                # Final memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"⚠️ Error in training step {global_step}: {e}")
                
                # Clear GPU memory on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Continue to next step
                continue

        # Break out of epoch loop if early stopping was triggered
        if (cfg.VALIDATION.EARLY_STOPPING_PATIENCE > 0 and 
            steps_without_improvement >= cfg.VALIDATION.EARLY_STOPPING_PATIENCE):
            break

        # Wait for all processes at end of epoch
        distributed_barrier_with_timeout(accelerator, 60, f"End of epoch {epoch}")

        # End of epoch logging (only main process)
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        if accelerator.is_main_process:
            print(f"📊 Epoch {epoch+1} completed - Avg Loss: {avg_epoch_loss:.6f}")
            if val_dataloader is not None:
                print(f"   Best Val Loss so far: {best_val_loss:.6f}")
                print(f"   Steps without improvement: {steps_without_improvement}")
            
            # Log epoch metrics
            if cfg.WANDB.USE_WANDB:
                epoch_log = {
                    "epoch/avg_loss": avg_epoch_loss,
                    "epoch/number": epoch + 1,
                    "epoch/best_loss_so_far": best_loss
                }
                if val_dataloader is not None:
                    epoch_log.update({
                        "epoch/best_val_loss_so_far": best_val_loss,
                        "epoch/steps_without_improvement": steps_without_improvement
                    })
                run.log(epoch_log, step=global_step)

    # Training complete - cleanup
    writer.close()
    
    if cfg.WANDB.USE_WANDB:
        run.finish()
        print("📊 Weights & Biases run finished!")
    
    # Final summary (only main process)
    if accelerator.is_main_process:
        print("🎉 Training complete!")
        print(f"📊 Best training loss achieved: {best_loss:.6f}")
        if val_dataloader is not None:
            print(f"📊 Best validation loss achieved: {best_val_loss:.6f}")
            print(f"💾 Best validation model saved at: {os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, 'best_val_model.pt')}")
            if steps_without_improvement >= cfg.VALIDATION.EARLY_STOPPING_PATIENCE:
                print(f"🛑 Training stopped early due to no validation improvement for {steps_without_improvement} checks")
        print(f"💾 Best training model saved at: {os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, 'best_model.pt')}")
        print(f"📈 Total training steps: {global_step}")
        
        # Save final training summary
        summary = {
            'final_loss': avg_epoch_loss,
            'best_loss': best_loss,
            'best_val_loss': best_val_loss,
            'total_steps': global_step,
            'loss_history': loss_history,
            'val_loss_history': val_loss_history,
            'steps_without_improvement': steps_without_improvement,
            'early_stopped': steps_without_improvement >= cfg.VALIDATION.EARLY_STOPPING_PATIENCE,
            'training_config': cfg
        }
        summary_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)  # default=str handles non-serializable objects
        print(f"📋 Training summary saved at: {summary_path}")
        
        # Upload final training summary to S3
        if cfg.OUTPUT.UPLOAD_TO_S3:
            upload_file_to_s3(
                summary_path,
                s3_bucket=cfg.OUTPUT.S3_BUCKET,
                s3_prefix=cfg.OUTPUT.S3_PREFIX,
                wandb_run_name=actual_run_name
            )

# ================================================================================================
# Rest of the functions (unchanged from original)
# ================================================================================================

def init_distributed(seed=42, train_on_cluster=False):
    """Initialize distributed training environment and set random seeds for reproducibility."""
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    if not train_on_cluster:
        import torch.distributed as dist
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=3600)
        )
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

def train_with_config(config_path=None):
    """Train model using a YAML configuration file."""
    if config_path:
        # Create a temporary args object for custom config
        class Args:
            config = config_path
            opts = []
        cfg = get_cfg_defaults()
        cfg = update_config(cfg, Args())
    
    train_model()

def train_with_tune_config(tune_config):
    """Train model using Ray Tune configuration."""
    train_model(train_config=tune_config)

def lilypad_train_model(**kwargs):
    """Lilypad-compatible training function that handles accelerate initialization."""
    import os
    import torch
    
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"🔍 Detected {num_gpus} GPU(s)")
    
    # Check if we're already in an accelerate environment
    if not is_accelerate_launched():
        print("🚀 Initializing accelerate environment for Lilypad...")
        
        # Set accelerate environment variables for distributed training
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = str(num_gpus)  # Use all available GPUs
        if 'RANK' not in os.environ:
            os.environ['RANK'] = '0'  # Main process rank
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = '0'  # Local rank on this node
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
            
        # For multi-GPU, we need to spawn processes
        if num_gpus > 1:
            print(f"🚀 Setting up for {num_gpus} GPUs...")
            # Set up for DataParallel mode
            os.environ['ACCELERATE_USE_FP16'] = 'false'
            os.environ['ACCELERATE_USE_BF16'] = 'true'
            
        print(f"   🌍 WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
        print(f"   📍 RANK: {os.environ.get('RANK')}")
        print(f"   🏠 LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    else:
        print("✅ Already in accelerate environment")
    
    # Call the main training function
    return train_model(train_config=kwargs)

def main():
    """Main function with argument parsing for configuration."""
    parser = argparse.ArgumentParser(description="Pi3 Cluster Training with Enhanced GPU Synchronization")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file (optional, defaults to config.yaml)"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--use_accelerate", 
        action='store_true',
        default=True,
        help="Use accelerate launcher (default: True)"
    )

    args = parser.parse_args()
    
    # Load and update configuration
    cfg = get_cfg_defaults()
    cfg = update_config(cfg, args)
    
    print("==> Final configuration:")
    print(cfg)
    train_model()

def is_accelerate_launched():
    """Check if script is already launched with accelerate."""
    import os
    return 'ACCELERATE_USE_FP16' in os.environ or 'ACCELERATE_USE_FSDP' in os.environ or 'LOCAL_RANK' in os.environ

def launch_with_accelerate():
    """Launch the script with accelerate."""
    import subprocess
    import sys
    
    # Get current script arguments
    current_args = sys.argv[1:]  # Exclude script name
    
    # Remove --use_accelerate flag to avoid recursion
    filtered_args = [arg for arg in current_args if arg != '--use_accelerate']
    
    # Build accelerate command
    cmd = ['accelerate', 'launch'] + [sys.argv[0]] + filtered_args
    
    print(f"🔄 Executing: {' '.join(cmd)}")
    
    # Execute with accelerate
    subprocess.run(cmd)

if __name__ == "__main__":
    main()