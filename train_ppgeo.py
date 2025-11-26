"""
PPGeo-style self-supervised depth and pose training using the YouTube S3 dataset.
Based on train_cluster.py dataset but with PPGeo's photometric self-supervision approach.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
import wandb
from tqdm import tqdm
import numpy as np
import time
import gc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

# Import datasets and config
from ppgeo_dataset import PPGeoDataset
from config.defaults import get_cfg_defaults, update_config

# PPGeo components
from ppgeo_model import PPGeoModel
from ppgeo_losses import PPGeoLoss
from ppgeo_motionnet import MotionNet


class SkipBatchSampler:
    """Custom sampler that starts from a specific batch index."""
    def __init__(self, data_source, start_batch_idx=0, batch_size=1, shuffle=True, generator=None):
        self.data_source = data_source
        self.start_batch_idx = start_batch_idx
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        
    def __iter__(self):
        n = len(self.data_source)
        if self.shuffle:
            if self.generator is None:
                generator = torch.Generator()
                generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            else:
                generator = self.generator
            
            indices = torch.randperm(n, generator=generator).tolist()
        else:
            indices = list(range(n))
        
        # Calculate starting index based on start_batch_idx and batch_size
        start_idx = self.start_batch_idx * self.batch_size
        
        # Skip to the correct starting position
        if start_idx < len(indices):
            indices = indices[start_idx:]
        else:
            indices = []  # All batches already processed
        
        # Group into batches
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]
    
    def __len__(self):
        n = len(self.data_source)
        start_idx = self.start_batch_idx * self.batch_size
        remaining = max(0, n - start_idx)
        return (remaining + self.batch_size - 1) // self.batch_size


def create_ppgeo_model(cfg):
    """Create PPGeo model with configurable encoder (ViT or ResNet)."""
    if cfg.PPGEO.STAGE == 1:
        scales = [0, 1, 2, 3]  # Multi-scale for stage 1
    else:
        scales = [0]  # Only full resolution for stage 2
        
    model = PPGeoModel(
        encoder_name=getattr(cfg.PPGEO, 'ENCODER', "dinov3"),  # Default to ViT
        img_size=(cfg.DATASET.IMG_HEIGHT, cfg.DATASET.IMG_WIDTH),
        min_depth=0.1,
        max_depth=100.0,
        scales=scales,
        resnet_layers=getattr(cfg.PPGEO, 'RESNET_LAYERS', 18)  # Default ResNet-18
    )
    return model


def visualize_dataset_sample(dataset, idx=0, save_path="ppgeo_sample_visualization.png"):
    """Visualize a sample from the PPGeo dataset including multiple frames."""
    print("🎨 Creating quick visualization of dataset sample...")
    
    # Get a sample
    sample = dataset[idx]
    
    # Figure out how many frames we have
    frame_ids = []
    for key in sample.keys():
        if isinstance(key, tuple) and key[0] == 'color' and key[2] == 0:  # Original scale
            frame_ids.append(key[1])
    frame_ids = sorted(set(frame_ids))
    
    # Create figure with subplots
    n_frames = len(frame_ids)
    fig, axes = plt.subplots(2, n_frames, figsize=(4*n_frames, 8))
    if n_frames == 1:
        axes = axes.reshape(-1, 1)
    
    for i, frame_id in enumerate(frame_ids):
        # Get color image
        color_key = ('color', frame_id, 0)
        if color_key in sample:
            img = sample[color_key]  # [3, H, W]
            # Convert to PIL format [H, W, 3] and normalize if needed
            img_np = img.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f"Frame {frame_id}")
            axes[0, i].axis('off')
        
        # Check if we have segmentation
        seg_key = ('segmentation', frame_id, 0)
        if seg_key in sample:
            seg = sample[seg_key].squeeze().cpu().numpy()  # Remove channel dimension if present
            axes[1, i].imshow(seg, cmap='tab20')
            axes[1, i].set_title(f"Segmentation {frame_id}")
            axes[1, i].axis('off')
        else:
            # If no segmentation, just show the image again
            axes[1, i].imshow(img_np)
            axes[1, i].set_title(f"Frame {frame_id} (no seg)")
            axes[1, i].axis('off')
    
    # Add sample info
    fig.suptitle(f"PPGeo Dataset Sample {idx}", fontsize=16)
    
    # Add metadata if available
    metadata_text = []
    if ('sequence_id', 0) in sample:
        metadata_text.append(f"Sequence: {sample[('sequence_id', 0)]}")
    if ('K', 0) in sample:
        K = sample[('K', 0)]
        metadata_text.append(f"Intrinsics shape: {K.shape}")
    
    if metadata_text:
        fig.text(0.5, 0.02, " | ".join(metadata_text), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {save_path}")
    
    # Print sample statistics
    print("\n📊 Sample statistics:")
    print(f"   - Number of frames: {n_frames}")
    print(f"   - Frame IDs: {frame_ids}")
    print(f"   - Image shape: {sample[(color_key)].shape}")
    print(f"   - Available keys: {len(sample)} items")
    
    # Print all available keys grouped by type
    key_types = {}
    for key in sample.keys():
        if isinstance(key, tuple) and len(key) >= 1:
            key_type = key[0]
            if key_type not in key_types:
                key_types[key_type] = []
            key_types[key_type].append(key)
    
    print("\n   Available data types:")
    for key_type, keys in key_types.items():
        print(f"     - {key_type}: {len(keys)} items")
    
    return


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, cfg, motionnet=None, global_step=0, lr_scheduler=None):
    """Train one epoch."""
    model.train()
    total_loss = 0
    
    # Check for early exit based on steps
    max_steps = int(os.environ.get('PPGEO_MAX_STEPS', 0))
    if max_steps > 0 and global_step >= max_steps:
        print(f"\n🛑 Reached max steps ({max_steps}). Exiting for memory management...")
        return total_loss / max(1, len(dataloader)), global_step, True  # Return early_exit flag
    
    # Gradient accumulation setup
    accumulation_steps = getattr(cfg.TRAINING, 'GRAD_ACCUM_STEPS', 16)
    accumulation_steps = 16  # Reduced from 64 to save memory
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    # Initialize gradients to zero at the start
    optimizer.zero_grad()
    
    # Track samples processed for rate limiting
    samples_processed = 0
    rate_limit_delay = 60  # 60 seconds delay
    rate_limit_samples = 100000  # Delay after every 5000 samples
    
    # DataLoader now starts from the correct position via SkipBatchSampler
    # No need to skip batches manually - the sampler handles it efficiently
    
    for batch_idx, batch in enumerate(progress_bar):
        
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        
        # Forward pass
        if cfg.PPGEO.STAGE == 1:
            outputs, updated_inputs = model(batch)
        else:
            # Stage 2: Use MotionNet for pose
            motionnet.train()  # Ensure MotionNet is in training mode
            motion = motionnet(batch)
            
            # Run model with frozen encoder for depth, but keep motion in computation graph
            model.eval()
            outputs, updated_inputs = model(batch, motion=motion)
        
        # Compute losses
        losses = loss_fn(outputs, updated_inputs, stage=cfg.PPGEO.STAGE, encoder_name=cfg.PPGEO.ENCODER)
        total_loss_batch = losses['total_loss']
        
        # Scale loss for gradient accumulation
        scaled_loss = total_loss_batch
        
        # Backward pass - gradients accumulate automatically
        scaled_loss.backward()
        
        # Only update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters() if cfg.PPGEO.STAGE == 1 else motionnet.parameters(), 
            #     cfg.TRAINING.MAX_GRAD_NORM
            # )
            optimizer.step()
            optimizer.zero_grad()  # Clear gradients AFTER weight update
            if lr_scheduler is not None:
                lr_scheduler.step()
            global_step += 1
            
            # Check if we should exit early
            if max_steps > 0 and global_step >= max_steps:
                print(f"\n🛑 Reached max steps ({max_steps}). Saving checkpoint and exiting...")
                step_checkpoint_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, f"ppgeo_stage{cfg.PPGEO.STAGE}_step_{global_step}.pt")
                if cfg.PPGEO.STAGE == 1:
                    save_checkpoint(model, optimizer, epoch, total_loss_batch.item(), step_checkpoint_path, global_step)
                else:
                    save_checkpoint(motionnet, optimizer, epoch, total_loss_batch.item(), step_checkpoint_path, global_step)
                print(f"💾 Saved final checkpoint: {step_checkpoint_path}")
                return total_loss / max(1, batch_idx + 1), global_step, True
        
        # Step-based checkpoint saving every 1000 steps
        if global_step > 0 and global_step % 100 == 0:
            step_checkpoint_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, f"ppgeo_stage{cfg.PPGEO.STAGE}_step_{global_step}.pt")
            if cfg.PPGEO.STAGE == 1:
                save_checkpoint(model, optimizer, epoch, total_loss_batch.item(), step_checkpoint_path, global_step)
            else:
                save_checkpoint(motionnet, optimizer, epoch, total_loss_batch.item(), step_checkpoint_path, global_step)
            print(f"\n💾 Saved step checkpoint: {step_checkpoint_path}")

            global_step += 1
        
        # Logging
        total_loss += total_loss_batch.item()
        
        # Update samples processed counter
        batch_size = batch[('color', 0, 0)].shape[0] if ('color', 0, 0) in batch else cfg.DATASET.BATCH_SIZE
        samples_processed += batch_size
        
        # Clear GPU cache periodically to prevent memory fragmentation
        if batch_idx % 100 == 0 and batch_idx > 0:
            torch.cuda.empty_cache()
            gc.collect()  # Force garbage collection
            
        # Rate limiting: pause after every rate_limit_samples
        if samples_processed >= rate_limit_samples:
            print(f"\n⏸️ Rate limit pause: Processed {samples_processed} samples, sleeping for {rate_limit_delay} seconds to avoid S3 rate limits...")
            # Clear GPU cache before pause
            torch.cuda.empty_cache()
            time.sleep(rate_limit_delay)
            samples_processed = 0  # Reset counter
            print("▶️ Resuming training...")
        
        if batch_idx % cfg.LOGGING.LOG_FREQ == 0:
            avg_loss = total_loss / (batch_idx + 1)
            
            # Get GPU memory stats
            if torch.cuda.is_available():
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            else:
                gpu_mem_allocated = 0
                gpu_mem_reserved = 0
            
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}', 
                'step': global_step,
                'acc_steps': f'{(batch_idx + 1) % accumulation_steps}/{accumulation_steps}',
                'samples': samples_processed,
                'GPU': f'{gpu_mem_allocated:.1f}/{gpu_mem_reserved:.1f}GB'
            })
            
            if cfg.WANDB.USE_WANDB:
                wandb.log({
                    'train/total_loss': total_loss_batch.item(),
                    'train/reprojection_loss': losses.get('reprojection_loss', 0),
                    'train/smoothness_loss': losses.get('smoothness_loss', 0),
                    'train/global_step': global_step,
                    'train/epoch': epoch
                }, step=global_step)
    
    # Handle any remaining accumulated gradients at the end of epoch
    if len(dataloader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters() if cfg.PPGEO.STAGE == 1 else motionnet.parameters(), 
            cfg.TRAINING.MAX_GRAD_NORM
        )
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        global_step += 1
    
    return total_loss / len(dataloader), global_step, False  # No early exit


def validate(model, dataloader, loss_fn, device, epoch, cfg, motionnet=None):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            if cfg.PPGEO.STAGE == 1:
                outputs, updated_inputs = model(batch)
            else:
                # Stage 2: Use MotionNet for pose, frozen model for depth
                model.eval()
                motionnet.eval()
                motion = motionnet(batch)
                outputs, updated_inputs = model(batch, motion=motion)
            
            # Compute losses
            losses = loss_fn(outputs, updated_inputs, stage=cfg.PPGEO.STAGE, encoder_name=cfg.PPGEO.ENCODER)
            total_loss += losses['total_loss'].item()
    
    avg_loss = total_loss / len(dataloader)
    
    if cfg.WANDB.USE_WANDB:
        wandb.log({
            'val/total_loss': avg_loss,
            'val/epoch': epoch
        }, step=epoch)
    
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, filepath, global_step=0, best_val_loss=None):
    """Save model checkpoint."""
    checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'global_step': global_step
    }
    if best_val_loss is not None:
        checkpoint_dict['best_val_loss'] = best_val_loss
    torch.save(checkpoint_dict, filepath)


def main():
    # Load configuration
    import argparse
    parser = argparse.ArgumentParser(description='PPGeo Training')
    parser.add_argument('--config', type=str, default='config_ppgeo.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)
    cfg = update_config(cfg, args)
    cfg.freeze()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    print(f"🏗️ Training PPGeo Stage {cfg.PPGEO.STAGE}")
    
    # Initialize wandb
    if cfg.WANDB.USE_WANDB:
        wandb.init(
            project=cfg.WANDB.PROJECT,
            name=f"ppgeo-{cfg.WANDB.RUN_NAME}",
            config=dict(cfg)
        )
    
    # Create full dataset first
    print("📂 Loading datasets...")
    print(f"Frame sampling rate: {cfg.DATASET.FRAME_SAMPLING_RATE} (every {cfg.DATASET.FRAME_SAMPLING_RATE} frames = {10.0/cfg.DATASET.FRAME_SAMPLING_RATE:.1f}Hz)")
    full_dataset = PPGeoDataset(
        root_prefix=cfg.DATASET.YOUTUBE_ROOT_PREFIX,
        cache_dir=cfg.DATASET.YOUTUBE_CACHE_DIR,
        img_size=(cfg.DATASET.IMG_HEIGHT, cfg.DATASET.IMG_WIDTH),
        is_train=True,  # We'll handle train/val split manually
        frame_sampling_rate=cfg.DATASET.FRAME_SAMPLING_RATE,
        max_samples=cfg.DATASET.MAX_SAMPLES
    )

    # index an item for a test to ensure dataset works
    _ = full_dataset[0]
    
    # Quick visualization of dataset samples
    visualize_dataset_sample(full_dataset, idx=0, save_path="ppgeo_training_sample_0.png")
    visualize_dataset_sample(full_dataset, idx=min(10, len(full_dataset)-1), save_path="ppgeo_training_sample_10.png")
    

    
    # Split dataset into train and validation
    if cfg.DATASET.VAL_SPLIT > 0:
        total_size = len(full_dataset)
        # val_size = int(total_size * cfg.DATASET.VAL_SPLIT)
        val_size = int(total_size * 0.001)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible split
        )
        
        print(f"📊 Dataset split: {train_size:,} train, {val_size:,} validation samples")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print(f"📊 Using full dataset for training: {len(train_dataset):,} samples")
    
    # Create models
    print("🏗️ Creating PPGeo model...")
    if cfg.PPGEO.STAGE == 2:
        # For Stage 2, create Stage 1 model with ResNet-18 (to match checkpoint)
        stage1_cfg = cfg.clone()
        stage1_cfg.defrost()
        stage1_cfg.PPGEO.RESNET_LAYERS = 18  # Force ResNet-18 for Stage 1 model
        stage1_cfg.freeze()
        model = create_ppgeo_model(stage1_cfg)
        print("   Created Stage 1 model with ResNet-18 (to match checkpoint)")
    else:
        model = create_ppgeo_model(cfg)

    
    # Load pre-trained weights based on encoder type
    if cfg.PPGEO.ENCODER in ["dinov3", "dinov2", "vit"]:
        if cfg.DEPTHANYTHING_CHECKPOINT and os.path.exists(cfg.DEPTHANYTHING_CHECKPOINT):
            model.load_pretrained_depth_weights(cfg.DEPTHANYTHING_CHECKPOINT)
        else:
            print(f"⚠️ DepthAnythingV2 checkpoint not found at {cfg.DEPTHANYTHING_CHECKPOINT}")
            print("🔄 Continuing with ViT random initialization...")
    else:
        print(f"📦 Using ResNet-{cfg.PPGEO.RESNET_LAYERS} with ImageNet pretrained weights")
        
        # Load PPGeo pretrained ResNet weights if available
        if hasattr(cfg, 'DEPTH_CHECKPOINT') and cfg.DEPTH_CHECKPOINT and os.path.exists(cfg.DEPTH_CHECKPOINT):
            print(f"📦 Loading pretrained depth weights from: {cfg.DEPTH_CHECKPOINT}")
            depth_ckpt = torch.load(cfg.DEPTH_CHECKPOINT, map_location=device)
            
            # Load depth encoder weights
            if 'depth_encoder_state_dict' in depth_ckpt:
                missing_keys, unexpected_keys = model.encoder.load_state_dict(
                    depth_ckpt['depth_encoder_state_dict'], strict=False
                )
                if missing_keys:
                    print(f"⚠️ Depth encoder missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"⚠️ Depth encoder unexpected keys: {len(unexpected_keys)}")
                print("✅ Loaded pretrained depth encoder")
            
            # Load depth decoder weights
            if 'depth_decoder_state_dict' in depth_ckpt:
                missing_keys, unexpected_keys = model.depth_decoder.load_state_dict(
                    depth_ckpt['depth_decoder_state_dict'], strict=False
                )
                if missing_keys:
                    print(f"⚠️ Depth decoder missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"⚠️ Depth decoder unexpected keys: {len(unexpected_keys)}")
                print("✅ Loaded pretrained depth decoder")
        
        # Load PPGeo pretrained pose weights if available
        if hasattr(cfg, 'POSE_CHECKPOINT') and cfg.POSE_CHECKPOINT and os.path.exists(cfg.POSE_CHECKPOINT):
            print(f"📦 Loading pretrained pose weights from: {cfg.POSE_CHECKPOINT}")
            pose_ckpt = torch.load(cfg.POSE_CHECKPOINT, map_location=device)
            
            # Load pose encoder weights
            if 'pose_encoder_state_dict' in pose_ckpt:
                missing_keys, unexpected_keys = model.pose_encoder.load_state_dict(
                    pose_ckpt['pose_encoder_state_dict'], strict=False
                )
                if missing_keys:
                    print(f"⚠️ Pose encoder missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"⚠️ Pose encoder unexpected keys: {len(unexpected_keys)}")
                print("✅ Loaded pretrained pose encoder")
    
    model.to(device)
    
    motionnet = None
    if cfg.PPGEO.STAGE == 2:
        print("🏗️ Creating MotionNet for Stage 2...")
        # Use ResNet layers specified in config for MotionNet
        motionnet_layers = cfg.PPGEO.RESNET_LAYERS if cfg.PPGEO.ENCODER == "resnet" else 34
        print(f"📐 Using ResNet-{motionnet_layers} for MotionNet")
        motionnet = MotionNet(resnet_layers=motionnet_layers)
        motionnet.to(device)
        
        # Load pretrained ResNet weights if specified
        if hasattr(cfg, 'MOTIONNET_CHECKPOINT') and cfg.MOTIONNET_CHECKPOINT:
            print(f"📦 Loading pretrained MotionNet weights from: {cfg.MOTIONNET_CHECKPOINT}")
            try:
                motionnet_ckpt = torch.load(cfg.MOTIONNET_CHECKPOINT, map_location=device)
                if 'state_dict' in motionnet_ckpt:
                    # Load encoder weights from ResNet checkpoint
                    motionnet_state = motionnet.state_dict()
                    for key, value in motionnet_ckpt['state_dict'].items():
                        encoder_key = f"encoder.{key}"
                        if encoder_key in motionnet_state:
                            motionnet_state[encoder_key] = value
                    motionnet.load_state_dict(motionnet_state, strict=False)
                    print("✅ Loaded pretrained ResNet-34 encoder weights for MotionNet")
                else:
                    print("⚠️ Unexpected checkpoint format for MotionNet")
            except Exception as e:
                print(f"⚠️ Failed to load MotionNet checkpoint: {e}")
        
        # Load Stage 1 checkpoint if specified
        if hasattr(cfg, 'STAGE1_CHECKPOINT') and cfg.STAGE1_CHECKPOINT:
            print(f"📦 Loading Stage 1 checkpoint: {cfg.STAGE1_CHECKPOINT}")
            checkpoint = torch.load(cfg.STAGE1_CHECKPOINT, map_location=device)
            
            # Load the full Stage 1 model (including pose encoder)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✅ Loaded Stage 1 model (ResNet-18 with pose encoder)")
            print(f"   - Stage 1 uses ResNet-18 for both depth and pose")
            print(f"   - MotionNet will use separate ResNet-34 architecture")
        
        # Freeze Stage 1 model
        print("❄️ Freezing Stage 1 model parameters...")
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    
    # Count trainable parameters
    if cfg.PPGEO.STAGE == 1:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🔢 Model parameters: {total_params/1e6:.1f}M")
    else:
        motion_params = sum(p.numel() for p in motionnet.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters())
        print(f"🔢 MotionNet parameters: {motion_params/1e6:.1f}M")
        print(f"❄️ Frozen Stage 1 parameters: {frozen_params/1e6:.1f}M")
    
    # Create optimizer
    if cfg.PPGEO.STAGE == 1:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.TRAINING.LEARNING_RATE,
            weight_decay=1e-4
        )
    else:
        # Stage 2: Only optimize MotionNet parameters
        optimizer = optim.AdamW(
            motionnet.parameters(),
            lr=cfg.TRAINING.LEARNING_RATE,
            weight_decay=1e-4
        )
    
    # Create learning rate scheduler (like original PPGeo)
    lr_scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=1e-6, 
        max_lr=1e-4, 
        step_size_up=2000, 
        cycle_momentum=False
    )
    
    # Create loss function
    loss_scales = [0] if cfg.PPGEO.STAGE == 2 else cfg.PPGEO.SCALES
    loss_fn = PPGeoLoss(
        scales=loss_scales,
        frame_ids=cfg.PPGEO.FRAME_IDS
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=device)
        
        # Load model state
        if cfg.PPGEO.STAGE == 1:
            model.load_state_dict(resume_checkpoint['model_state_dict'])
        else:
            motionnet.load_state_dict(resume_checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        
        # Restore training state
        start_epoch = resume_checkpoint.get('epoch', 0) + 1  # Start from next epoch
        global_step = resume_checkpoint.get('global_step', 0)
        best_val_loss = resume_checkpoint.get('best_val_loss', float('inf'))
        
        print(f"✅ Resumed from epoch {start_epoch}, global step {global_step}")
        print(f"   Best validation loss so far: {best_val_loss:.4f}")
    
    # Create data loaders with dynamic random seed based on global step
    # This ensures different data order on each restart
    data_seed = 42 + global_step
    train_generator = torch.Generator()
    train_generator.manual_seed(data_seed)
    print(f"🎲 Using data seed: {data_seed} (base=42 + global_step={global_step})")
    
    # Calculate starting batch index for efficient skip
    batches_per_epoch = len(train_dataset) // cfg.DATASET.BATCH_SIZE
    batches_completed_in_virtual_epoch = global_step % batches_per_epoch
    
    # Create custom batch sampler for efficient skipping
    batch_sampler = SkipBatchSampler(
        train_dataset,
        start_batch_idx=batches_completed_in_virtual_epoch,
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=True,
        generator=train_generator
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=16,
        persistent_workers=True,  # Keep workers alive between epochs
        pin_memory=True  # Faster GPU transfer
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.DATASET.BATCH_SIZE, 
            shuffle=False,
            num_workers=4,
            drop_last=True,
            persistent_workers=True,
            pin_memory=True
        )
        print(f"📊 Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")
    else:
        val_loader = None
        print(f"📊 Train samples: {len(train_dataset):,} (no validation split)")
    
    # Training loop
    print("🎯 Starting training...")
    
    for epoch in range(start_epoch, cfg.TRAINING.NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{cfg.TRAINING.NUM_EPOCHS} ===")
        
        # Train
        train_loss, global_step, early_exit = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, cfg, motionnet, global_step, lr_scheduler)
        print(f"📈 Train loss: {train_loss:.4f}, Global step: {global_step}")
        
        # Check for early exit
        if early_exit:
            print(f"\n🛑 Early exit requested. Stopping training at step {global_step}")
            break
        
        # Validate every few epochs (only if validation data available)
        if val_loader is not None and epoch % cfg.VALIDATION.VAL_FREQ == 0:
            val_loss = validate(model, val_loader, loss_fn, device, epoch, cfg, motionnet)
            print(f"📊 Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if cfg.PPGEO.STAGE == 1:
                    checkpoint_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, "best_ppgeo_stage1_model.pt")
                    save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, global_step, best_val_loss)
                else:
                    checkpoint_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, "best_ppgeo_stage2_motionnet.pt")
                    save_checkpoint(motionnet, optimizer, epoch, val_loss, checkpoint_path, global_step, best_val_loss)
                print(f"💾 Saved best model: {checkpoint_path}")
        elif val_loader is None and epoch % 5 == 0:  # Save checkpoints periodically if no validation
            if cfg.PPGEO.STAGE == 1:
                checkpoint_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, f"ppgeo_stage1_epoch_{epoch}.pt")
                save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path, global_step)
            else:
                checkpoint_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, f"ppgeo_stage2_epoch_{epoch}.pt")
                save_checkpoint(motionnet, optimizer, epoch, train_loss, checkpoint_path, global_step)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    print("✅ Training completed!")
    
    if cfg.WANDB.USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()