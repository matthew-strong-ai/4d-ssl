"""
PPGeo-style self-supervised depth and pose training using the YouTube S3 dataset.
Based on train_cluster.py dataset but with PPGeo's photometric self-supervision approach.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np

# Import datasets and config
from ppgeo_dataset import PPGeoDataset
from config.defaults import get_cfg_defaults, update_config

# PPGeo components
from ppgeo_model import PPGeoModel
from ppgeo_losses import PPGeoLoss
from ppgeo_motionnet import MotionNet


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


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, cfg, motionnet=None, global_step=0):
    """Train one epoch."""
    model.train()
    total_loss = 0
    
    # Gradient accumulation setup
    accumulation_steps = getattr(cfg.TRAINING, 'GRAD_ACCUM_STEPS', 16)
    accumulation_steps = 64
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    # Initialize gradients to zero at the start
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Forward pass
        if cfg.PPGEO.STAGE == 1:
            outputs, updated_inputs = model(batch)
        else:
            # Stage 2: Use MotionNet for pose, frozen model for depth
            with torch.no_grad():
                model.eval()
                motion = motionnet(batch)
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
            global_step += 1
        
        # Step-based checkpoint saving every 1000 steps
        if global_step > 0 and global_step % 100 == 0:
            step_checkpoint_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, f"ppgeo_stage{cfg.PPGEO.STAGE}_step_{global_step}.pt")
            if cfg.PPGEO.STAGE == 1:
                save_checkpoint(model, optimizer, epoch, total_loss_batch.item(), step_checkpoint_path)
            else:
                save_checkpoint(motionnet, optimizer, epoch, total_loss_batch.item(), step_checkpoint_path)
            print(f"\n💾 Saved step checkpoint: {step_checkpoint_path}")

            global_step += 1
        
        # Logging
        total_loss += total_loss_batch.item()
        
        if batch_idx % cfg.LOGGING.LOG_FREQ == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}', 
                'step': global_step,
                'acc_steps': f'{(batch_idx + 1) % accumulation_steps}/{accumulation_steps}'
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
        global_step += 1
    
    return total_loss / len(dataloader), global_step


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


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def main():
    # Load configuration
    import argparse
    parser = argparse.ArgumentParser(description='PPGeo Training')
    parser.add_argument('--config', type=str, default='config_ppgeo.yaml', help='Path to config file')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)
    cfg = update_config(cfg)
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
    print("Frame sampling rate:", cfg.DATASET.FRAME_SAMPLING_RATE)
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.DATASET.BATCH_SIZE, 
            shuffle=False,
            num_workers=4,
            drop_last=True
        )
        print(f"📊 Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")
    else:
        val_loader = None
        print(f"📊 Train samples: {len(train_dataset):,} (no validation split)")
    
    # Create models
    print("🏗️ Creating PPGeo model...")
    model = create_ppgeo_model(cfg)

    
    # Load pre-trained DepthAnythingV2 weights (only for ViT encoders)
    if cfg.PPGEO.ENCODER in ["dinov3", "dinov2", "vit"]:
        if cfg.DEPTHANYTHING_CHECKPOINT and os.path.exists(cfg.DEPTHANYTHING_CHECKPOINT):
            model.load_pretrained_depth_weights(cfg.DEPTHANYTHING_CHECKPOINT)
        else:
            print(f"⚠️ DepthAnythingV2 checkpoint not found at {cfg.DEPTHANYTHING_CHECKPOINT}")
            print("🔄 Continuing with ViT random initialization...")
    else:
        print(f"📦 Using ResNet-{cfg.PPGEO.RESNET_LAYERS} with ImageNet pretrained weights")
        print("🔄 DepthAnything weights skipped (only compatible with ViT encoders)")
    
    model.to(device)
    
    motionnet = None
    if cfg.PPGEO.STAGE == 2:
        print("🏗️ Creating MotionNet for Stage 2...")
        motionnet = MotionNet(model_size="vitl")
        motionnet.to(device)
        
        # Load Stage 1 checkpoint if specified
        if hasattr(cfg, 'STAGE1_CHECKPOINT') and cfg.STAGE1_CHECKPOINT:
            print(f"📦 Loading Stage 1 checkpoint: {cfg.STAGE1_CHECKPOINT}")
            checkpoint = torch.load(cfg.STAGE1_CHECKPOINT, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
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
    
    # Create loss function
    loss_scales = [0] if cfg.PPGEO.STAGE == 2 else cfg.PPGEO.SCALES
    loss_fn = PPGeoLoss(
        scales=loss_scales,
        frame_ids=cfg.PPGEO.FRAME_IDS
    )
    
    # Training loop
    print("🎯 Starting training...")
    best_val_loss = float('inf')
    global_step = 0  # Track global step across epochs
    
    for epoch in range(cfg.TRAINING.NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{cfg.TRAINING.NUM_EPOCHS} ===")
        
        # Train
        train_loss, global_step = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, cfg, motionnet, global_step)
        print(f"📈 Train loss: {train_loss:.4f}, Global step: {global_step}")
        
        # Validate every few epochs (only if validation data available)
        if val_loader is not None and epoch % cfg.VALIDATION.VAL_FREQ == 0:
            val_loss = validate(model, val_loader, loss_fn, device, epoch, cfg, motionnet)
            print(f"📊 Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if cfg.PPGEO.STAGE == 1:
                    checkpoint_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, "best_ppgeo_stage1_model.pt")
                    save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
                else:
                    checkpoint_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, "best_ppgeo_stage2_motionnet.pt")
                    save_checkpoint(motionnet, optimizer, epoch, val_loss, checkpoint_path)
                print(f"💾 Saved best model: {checkpoint_path}")
        elif val_loader is None and epoch % 5 == 0:  # Save checkpoints periodically if no validation
            if cfg.PPGEO.STAGE == 1:
                checkpoint_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, f"ppgeo_stage1_epoch_{epoch}.pt")
                save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
            else:
                checkpoint_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, f"ppgeo_stage2_epoch_{epoch}.pt")
                save_checkpoint(motionnet, optimizer, epoch, train_loss, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    print("✅ Training completed!")
    
    if cfg.WANDB.USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()