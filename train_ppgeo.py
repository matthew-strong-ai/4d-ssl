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


def create_ppgeo_model(cfg):
    """Create PPGeo model with DinOV3 encoder and DPT decoder."""
    model = PPGeoModel(
        encoder_name="dinov3",
        img_size=(cfg.DATASET.IMG_HEIGHT, cfg.DATASET.IMG_WIDTH),
        min_depth=0.1,
        max_depth=100.0,
        scales=[0, 1, 2, 3]
    )
    return model


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, cfg):
    """Train one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch)
        
        # Compute losses
        losses = loss_fn(outputs, batch)
        total_loss_batch = losses['total_loss']
        
        # Backward pass
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAINING.MAX_GRAD_NORM)
        optimizer.step()
        
        # Logging
        total_loss += total_loss_batch.item()
        
        if batch_idx % cfg.LOGGING.LOG_FREQ == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            if cfg.WANDB.USE_WANDB:
                step = epoch * len(dataloader) + batch_idx
                wandb.log({
                    'train/total_loss': total_loss_batch.item(),
                    'train/reprojection_loss': losses.get('reprojection_loss', 0),
                    'train/smoothness_loss': losses.get('smoothness_loss', 0),
                    'train/step': step
                }, step=step)
    
    return total_loss / len(dataloader)


def validate(model, dataloader, loss_fn, device, epoch, cfg):
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
            outputs = model(batch)
            
            # Compute losses
            losses = loss_fn(outputs, batch)
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
    cfg = get_cfg_defaults()
    cfg = update_config(cfg)
    
    # Add PPGeo specific config
    cfg.defrost()
    cfg.PPGEO = cfg.get('PPGEO', {})
    cfg.PPGEO.STAGE = 1  # Start with stage 1 (depth + pose)
    cfg.PPGEO.FRAME_IDS = [-1, 0, 1]  # Previous, current, next frames
    cfg.PPGEO.SCALES = [0, 1, 2, 3]   # Multi-scale loss
    
    # Image settings for PPGeo
    cfg.DATASET.IMG_HEIGHT = 160
    cfg.DATASET.IMG_WIDTH = 320
    cfg.freeze()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Initialize wandb
    if cfg.WANDB.USE_WANDB:
        wandb.init(
            project=cfg.WANDB.PROJECT,
            name=f"ppgeo-{cfg.WANDB.RUN_NAME}",
            config=dict(cfg)
        )
    
    # Create datasets
    print("📂 Loading datasets...")
    train_dataset = PPGeoDataset(
        root_prefix=cfg.DATASET.YOUTUBE_ROOT_PREFIX,
        cache_dir=cfg.DATASET.YOUTUBE_CACHE_DIR,
        img_size=(cfg.DATASET.IMG_HEIGHT, cfg.DATASET.IMG_WIDTH),
        is_train=True,
        frame_sampling_rate=cfg.DATASET.FRAME_SAMPLING_RATE,
        max_samples=cfg.DATASET.MAX_SAMPLES
    )
    
    val_dataset = PPGeoDataset(
        root_prefix=cfg.DATASET.YOUTUBE_ROOT_PREFIX,
        cache_dir=cfg.DATASET.YOUTUBE_CACHE_DIR,
        img_size=(cfg.DATASET.IMG_HEIGHT, cfg.DATASET.IMG_WIDTH),
        is_train=False,
        frame_sampling_rate=cfg.DATASET.FRAME_SAMPLING_RATE,
        max_samples=cfg.DATASET.MAX_SAMPLES // 10 if cfg.DATASET.MAX_SAMPLES > 0 else -1
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.DATASET.BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        drop_last=True
    )
    
    print(f"📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print("🏗️ Creating PPGeo model...")
    model = create_ppgeo_model(cfg)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Model parameters: {total_params/1e6:.1f}M")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.TRAINING.LEARNING_RATE,
        weight_decay=1e-4
    )
    
    # Create loss function
    loss_fn = PPGeoLoss(
        scales=cfg.PPGEO.SCALES,
        frame_ids=cfg.PPGEO.FRAME_IDS
    )
    
    # Training loop
    print("🎯 Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(cfg.TRAINING.NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{cfg.TRAINING.NUM_EPOCHS} ===")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, cfg)
        print(f"📈 Train loss: {train_loss:.4f}")
        
        # Validate every few epochs
        if epoch % cfg.VALIDATION.VAL_FREQ == 0:
            val_loss = validate(model, val_loader, loss_fn, device, epoch, cfg)
            print(f"📊 Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(cfg.OUTPUT.CHECKPOINT_DIR, "best_ppgeo_model.pt")
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
                print(f"💾 Saved best model: {checkpoint_path}")
    
    print("✅ Training completed!")
    
    if cfg.WANDB.USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()