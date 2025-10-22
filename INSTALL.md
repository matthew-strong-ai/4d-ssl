# 4D SSL Installation Guide

This guide covers the complete installation process for running the 4D Self-Supervised Learning project, specifically for executing `python train_cluster.py`.

## Prerequisites

- **Python**: 3.8+ (recommended: 3.10+)
- **CUDA**: 12.1+ (for GPU training)
- **Git**: For cloning repositories
- **System RAM**: 32GB+ recommended
- **GPU**: 24GB+ VRAM recommended (e.g., RTX 4090, A100)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/matthew-strong-ai/4d-ssl.git
cd 4d-ssl
```

### 2. Create Python Environment

#### Using Conda (Recommended)
```bash
conda create -n 4d-ssl python=3.10
conda activate 4d-ssl
```

#### Using venv
```bash
python -m venv 4d-ssl-env
source 4d-ssl-env/bin/activate  # Linux/Mac
# or
4d-ssl-env\Scripts\activate     # Windows
```

### 3. Install Core Dependencies

#### Install PyTorch with CUDA Support
```bash
pip install torch>=2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Install Required Packages
```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- **Core ML packages**: numpy, scipy, scikit-learn
- **Training utilities**: accelerate, tensorboard, tqdm, wandb
- **Computer vision**: opencv-python, pillow, imageio
- **PyTorch ecosystem**: torch-ema, timm, transformers
- **Configuration**: yacs
- **Cloud storage**: boto3, lance
- **3D utilities**: utils3d
- **Video processing**: torchcodec

### 4. Install External Repositories

#### CoTracker (Point Tracking)
```bash
pip install git+https://github.com/facebookresearch/co-tracker.git
```

#### Grounded SAM 2 (Segmentation)
```bash
pip install -e git+https://github.com/IDEA-Research/Grounded-SAM-2.git#egg=SAM-2
```

### 5. Clone Submodules and Dependencies

#### Initialize Git Submodules
```bash
git submodule update --init --recursive
```

#### Clone Additional Repositories
The following repositories should be cloned into the project directory:

```bash
# SpaTracker for motion tracking
git clone https://github.com/henry123-boy/SpaTracker_V2.git SpaTrackerV2

# GroundingDINO for object detection
git clone https://github.com/IDEA-Research/GroundingDINO.git

# DINOv3 for feature extraction
git clone https://github.com/facebookresearch/dinov3.git

# CroCo for 3D vision
git clone https://github.com/naver/croco.git
```

### 6. Download Model Weights

#### Create directories for model weights
```bash
mkdir -p gdino
mkdir -p checkpoints
```

#### Download GroundingDINO weights
```bash
cd gdino
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

#### Download Segformer weights (if using segmentation)
```bash
wget https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-e4f52ffa.pth -O segformer.b0.512x512.ade.160k.pth
```

### 7. Environment Setup

#### AWS Configuration (for S3 dataset access)
```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
```

Set the following environment variables:
```bash
export AWS_REQUEST_CHECKSUM_CALCULATION="WHEN_SUPPORTED"
export AWS_RESPONSE_CHECKSUM_VALIDATION="WHEN_SUPPORTED"
```

#### Weights & Biases (for logging)
```bash
# Login to W&B
wandb login

# Set API key as environment variable
export WANDB_API_KEY="your_wandb_api_key"
```

### 8. Directory Structure Verification

After installation, your directory should look like:
```
4d-ssl/
├── Pi3/                    # Pi3 model submodule
├── SpaTrackerV2/          # Motion tracking
├── GroundingDINO/         # Object detection
├── Grounded-SAM-2/        # Segmentation
├── co-tracker/            # Point tracking
├── dinov3/                # Feature extraction
├── croco/                 # 3D vision
├── gdino/                 # GroundingDINO weights
├── config/                # Configuration files
├── utils/                 # Utility modules
├── vision/                # Vision components
├── train_cluster.py       # Main training script
├── config.yaml           # Main configuration
├── requirements.txt      # Python dependencies
└── ...
```

## Running the Training

### Basic Usage
```bash
python train_cluster.py
```

### With Custom Configuration
```bash
python train_cluster.py --config config/custom_config.yaml
```

### Common Arguments
- `--config`: Path to configuration file
- `--resume`: Resume from checkpoint
- `--debug`: Enable debug mode

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
- Reduce batch size in `config.yaml`: `DATASET.BATCH_SIZE: 1`
- Enable gradient accumulation: `TRAINING.GRAD_ACCUM_STEPS: 4`

#### Missing Dependencies
```bash
# Install additional scipy dependencies
pip install scipy>=1.9.0

# Install rich for better console output
pip install rich
```

#### Git Submodule Issues
```bash
# Remove and re-initialize submodules
git submodule deinit --all
git submodule update --init --recursive
```

#### S3 Access Issues
```bash
# Verify AWS credentials
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://research-datasets/
```

### Performance Optimization

#### For Training
- Enable mixed precision: Set `TRAINING.USE_AMP: True` in config
- Use DataLoader workers: Set `DATASET.NUM_WORKERS: 4`
- Enable CUDA kernel optimizations:
  ```bash
  export TORCH_CUDNN_V8_API_ENABLED=1
  export CUDA_LAUNCH_BLOCKING=0
  ```

#### For Memory Usage
- Set `DATASET.BATCH_SIZE: 1`
- Enable gradient checkpointing in model config
- Use `torch.compile()` for PyTorch 2.0+

## Verification

### Test Installation
```bash
# Test basic imports
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Test model loading
python test.py

# Test dataset loading
python test_youtube_dataset.py
```

### Check GPU Setup
```bash
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}, Current: {torch.cuda.current_device()}')"
```

## Hardware Recommendations

### Minimum Requirements
- **GPU**: RTX 3080 (10GB VRAM)
- **RAM**: 16GB
- **Storage**: 100GB SSD

### Recommended Setup
- **GPU**: RTX 4090 (24GB VRAM) or A100 (40GB)
- **RAM**: 64GB
- **Storage**: 500GB+ NVMe SSD
- **CPU**: 16+ cores

### Cluster Setup
For distributed training, ensure:
- Multiple GPUs with NVLink/PCIe connectivity
- High-bandwidth interconnect (InfiniBand recommended)
- Shared filesystem (NFS/Lustre) for dataset access

## Support

For issues related to:
- **Installation**: Check this guide and GitHub issues
- **Model training**: Review configuration files and logs
- **Dataset access**: Verify AWS/S3 credentials
- **GPU issues**: Check CUDA installation and driver versions

Create an issue on the [GitHub repository](https://github.com/matthew-strong-ai/4d-ssl/issues) with:
- Full error messages
- System specifications
- Installation steps attempted
- Configuration files used