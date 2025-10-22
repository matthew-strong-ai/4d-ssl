# 4D-SSL: Self-Supervised Learning from In-the-Wild Driving Videos

A PyTorch implementation of 4D self-supervised learning for autonomous driving, enabling 3D scene understanding and motion prediction from monocular video sequences.

## Overview

This project implements a self-supervised learning framework that learns 3D geometry, object segmentation, and motion dynamics from unlabeled driving videos. The model can:

- **3D Scene Reconstruction**: Generate depth maps and 3D point clouds from monocular video
- **Object Segmentation**: Identify vehicles, pedestrians, bicycles, road signs, and traffic lights
- **Motion Prediction**: Track dynamic objects and predict their 3D motion trajectories
- **Temporal Understanding**: Learn consistent 4D representations across video sequences

## Key Features

- **Multi-Model Architecture**: Supports Pi3, AutoregressivePi3, and AutonomyPi3 variants
- **Large-Scale Training**: Optimized for YouTube driving dataset (58M+ samples)
- **Advanced Segmentation**: 6-class segmentation with GSAM2 integration
- **Motion Analysis**: Dynamic vs static object classification with CoTracker
- **Distributed Training**: Multi-GPU support with Accelerate
- **Cloud Integration**: S3 dataset streaming and checkpoint management

## Documentation

- **[Installation Guide](INSTALL.md)** - Complete setup instructions for dependencies and environment
- **[Configuration Guide](README_YACS_CONFIG.md)** - Detailed explanation of configuration options
- **[Training Guide](MAPANYTHING_TRAINING_STEPS.md)** - Step-by-step training instructions
- **[Loss Functions Documentation](LOSS_CHANGES.md)** - Details on loss computation and optimization

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA 12.1+
- 24GB+ GPU (RTX 4090, A100 recommended)

### Basic Setup
```bash
# Clone repository
git clone https://github.com/matthew-strong-ai/4d-ssl.git
cd 4d-ssl

# Create environment
conda create -n 4d-ssl python=3.10
conda activate 4d-ssl

# Install dependencies
pip install -r requirements.txt

# Initialize submodules
git submodule update --init --recursive
```

### Training
```bash
# Basic training
python train_cluster.py

# With custom config
python train_cluster.py --config config.yaml

# Resume from checkpoint
python train_cluster.py --resume checkpoints/latest.pt
```

## Model Architecture

The project implements three main architectures:

### 1. **Pi3 (Base Model)**
- 3D point prediction from video sequences
- Camera pose estimation
- Depth and normal map generation

### 2. **AutoregressivePi3**
- Transformer-based temporal modeling
- Future frame prediction
- Enhanced motion understanding

### 3. **AutonomyPi3**
- Extended Pi3 with detection capabilities
- Traffic light and road sign detection
- Multi-task learning framework

## Dataset

The model is trained on large-scale driving video datasets:

- **YouTube Driving Dataset**: 58M+ frames from 200+ cities worldwide
- **Custom S3 Dataset**: Support for proprietary driving data
- **Local Dataset**: For testing and development

### Dataset Configuration
```yaml
DATASET:
  USE_YOUTUBE: True
  YOUTUBE_ROOT_PREFIX: "openDV-YouTube/full_images/"
  BATCH_SIZE: 1
  MAX_SAMPLES: -1  # Use full dataset
```

## Training Pipeline

1. **Data Loading**: Streaming from S3/YouTube dataset
2. **GSAM2 Processing**: Object detection and segmentation
3. **CoTracker Integration**: Point tracking across frames
4. **Model Forward Pass**: 3D prediction and motion estimation
5. **Loss Computation**: Multi-task losses with class weighting
6. **Optimization**: AdamW with cosine annealing

## Key Components

### Object Classes
```python
1. Vehicle (cars, trucks, buses, motorcycles)
2. Bicycle
3. Person
4. Road Sign
5. Traffic Light
0. Background
```

### Loss Functions
- **Point Cloud Loss**: 3D geometry supervision
- **Segmentation Loss**: Cross-entropy with class weights
- **Motion Loss**: 3D motion field prediction
- **Camera Pose Loss**: SE(3) pose estimation
- **Confidence Loss**: Prediction uncertainty

## Results

The model produces:
- **3D Point Clouds**: Dense depth estimation
- **Segmentation Masks**: Per-pixel class predictions
- **Motion Vectors**: 3D motion flow fields
- **Dynamic Masks**: Moving vs static object classification

## Monitoring

### Weights & Biases
```bash
# Set up W&B
wandb login
export WANDB_API_KEY="your_key"
```

Training metrics are logged to W&B including:
- Loss curves
- Validation metrics
- Sample visualizations
- Model checkpoints

### TensorBoard
```bash
tensorboard --logdir runs/
```

## Advanced Usage

### Multi-GPU Training
```bash
# Using Accelerate
accelerate launch train_cluster.py

# Distributed training
torchrun --nproc_per_node=4 train_cluster.py
```

### Custom Model Configuration
```python
MODEL:
  ARCHITECTURE: "AutoregressivePi3"
  USE_MOTION_HEAD: True
  USE_SEGMENTATION_HEAD: True
  FREEZE_DECODERS: True
```

### Hyperparameter Tuning
See [hyperparameter_search.yaml](config/hyperparameter_search.yaml) for sweep configurations.

## Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**
   ```yaml
   DATASET.BATCH_SIZE: 1
   TRAINING.GRAD_ACCUM_STEPS: 4
   ```

2. **S3 Access Issues**
   ```bash
   export AWS_REQUEST_CHECKSUM_CALCULATION="WHEN_SUPPORTED"
   aws configure
   ```

3. **Missing Dependencies**
   ```bash
   pip install -e git+https://github.com/IDEA-Research/Grounded-SAM-2.git#egg=SAM-2
   ```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this code in your research, please cite:
```bibtex
@software{4d-ssl,
  author = {Matthew Strong},
  title = {4D-SSL: Self-Supervised Learning from In-the-Wild Driving Videos},
  year = {2025},
  url = {https://github.com/matthew-strong-ai/4d-ssl}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Pi3 Model](https://github.com/matthew-strong-ai/Pi3) - Base 3D prediction architecture
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) - Segmentation backend
- [CoTracker](https://github.com/facebookresearch/co-tracker) - Point tracking
- [DINOv3](https://github.com/facebookresearch/dinov3) - Feature extraction

## Contact

For questions and feedback:
- Create an issue on [GitHub](https://github.com/matthew-strong-ai/4d-ssl/issues)
- Email: [your-email@example.com]

---

**Note**: This is an active research project. APIs and features may change.