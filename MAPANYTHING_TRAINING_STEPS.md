# MapAnything Training Steps

This guide explains how to train the MapAnything model for autonomous driving scene understanding.

## Prerequisites

1. **Environment Setup**
   ```bash
   # Ensure you're in the autonomy-wild directory
   cd /home/matthew_strong/Desktop/autonomy-wild
   
   # Install MapAnything dependencies
   cd map-anything
   pip install -e .
   cd ..
   ```

2. **Verify MapAnything Installation**
   ```bash
   python -c "from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_model; print('MapAnything import successful')"
   ```

## Training Steps

### 1. Single GPU Training

```bash
python train_cluster.py --config config_mapanything.yaml
```

### 2. Multi-GPU Training (Recommended)

For distributed training across multiple GPUs:

```bash
# 4 GPUs example
torchrun --nproc_per_node=4 train_cluster.py --config config_mapanything.yaml

# 8 GPUs example
torchrun --nproc_per_node=8 train_cluster.py --config config_mapanything.yaml
```

### 3. SLURM Cluster Training

If using SLURM for cluster training:

```bash
sbatch run_mapanything_training.sh
```

Create `run_mapanything_training.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=mapanything_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=480G
#SBATCH --time=72:00:00

# Load modules
module load cuda/11.8
module load python/3.10

# Activate environment
source /path/to/your/venv/bin/activate

# Run training
cd /home/matthew_strong/Desktop/autonomy-wild
srun torchrun --nproc_per_node=8 train_cluster.py --config config_mapanything.yaml
```

## Configuration Options

### Key MapAnything Settings in config_mapanything.yaml:

1. **Model Architecture**
   - `MODEL.ARCHITECTURE: "MapAnything"` - Switches from Pi3 to MapAnything
   - `MODEL.MAPANYTHING.TASK`: Options include:
     - `"images_only"` - Standard monocular depth estimation
     - `"mvs"` - Multi-view stereo
     - `"registration"` - Point cloud registration
     - `"calibrated_sfm"` - Structure from motion with calibration

2. **Resolution and Input**
   - `MODEL.MAPANYTHING.RESOLUTION: 518` - Input image resolution
   - `MODEL.MAPANYTHING.PATCH_SIZE: 14` - Vision transformer patch size
   - `MODEL.M: 3` - Number of input frames
   - `MODEL.N: 3` - Number of target frames

3. **Training Parameters**
   - `BATCH_SIZE: 10` - Adjust based on GPU memory
   - `TRAINING.GRAD_ACCUM_STEPS: 1` - Gradient accumulation for larger effective batch size
   - `TRAINING.LEARNING_RATE: 5e-5` - Initial learning rate
   - `TRAINING.NUM_EPOCHS: 50` - Total training epochs

## Monitoring Training

### 1. Weights & Biases
Training metrics are automatically logged to W&B:
```bash
# View in browser
https://wandb.ai/your-username/mapanything-cluster-training
```

### 2. Local Logs
```bash
# Monitor training progress
tail -f logs/mapanything_training.log

# Check GPU usage
nvidia-smi -l 1
```

### 3. Checkpoints
Checkpoints are saved to:
- Local: `checkpoints/mapanything_*.pt`
- S3: `s3://research-datasets/autonomy_checkpoints/mapanything_*.pt`

## Common Issues and Solutions

### 1. Out of Memory (OOM)
```yaml
# Reduce batch size in config_mapanything.yaml
BATCH_SIZE: 4  # or even 2

# Or increase gradient accumulation
TRAINING:
  GRAD_ACCUM_STEPS: 4
```

### 2. MapAnything Import Error
```bash
# Add to Python path
export PYTHONPATH=$PYTHONPATH:/home/matthew_strong/Desktop/autonomy-wild/map-anything
```

### 3. Mixed Precision Issues
```yaml
# Try fp16 instead of bf16
MODEL:
  MAPANYTHING:
    AMP_DTYPE: "fp16"
```

## Evaluation and Inference

### 1. Evaluate Checkpoint
```bash
python evaluate_mapanything.py --checkpoint checkpoints/best_mapanything.pt
```

### 2. Run Inference
```bash
python inference_mapanything.py \
    --input_dir test_images/ \
    --output_dir outputs/ \
    --checkpoint checkpoints/best_mapanything.pt
```

## Tips for Best Results

1. **Start with Pretrained Weights**: MapAnything benefits from pretrained initialization
2. **Use Multiple GPUs**: Distributed training significantly speeds up convergence
3. **Monitor Depth Quality**: Check depth visualizations every 200 steps
4. **Adjust Learning Rate**: If loss plateaus, reduce learning rate by 10x
5. **Enable Motion Head**: For dynamic scenes, keep `USE_MOTION_HEAD: True`

## Next Steps

After training:
1. Evaluate on validation set
2. Fine-tune on specific driving scenarios
3. Export model for deployment
4. Integrate with downstream tasks (planning, control)