# YACS Configuration System for Pi3 Training

This repository now supports flexible configuration management using YACS (Yet Another Configuration System), allowing you to easily manage training parameters through YAML files.

## 🚀 Quick Start

### 1. Basic Training with Default Config

```bash
# Train with default config.yaml (automatically loaded)
python train_cluster.py

# Train with custom config file
python train_cluster.py --config my_custom_config.yaml
```

### 2. Override Parameters from Command Line

```bash
# Override specific parameters (uses default config.yaml)
python train_cluster.py --opts \
    TRAINING.LEARNING_RATE 1e-4 \
    LOSS.FUTURE_FRAME_WEIGHT 3.0 \
    TRAINING.NUM_EPOCHS 10

# Override with custom config + parameters
python train_cluster.py --config my_config.yaml --opts \
    TRAINING.LEARNING_RATE 1e-4 \
    LOSS.FUTURE_FRAME_WEIGHT 3.0
```

### 3. Programmatic Usage

```python
from train_cluster import train_with_config

# Train with default config.yaml
train_with_config()

# Train with specific config file
train_with_config("config/train_config.yaml")
```

## 📁 Configuration Structure

The configuration is organized into logical sections:

```yaml
# config/train_config.yaml

DATASET:
  ROOT_DIR: "/path/to/your/data"
  BATCH_SIZE: 1
  VAL_SPLIT: 0.1

MODEL:
  M: 3                    # Input frames
  N: 3                    # Future frames
  GRID_SIZE: 10

TRAINING:
  NUM_EPOCHS: 5
  LEARNING_RATE: 5e-5
  WARMUP_STEPS: 500
  WARMUP_START_FACTOR: 0.1
  GRAD_ACCUM_STEPS: 4
  MAX_GRAD_NORM: 1.0

LOSS:
  PC_LOSS_WEIGHT: 0.1
  POSE_LOSS_WEIGHT: 0.9
  CONF_LOSS_WEIGHT: 0.5
  FUTURE_FRAME_WEIGHT: 2.0  # Key parameter for future frame emphasis

VALIDATION:
  VAL_FREQ: 1000
  VAL_SAMPLES: 50
  EARLY_STOPPING_PATIENCE: 10

WANDB:
  PROJECT: "pi3-cluster-training"
  USE_WANDB: True
```

## 🔍 Hyperparameter Search with Ray Tune

The system integrates seamlessly with Ray Tune for hyperparameter optimization:

```python
from ray import tune
from train_cluster import train_with_tune_config

# Define search space
search_space = {
    "learning_rate": tune.loguniform(1e-6, 1e-3),
    "future_frame_weight": tune.uniform(1.0, 4.0),
    "pc_loss_weight": tune.uniform(0.05, 0.5),
}

# Run hyperparameter search
tuner = tune.Tuner(
    train_with_tune_config,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        metric="accuracy",
        mode="max",
        num_samples=20
    )
)

results = tuner.fit()
```

## 📝 Configuration Files

### Available Configurations

1. **`config/train_config.yaml`** - Default training configuration
2. **`config/hyperparameter_search.yaml`** - Template for hyperparameter search
3. **`examples/usage_examples.py`** - Usage examples and tutorials

### Creating Custom Configurations

1. Copy `config/train_config.yaml`
2. Modify parameters as needed
3. Use with `--config path/to/your/config.yaml`

## ⚙️ Key Parameters

### Future Frame Supervision
- **`LOSS.FUTURE_FRAME_WEIGHT`**: Controls emphasis on future frames (default: 2.0)
  - `1.0` = Equal weighting for all frames
  - `>1.0` = Stronger supervision on future frames
  - `3.0` = 3x stronger supervision on future frames

### Learning Rate Scheduling
- **`TRAINING.WARMUP_STEPS`**: Number of warmup steps (default: 500)
- **`TRAINING.WARMUP_START_FACTOR`**: Starting LR factor (default: 0.1)

### Loss Weights
- **`LOSS.PC_LOSS_WEIGHT`**: Point cloud loss weight (default: 0.1)
- **`LOSS.POSE_LOSS_WEIGHT`**: Camera pose loss weight (default: 0.9)
- **`LOSS.CONF_LOSS_WEIGHT`**: Confidence loss weight (default: 0.5)

## 🛠️ Advanced Usage

### Experiment Management

Create different config files for different experiments:

```bash
# Baseline experiment
config/baseline.yaml

# Future frame emphasis experiment  
config/future_emphasis.yaml

# High learning rate experiment
config/high_lr.yaml
```

### Integration with Ray Tune

The system provides two main functions:

- **`train_with_config()`**: For direct training with YAML config
- **`train_with_tune_config()`**: For Ray Tune integration

### Command Line Flexibility

```bash
# Multiple parameter overrides
python train_cluster.py \
  --config config/train_config.yaml \
  --opts \
    DATASET.ROOT_DIR "/new/data/path" \
    TRAINING.LEARNING_RATE 1e-4 \
    TRAINING.NUM_EPOCHS 20 \
    LOSS.FUTURE_FRAME_WEIGHT 4.0 \
    WANDB.PROJECT "my-experiment"
```

## 📊 Benefits

1. **Reproducibility**: Easy to share and reproduce experiments
2. **Flexibility**: Override any parameter from command line
3. **Organization**: Logical grouping of related parameters
4. **Ray Tune Integration**: Seamless hyperparameter optimization
5. **Version Control**: Track configuration changes with git

## 🔗 Migration from argparse

The new system maintains backward compatibility while providing enhanced flexibility. Instead of command-line arguments, use YAML configurations for better experiment management and reproducibility.