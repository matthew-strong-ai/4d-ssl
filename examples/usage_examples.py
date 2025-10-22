"""
Usage examples for the YACS-based training configuration.
"""

import os
import sys
sys.path.append('..')

from train_cluster import train_with_config, train_with_tune_config
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

def example_direct_training():
    """Example: Direct training with default config.yaml."""
    print("==> Example 1: Direct training with default config.yaml")
    
    # Train using default config.yaml (automatically loaded)
    train_with_config()
    
    # Or train with specific config file
    train_with_config("config/train_config.yaml")

def example_override_config():
    """Example: Override config parameters from command line."""
    print("==> Example 2: Override config from command line")
    print("Run: python train_cluster.py --opts TRAINING.LEARNING_RATE 1e-4 LOSS.FUTURE_FRAME_WEIGHT 3.0")
    print("Or with custom config: python train_cluster.py --config custom_config.yaml --opts TRAINING.LEARNING_RATE 1e-4")

def example_ray_tune_search():
    """Example: Hyperparameter search with Ray Tune."""
    print("==> Example 3: Ray Tune hyperparameter search")
    
    # Define search space
    search_space = {
        "learning_rate": tune.loguniform(1e-6, 1e-3),
        "future_frame_weight": tune.uniform(1.0, 4.0),
        "pc_loss_weight": tune.uniform(0.05, 0.5),
        "warmup_start_factor": tune.uniform(0.05, 0.2),
        "num_epochs": 5,  # Fixed for faster search
    }
    
    # Configure search algorithm
    hyperopt_search = HyperOptSearch(
        metric="accuracy", 
        mode="max"
    )
    
    # Configure scheduler for early stopping
    scheduler = ASHAScheduler(
        time_attr='epoch',
        metric='accuracy',
        mode='max',
        max_t=5,
        grace_period=1,
        reduction_factor=2
    )
    
    # Run hyperparameter search
    tuner = tune.Tuner(
        train_with_tune_config,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            scheduler=scheduler,
            search_alg=hyperopt_search,
            num_samples=10  # Number of trials
        ),
        run_config=tune.RunConfig(
            name="pi3_hyperparameter_search",
            stop={"epoch": 5},  # Stop after 5 epochs
        )
    )
    
    results = tuner.fit()
    
    print("Best trial config: ", results.get_best_result().config)
    print("Best trial final accuracy: ", results.get_best_result().metrics["accuracy"])

def example_config_variations():
    """Example: Creating different config files for different experiments."""
    print("==> Example 4: Different experiment configurations")
    
    configs = {
        "baseline": {
            "future_frame_weight": 1.0,
            "learning_rate": 5e-5,
            "description": "Equal weighting for all frames"
        },
        "future_emphasis": {
            "future_frame_weight": 3.0,
            "learning_rate": 5e-5,
            "description": "3x weight on future frames"
        },
        "high_lr": {
            "future_frame_weight": 2.0,
            "learning_rate": 1e-4,
            "description": "Higher learning rate experiment"
        }
    }
    
    for name, config in configs.items():
        print(f"\n--- {name.upper()} EXPERIMENT ---")
        print(f"Description: {config['description']}")
        print("Create config file with these parameters:")
        for key, value in config.items():
            if key != 'description':
                print(f"  {key}: {value}")

if __name__ == "__main__":
    print("Pi3 Training Configuration Examples")
    print("=" * 50)
    
    # Show different usage patterns
    example_override_config()
    print()
    example_config_variations()
    
    # Uncomment to run actual training
    # example_direct_training()
    # example_ray_tune_search()