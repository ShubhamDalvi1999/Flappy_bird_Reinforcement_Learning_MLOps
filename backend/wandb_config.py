"""Weights & Biases configuration and utilities."""

import os
import wandb
import platform
import psutil
import tensorflow as tf

# Default W&B settings
DEFAULT_CONFIG = {
    "entity": os.getenv("WANDB_ENTITY", None),  # Your team name or username
    "project": os.getenv("WANDB_PROJECT", "flappy-bird-rl"),
    "name": None,  # Will be auto-generated based on timestamp
    "config": {
        # Model architecture
        "architecture": "DQN",
        "state_size": 4,  # [bird_y, bird_velocity, pipe_height, pipe_distance]
        "action_size": 2,  # [do_nothing, flap]
        
        # Training hyperparameters
        "learning_rate": 0.001,
        "gamma": 0.99,  # Discount factor
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "memory_size": 10000,
        "batch_size": 64,
        
        # Environment settings
        "env_name": "FlappyBird",
        "max_episodes": 1000,
        "target_update_freq": 10,
        
        # System info
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024 ** 3),  # GB
            "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
            "gpu_count": len(tf.config.list_physical_devices('GPU')),
            "tf_version": tf.__version__
        }
    }
}

def init_wandb(config_updates=None):
    """Initialize a new W&B run with given config updates.
    
    Args:
        config_updates (dict, optional): Updates to the default config.
    
    Returns:
        wandb.Run: The initialized W&B run.
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_updates:
        # Update the nested config dictionary
        if 'config' in config_updates:
            config['config'].update(config_updates['config'])
        # Update top-level settings
        for k, v in config_updates.items():
            if k != 'config':
                config[k] = v
    
    # Initialize W&B run
    run = wandb.init(
        entity=config['entity'],
        project=config['project'],
        name=config['name'],
        config=config['config'],
        reinit=True  # Allow multiple runs in the same process
    )
    
    return run

def log_episode_metrics(episode, score, reward, epsilon, loss=None):
    """Log metrics for a single training episode.
    
    Args:
        episode (int): Current episode number
        score (float): Episode score
        reward (float): Total episode reward
        epsilon (float): Current exploration rate
        loss (float, optional): Training loss
    """
    metrics = {
        "episode": episode,
        "score": score,
        "reward": reward,
        "epsilon": epsilon
    }
    
    if loss is not None:
        metrics["loss"] = loss
    
    wandb.log(metrics)

def log_model_summary(model):
    """Log model architecture summary to W&B.
    
    Args:
        model: The neural network model
    """
    wandb.run.summary["model_summary"] = str(model.summary()) 