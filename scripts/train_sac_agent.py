"""
Train SAC agent for Reacher Object Detection environment.

This script:
1. Sets up reproducible training with seeds
2. Creates environment with YOLO wrapper
3. Initializes SAC agent with configured hyperparameters
4. Sets up callbacks for logging, checkpointing, and early stopping
5. Trains the agent
6. Saves final model and training statistics
"""

import sys
from pathlib import Path
import os
import random
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import ReacherBoxEnv
from src.environment.wrappers import YOLOReacherWrapper
from src.agent.sac_agent import SACAgent
from src.agent.callbacks import (
    SuccessRateCallback,
    SaveBestModelCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    CustomMetricsCallback
)


def set_seeds(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✓ Random seeds set to: {seed}")


def make_env(env_id: str, use_yolo: bool, rank: int, seed: int = 0):
    """
    Utility function for creating a single environment.

    Args:
        env_id: Gymnasium environment ID
        use_yolo: Whether to use YOLO detection
        rank: Index of the subprocess
        seed: Random seed

    Returns:
        Function that creates and returns the environment
    """
    def _init():
        env = gym.make(env_id)
        env = YOLOReacherWrapper(env, use_yolo=use_yolo, verbose=False)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_environment(config: dict, use_yolo: bool = True, seed: int = None, n_envs: int = 1):
    """
    Create training environment(s) with optional YOLO wrapper and vectorization.

    Args:
        config: Environment configuration dictionary
        use_yolo: Whether to use YOLO detection (vs ground truth)
        seed: Random seed for environment
        n_envs: Number of parallel environments (1 = no vectorization)

    Returns:
        Wrapped Gymnasium environment or VecEnv
    """
    if seed is None:
        seed = 0

    if n_envs == 1:
        # Single environment (no vectorization)
        env = gym.make(
            config['env_id'],
            render_mode=config.get('render_mode', None)
        )

        # Wrap with YOLO detector
        if use_yolo:
            env = YOLOReacherWrapper(env, use_yolo=True, verbose=False)
            print("✓ Single environment created with YOLO detection")
        else:
            env = YOLOReacherWrapper(env, use_yolo=False, verbose=False)
            print("✓ Single environment created with ground truth detection")

        # Set environment seed
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        print(f"✓ Environment seed set to: {seed}")

    else:
        # Vectorized environments (parallel)
        print(f"Creating {n_envs} parallel environments...")

        # Create vectorized environments
        env = DummyVecEnv([
            make_env(config['env_id'], use_yolo, i, seed)
            for i in range(n_envs)
        ])

        print(f"✓ {n_envs} parallel environments created with {'YOLO' if use_yolo else 'ground truth'} detection")
        print(f"✓ Seeds set to: {seed} to {seed + n_envs - 1}")

    return env


def setup_callbacks(config: dict, eval_env):
    """
    Set up training callbacks for logging and model saving.

    Args:
        config: SAC configuration dictionary
        eval_env: Environment for evaluation

    Returns:
        List of callback objects
    """
    callbacks = []

    # Success rate tracking
    success_callback = SuccessRateCallback(verbose=1)
    callbacks.append(success_callback)

    # Checkpoint saving
    checkpoint_callback = CheckpointCallback(
        save_freq=config['checkpoint_freq'],
        save_path=config['save_path'],
        name_prefix='checkpoint',
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    # Custom metrics
    metrics_callback = CustomMetricsCallback(verbose=0)
    callbacks.append(metrics_callback)

    # Save best model based on evaluation
    save_best_callback = SaveBestModelCallback(
        eval_env=eval_env,
        eval_freq=config['eval_freq'],
        n_eval_episodes=config['n_eval_episodes'],
        save_path=config['save_path'],
        verbose=1
    )
    callbacks.append(save_best_callback)

    print(f"✓ Callbacks configured:")
    print(f"  - Success rate tracking")
    print(f"  - Checkpoints (every {config['checkpoint_freq']:,} steps)")
    print(f"  - Custom metrics logging")
    print(f"  - Evaluation (every {config['eval_freq']:,} steps)")
    print(f"  - Best model saving (based on evaluation)")
    print(f"  - Early stopping: DISABLED (will train to completion)")

    return callbacks


def train(
    use_yolo: bool = True,
    config_path: str = "configs/training_config.yaml",
    resume_from: str = None
):
    """
    Main training function.

    Args:
        use_yolo: Whether to use YOLO detection (True) or ground truth (False)
        config_path: Path to training configuration file
        resume_from: Path to checkpoint to resume from (optional)
    """
    print("\n" + "="*70)
    print("SAC AGENT TRAINING - REACHER OBJECT DETECTION")
    print("="*70)

    # Load configuration
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    sac_config = config['sac']
    env_config = config['environment']

    # Set random seeds for reproducibility
    seed = sac_config.get('seed', 42)
    set_seeds(seed)

    # Create training environment(s)
    print("\n" + "-"*70)
    print("Environment Setup")
    print("-"*70)
    n_envs = sac_config.get('n_envs', 1)
    train_env = create_environment(env_config, use_yolo=use_yolo, seed=seed, n_envs=n_envs)

    # Print environment info
    if n_envs > 1:
        print(f"  Observation space: {train_env.observation_space.shape}")
        print(f"  Action space: {train_env.action_space.shape}")
        print(f"  Number of parallel environments: {n_envs}")
        print(f"  Total samples per step: {n_envs}")
    else:
        print(f"  Observation space: {train_env.observation_space.shape}")
        print(f"  Action space: {train_env.action_space.shape}")
        print(f"  Action bounds: [{train_env.action_space.low[0]:.1f}, {train_env.action_space.high[0]:.1f}]")

    # Create evaluation environment
    print("\n" + "-"*70)
    print("Evaluation Environment Setup")
    print("-"*70)
    eval_env = create_environment(env_config, use_yolo=use_yolo, seed=seed + 1000, n_envs=1)
    print(f"✓ Evaluation environment created (seed: {seed + 1000})")

    # Create SAC agent
    print("\n" + "-"*70)
    print("SAC Agent Initialization")
    print("-"*70)

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        agent = SACAgent(
            env=train_env,
            config_path=config_path,
            model_path=resume_from,
            verbose=1
        )
    else:
        agent = SACAgent(
            env=train_env,
            config_path=config_path,
            verbose=1
        )

    # Setup callbacks
    print("\n" + "-"*70)
    print("Callback Setup")
    print("-"*70)
    callbacks = setup_callbacks(sac_config, eval_env)

    # Create save directory
    os.makedirs(sac_config['save_path'], exist_ok=True)

    # Training
    print("\n" + "-"*70)
    print("Training Configuration")
    print("-"*70)
    print(f"  Parallel environments: {n_envs}")
    print(f"  Total timesteps: {sac_config['total_timesteps']:,}")
    print(f"  Learning starts: {sac_config['learning_starts']:,}")
    print(f"  Learning rate: {sac_config['learning_rate']}")
    print(f"  Batch size: {sac_config['batch_size']}")
    print(f"  Buffer size: {sac_config['buffer_size']:,}")
    print(f"  Gamma: {sac_config['gamma']}")
    print(f"  Tau: {sac_config['tau']}")
    print(f"  Device: {sac_config['device']}")
    print(f"  Seed: {seed}")
    print(f"  YOLO detection: {'Enabled' if use_yolo else 'Disabled (ground truth)'}")
    if n_envs > 1:
        print(f"  Effective samples: {sac_config['total_timesteps'] * n_envs:,} (with {n_envs} envs)")
        print(f"  Expected speedup: ~{n_envs}x")

    # Get start time
    start_time = datetime.now()
    print(f"\n  Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Train the agent
    try:
        agent.train(
            total_timesteps=sac_config['total_timesteps'],
            callbacks=callbacks,
            log_interval=4,
            tb_log_name=f"SAC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        raise

    # Calculate training time
    end_time = datetime.now()
    training_time = end_time - start_time

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"  Training time: {training_time}")
    print(f"  Final timesteps: {agent.model.num_timesteps:,}")

    # Save final model
    final_model_path = os.path.join(sac_config['save_path'], 'final_model')
    agent.save(final_model_path)
    print(f"  Final model saved to: {final_model_path}.zip")

    # Save training summary (without evaluation)
    summary_path = os.path.join(sac_config['save_path'], 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SAC TRAINING SUMMARY - REACHER OBJECT DETECTION\n")
        f.write("="*70 + "\n\n")

        f.write("Training Configuration:\n")
        f.write(f"  Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Duration: {training_time}\n")
        f.write(f"  Total timesteps: {agent.model.num_timesteps:,}\n")
        f.write(f"  YOLO detection: {'Enabled' if use_yolo else 'Disabled'}\n")
        f.write(f"  Seed: {seed}\n")
        f.write(f"  Initial collision check: ENABLED\n\n")

        f.write("Hyperparameters:\n")
        f.write(f"  Parallel environments: {n_envs}\n")
        f.write(f"  Learning rate: {sac_config['learning_rate']}\n")
        f.write(f"  Batch size: {sac_config['batch_size']}\n")
        f.write(f"  Buffer size: {sac_config['buffer_size']:,}\n")
        f.write(f"  Gamma: {sac_config['gamma']}\n")
        f.write(f"  Tau: {sac_config['tau']}\n")
        f.write(f"  Device: {sac_config['device']}\n\n")

        f.write("Model Locations:\n")
        f.write(f"  Final model: {final_model_path}.zip\n")
        f.write(f"  Checkpoints: {os.path.join(sac_config['save_path'], 'checkpoint_*.zip')}\n")
        f.write(f"  TensorBoard logs: {sac_config['tensorboard_log']}\n\n")

        f.write("Note: Run evaluation script separately:\n")
        f.write(f"  python scripts/eval_sac_agent.py --model {final_model_path}.zip\n")

    print(f"  Training summary saved to: {summary_path}")

    # Print TensorBoard command
    print("\n" + "-"*70)
    print("View training progress with TensorBoard:")
    print(f"  tensorboard --logdir {sac_config['tensorboard_log']}")
    print("-"*70 + "\n")

    # Cleanup
    train_env.close()
    eval_env.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train SAC agent for Reacher Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with YOLO detection (default)
  python scripts/train_sac_agent.py

  # Train with ground truth (no YOLO)
  python scripts/train_sac_agent.py --no-yolo

  # Resume from checkpoint
  python scripts/train_sac_agent.py --resume models/sac/checkpoint_100000_steps.zip

  # Use custom config
  python scripts/train_sac_agent.py --config my_config.yaml
        """
    )

    parser.add_argument(
        '--no-yolo',
        action='store_true',
        help='Use ground truth detection instead of YOLO'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()

    train(
        use_yolo=not args.no_yolo,
        config_path=args.config,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()
