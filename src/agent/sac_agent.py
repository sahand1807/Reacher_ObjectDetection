"""
SAC Agent wrapper for Reacher Object Detection environment.

This module provides a clean interface to the Stable Baselines 3 SAC algorithm
for training a robotic arm to reach detected objects.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList


class SACAgent:
    """
    Soft Actor-Critic agent wrapper for the Reacher environment.

    This class handles:
    - SAC model creation with config-based hyperparameters
    - Training with custom callbacks
    - Model saving/loading
    - Evaluation and inference
    """

    def __init__(
        self,
        env: gym.Env,
        config_path: str = "configs/training_config.yaml",
        model_path: Optional[str] = None,
        verbose: int = 1
    ):
        """
        Initialize SAC agent.

        Args:
            env: Gymnasium environment (should be wrapped with YOLOReacherWrapper)
            config_path: Path to training configuration YAML
            model_path: Path to pretrained model (if loading existing model)
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
        """
        self.env = env
        self.verbose = verbose

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config['sac']
        self.env_config = config.get('environment', {})

        # Create or load SAC model
        if model_path and os.path.exists(model_path):
            print(f"Loading existing model from: {model_path}")
            self.model = SAC.load(model_path, env=env)
        else:
            print("Creating new SAC model...")
            self.model = self._create_model()

        print(f"SAC Agent initialized:")
        print(f"  - Device: {self.config['device']}")
        print(f"  - Learning rate: {self.config['learning_rate']}")
        print(f"  - Buffer size: {self.config['buffer_size']:,}")
        print(f"  - Batch size: {self.config['batch_size']}")

    def _create_model(self) -> SAC:
        """
        Create SAC model with hyperparameters from config.

        Returns:
            Configured SAC model
        """
        return SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.config['learning_rate'],
            buffer_size=self.config['buffer_size'],
            learning_starts=self.config['learning_starts'],
            batch_size=self.config['batch_size'],
            tau=self.config['tau'],
            gamma=self.config['gamma'],
            train_freq=self.config['train_freq'],
            gradient_steps=self.config['gradient_steps'],
            device=self.config['device'],
            tensorboard_log=self.config['tensorboard_log'],
            verbose=self.verbose,
        )

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callbacks: Optional[list] = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC"
    ) -> None:
        """
        Train the SAC agent.

        Args:
            total_timesteps: Number of timesteps to train (uses config if None)
            callbacks: List of callback objects for logging/checkpointing
            log_interval: How often to log training info (in episodes)
            tb_log_name: Name for TensorBoard log
        """
        if total_timesteps is None:
            total_timesteps = self.config['total_timesteps']

        print("\n" + "="*70)
        print("Starting SAC Training")
        print("="*70)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Environment: {self.env.spec.id if hasattr(self.env, 'spec') else 'Custom'}")
        print(f"Observation space: {self.env.observation_space.shape}")
        print(f"Action space: {self.env.action_space.shape}")
        print("="*70 + "\n")

        # Combine callbacks if provided
        callback = CallbackList(callbacks) if callbacks else None

        # Train model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            progress_bar=True
        )

        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)

    def save(self, save_path: str) -> None:
        """
        Save trained model to disk.

        Args:
            save_path: Path to save model (without extension)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.model.save(save_path)
        print(f"Model saved to: {save_path}.zip")

    def load(self, model_path: str) -> None:
        """
        Load trained model from disk.

        Args:
            model_path: Path to saved model
        """
        self.model = SAC.load(model_path, env=self.env)
        print(f"Model loaded from: {model_path}")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get action from policy for given observation.

        Args:
            observation: Current observation from environment
            deterministic: If True, use deterministic policy (mean action)
                         If False, sample from policy distribution

        Returns:
            action: Action to take
            state: Internal state (for recurrent policies, None for SAC)
        """
        action, state = self.model.predict(
            observation,
            deterministic=deterministic
        )
        return action, state

    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = False,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate agent performance over multiple episodes.

        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render environment during evaluation
            deterministic: Use deterministic policy

        Returns:
            Dictionary with evaluation metrics:
                - mean_reward: Average total reward per episode
                - std_reward: Standard deviation of rewards
                - mean_length: Average episode length
                - success_rate: Percentage of successful reaches (distance < 2cm)
                - mean_final_distance: Average distance to target at episode end
        """
        print(f"\nEvaluating agent over {n_episodes} episodes...")

        # Check if vectorized environment
        from stable_baselines3.common.vec_env import VecEnv
        is_vec_env = isinstance(self.env, VecEnv)

        episode_rewards = []
        episode_lengths = []
        successes = []
        final_distances = []

        for episode in range(n_episodes):
            # Handle both vectorized and non-vectorized environments
            if is_vec_env:
                obs = self.env.reset()
            else:
                obs, _ = self.env.reset()

            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)

                if is_vec_env:
                    obs, reward, done_array, info = self.env.step(action)
                    done = done_array[0]
                    reward = reward[0]
                    info = info[0] if isinstance(info, list) else info
                else:
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                if render and not is_vec_env:
                    self.env.render()

            # Check success (distance < 2cm = 0.02m)
            final_distance = info.get('distance', float('inf'))
            success = final_distance < 0.02

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            successes.append(success)
            final_distances.append(final_distance * 100)  # Convert to cm

            if self.verbose > 0:
                print(f"  Episode {episode+1}/{n_episodes}: "
                      f"Reward={episode_reward:.2f}, "
                      f"Length={episode_length}, "
                      f"Distance={final_distance*100:.1f}cm, "
                      f"Success={'✓' if success else '✗'}")

        # Compute statistics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': np.mean(successes) * 100,
            'mean_final_distance': np.mean(final_distances),
            'std_final_distance': np.std(final_distances),
        }

        print(f"\n{'='*50}")
        print("Evaluation Results:")
        print(f"{'='*50}")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Length: {results['mean_length']:.1f} steps")
        print(f"  Success Rate: {results['success_rate']:.1f}%")
        print(f"  Final Distance: {results['mean_final_distance']:.2f} ± {results['std_final_distance']:.2f} cm")
        print(f"{'='*50}\n")

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information
        """
        return {
            'policy': 'MlpPolicy',
            'learning_rate': self.model.learning_rate,
            'buffer_size': self.model.buffer_size,
            'batch_size': self.model.batch_size,
            'gamma': self.model.gamma,
            'tau': self.model.tau,
            'device': str(self.model.device),
            'n_calls': self.model.num_timesteps,
        }
