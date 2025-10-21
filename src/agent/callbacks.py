"""
Custom callbacks for SAC training.

Provides specialized callbacks for monitoring and controlling the training process:
- Success rate tracking (reaching within 2cm)
- Best model saving based on performance
- Custom metric logging to TensorBoard
- Early stopping based on success criteria
"""

import os
import numpy as np
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization


class SuccessRateCallback(BaseCallback):
    """
    Tracks success rate during training episodes.

    Success is defined as reaching within 2cm of the target.
    Logs success rate, final distances, and episode rewards to TensorBoard.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_distances = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        """
        Called at every step.
        Accumulates episode statistics and logs when episode ends.
        """
        # Get current info
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]

            # Accumulate reward and length
            self.current_episode_reward += self.locals['rewards'][0]
            self.current_episode_length += 1

            # Check if episode ended
            done = self.locals['dones'][0]
            if done:
                # Get final distance
                distance = info.get('distance', float('inf'))
                success = distance < 0.02  # 2cm threshold

                # Store episode statistics
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.episode_successes.append(float(success))
                self.episode_distances.append(distance * 100)  # cm

                # Log to TensorBoard
                self.logger.record('rollout/ep_success', float(success))
                self.logger.record('rollout/ep_final_distance_cm', distance * 100)
                self.logger.record('rollout/ep_rew_mean', self.current_episode_reward)
                self.logger.record('rollout/ep_len_mean', self.current_episode_length)

                # Log running averages (last 100 episodes)
                if len(self.episode_successes) >= 100:
                    recent_successes = self.episode_successes[-100:]
                    recent_distances = self.episode_distances[-100:]
                    recent_rewards = self.episode_rewards[-100:]

                    self.logger.record('rollout/success_rate_100ep', np.mean(recent_successes) * 100)
                    self.logger.record('rollout/mean_distance_100ep', np.mean(recent_distances))
                    self.logger.record('rollout/mean_reward_100ep', np.mean(recent_rewards))

                # Print progress
                if self.verbose > 0:
                    print(f"Episode: Reward={self.current_episode_reward:.2f}, "
                          f"Length={self.current_episode_length}, "
                          f"Distance={distance*100:.1f}cm, "
                          f"Success={'✓' if success else '✗'}")

                # Reset counters
                self.current_episode_reward = 0
                self.current_episode_length = 0

        return True

    def get_success_rate(self, n_episodes: int = 100) -> float:
        """
        Get success rate over last n episodes.

        Args:
            n_episodes: Number of recent episodes to consider

        Returns:
            Success rate as percentage (0-100)
        """
        if len(self.episode_successes) == 0:
            return 0.0
        recent = self.episode_successes[-n_episodes:]
        return np.mean(recent) * 100


class SaveBestModelCallback(BaseCallback):
    """
    Saves the best model based on success rate during training.

    Evaluates the model periodically and saves when performance improves.
    """

    def __init__(
        self,
        save_path: str,
        eval_env: VecEnv,
        n_eval_episodes: int = 10,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 1
    ):
        """
        Args:
            save_path: Directory to save models
            eval_env: Environment for evaluation
            n_eval_episodes: Number of episodes for evaluation
            eval_freq: Evaluate every n steps
            deterministic: Use deterministic policy for evaluation
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic

        self.best_success_rate = -np.inf
        self.best_mean_reward = -np.inf
        self.evaluations_success_rate = []
        self.evaluations_rewards = []

        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Evaluate and save model if performance improves.
        """
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation
            episode_rewards = []
            episode_successes = []
            episode_distances = []

            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()  # Gymnasium API: returns (obs, info)
                done = False
                episode_reward = 0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)  # Gymnasium API: 5 returns
                    done = terminated or truncated
                    episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward

                    if done:
                        distance = info[0].get('distance', float('inf')) if isinstance(info, list) else info.get('distance', float('inf'))
                        success = distance < 0.02
                        episode_successes.append(float(success))
                        episode_distances.append(distance * 100)

                episode_rewards.append(episode_reward)

            # Compute metrics
            mean_reward = np.mean(episode_rewards)
            success_rate = np.mean(episode_successes) * 100
            mean_distance = np.mean(episode_distances)

            # Log to TensorBoard
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/success_rate', success_rate)
            self.logger.record('eval/mean_distance_cm', mean_distance)

            # Store history
            self.evaluations_success_rate.append(success_rate)
            self.evaluations_rewards.append(mean_reward)

            # Save if best
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                model_path = os.path.join(self.save_path, 'best_model')
                self.model.save(model_path)

                if self.verbose > 0:
                    print(f"\n{'='*60}")
                    print(f"New best model! Success rate: {success_rate:.1f}%")
                    print(f"Saved to: {model_path}.zip")
                    print(f"{'='*60}\n")

            if self.verbose > 0:
                print(f"\n--- Evaluation at {self.n_calls} steps ---")
                print(f"Mean reward: {mean_reward:.2f}")
                print(f"Success rate: {success_rate:.1f}%")
                print(f"Mean distance: {mean_distance:.2f} cm")
                print(f"Best success rate so far: {self.best_success_rate:.1f}%\n")

        return True


class CheckpointCallback(BaseCallback):
    """
    Saves model checkpoints at regular intervals.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = 'checkpoint',
        verbose: int = 0
    ):
        """
        Args:
            save_freq: Save checkpoint every n steps
            save_path: Directory to save checkpoints
            name_prefix: Prefix for checkpoint files
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Save checkpoint at specified intervals.
        """
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(
                self.save_path,
                f'{self.name_prefix}_{self.n_calls}_steps'
            )
            self.model.save(model_path)

            if self.verbose > 0:
                print(f"Checkpoint saved: {model_path}.zip")

        return True


class EarlyStoppingCallback(BaseCallback):
    """
    Stops training early if success rate reaches target.
    """

    def __init__(
        self,
        success_threshold: float = 90.0,
        n_episodes: int = 100,
        verbose: int = 1
    ):
        """
        Args:
            success_threshold: Stop if success rate exceeds this (percentage)
            n_episodes: Compute success rate over last n episodes
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.n_episodes = n_episodes
        self.episode_successes = []

    def _on_step(self) -> bool:
        """
        Check success rate and stop if threshold reached.
        """
        # Get episode info
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            done = self.locals['dones'][0]

            if done:
                distance = info.get('distance', float('inf'))
                success = distance < 0.02
                self.episode_successes.append(float(success))

                # Check if we have enough episodes
                if len(self.episode_successes) >= self.n_episodes:
                    recent = self.episode_successes[-self.n_episodes:]
                    success_rate = np.mean(recent) * 100

                    if success_rate >= self.success_threshold:
                        if self.verbose > 0:
                            print(f"\n{'='*60}")
                            print(f"Early stopping triggered!")
                            print(f"Success rate: {success_rate:.1f}% >= {self.success_threshold}%")
                            print(f"{'='*60}\n")
                        return False  # Stop training

        return True  # Continue training


class CustomMetricsCallback(BaseCallback):
    """
    Logs additional custom metrics to TensorBoard.

    Tracks action magnitudes, velocities, and other environment-specific metrics.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_action_magnitudes = []
        self.current_actions = []

    def _on_step(self) -> bool:
        """
        Log custom metrics.
        """
        # Track action magnitudes
        action = self.locals.get('actions', [np.zeros(2)])[0]
        action_magnitude = np.linalg.norm(action)
        self.current_actions.append(action_magnitude)

        # Log when episode ends
        if self.locals.get('dones', [False])[0]:
            if len(self.current_actions) > 0:
                mean_action = np.mean(self.current_actions)
                self.logger.record('metrics/mean_action_magnitude', mean_action)
                self.current_actions = []

            # Log observation statistics if available
            if len(self.locals.get('infos', [])) > 0:
                info = self.locals['infos'][0]

                # Log velocities if available
                obs = self.locals.get('new_obs', [np.zeros(8)])[0]
                if len(obs) >= 6:
                    velocities = obs[4:6]
                    self.logger.record('metrics/mean_velocity', np.mean(np.abs(velocities)))

        return True
