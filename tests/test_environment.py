"""Test script for custom Reacher environment with red box."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from src.environment import ReacherBoxEnv


def test_environment():
    """Test the custom Reacher environment."""
    print("=" * 60)
    print("Testing ReacherObjectDetection-v0 Environment")
    print("=" * 60)

    # Create environment
    env = gym.make("ReacherObjectDetection-v0", render_mode="rgb_array")

    print(f"\n✓ Environment created successfully")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"\n✓ Environment reset")
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Initial observation: {obs}")
    print(f"  Box position (YOLO): x={obs[6]:.4f}, y={obs[7]:.4f}")
    print(f"  Distance to box: {np.linalg.norm(obs[6:8]):.4f}m")

    # Render top-down view
    top_view = env.unwrapped.render_top_view()
    print(f"\n✓ Top-down view rendered")
    print(f"  Image shape: {top_view.shape}")

    # Take a few random steps
    print(f"\n✓ Testing random actions...")
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Step {step+1}: reward={reward:.4f}, terminated={terminated}")

    print(f"\n✓ Total reward over 5 steps: {total_reward:.4f}")

    # Visualize top-down view
    print(f"\n✓ Saving top-down view visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Reset for 3 different random box positions
    for i in range(3):
        env.reset()
        img = env.unwrapped.render_top_view()
        axes[i].imshow(img)
        axes[i].set_title(f"Random Box Position {i+1}")
        axes[i].axis('off')

    plt.tight_layout()
    output_path = "data/test_environment_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")

    env.close()
    print(f"\n{'=' * 60}")
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_environment()
