"""Custom Gymnasium environment for Reacher with object detection."""

from gymnasium.envs.registration import register
from src.environment.reacher_box_env import ReacherBoxEnv

# Register custom environment
register(
    id="ReacherObjectDetection-v0",
    entry_point="src.environment.reacher_box_env:ReacherBoxEnv",
    max_episode_steps=50,
)

__all__ = ["ReacherBoxEnv"]
