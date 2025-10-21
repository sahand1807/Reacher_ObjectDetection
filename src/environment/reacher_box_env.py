"""
Custom Reacher environment with red box target for object detection.
Based on Gymnasium's Reacher-v5 with modified target (box instead of sphere).
"""

import os
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class ReacherBoxEnv(MujocoEnv, utils.EzPickle):
    """
    2-link robotic arm environment with a red box target.

    Identical dynamics to Reacher-v5, but with:
    - Red box target (instead of sphere)
    - Top-down camera for YOLO detection
    - Random box spawning within reach radius
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, render_mode=None, **kwargs):
        utils.EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        # Get XML file path
        xml_file = os.path.join(
            os.path.dirname(__file__),
            "assets",
            "reacher_box.xml"
        )

        # Observation space: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ̇1, θ̇2, x_box, y_box]
        # Use float32 for MPS (Metal Performance Shaders) compatibility on M2 GPU
        observation_space = Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode,
            **kwargs,
        )

        self.box_position_2d = np.zeros(2, dtype=np.float32)  # Store YOLO-detected box position
        self.current_step = 0  # Track episode step count
        self.max_episode_steps = 50  # Default max steps (can be overridden by TimeLimit wrapper)

    def step(self, action):
        """
        Execute action and compute shaped reward.

        Reward components (Version 2.0 - Literature-based):
        1. Distance reward (dense, scaled): -10 * distance to target
        2. Collision penalty (safety): -50 if arm links hit target
        3. Time penalty (efficiency): -0.1 per step
        4. Control penalty (smoothness): -0.01 * ||action||^2
        5. Success bonus (sparse, at episode end): +10 + time_bonus

        Changes from v1.0:
        - Removed exploitable progress reward
        - Scaled up distance reward 10x
        - Added collision detection for arm links
        - Added time penalty for speed
        - Success bonus only at episode end with time bonus
        - Early termination on success

        Args:
            action: Joint torques [τ1, τ2]

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Increment step counter
        self.current_step += 1

        # Execute action
        self.do_simulation(action, self.frame_skip)

        # Get fingertip and target positions (xy only, ignore z)
        fingertip_pos = self.get_body_com("fingertip")[:2]
        target_pos = self.box_position_2d  # Use YOLO-detected position

        # Calculate distance to target
        distance = np.linalg.norm(fingertip_pos - target_pos)

        # === COMPONENT 1: Dense Distance Reward (Primary Signal) ===
        # Scaled by 10x to match Gymnasium Reacher-v5 scale
        # Typical range: [-1.8, 0.0] for distances [0.18m, 0m]
        reward_dist = -10.0 * distance

        # === COMPONENT 2: Collision Penalty (Safety Constraint) ===
        # Penalize if arm links (not fingertip) collide with target
        reward_collision = 0.0
        arm_collision = False

        if self.data.ncon > 0:  # If any contacts exist
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                geom1_name = self.model.geom(contact.geom1).name
                geom2_name = self.model.geom(contact.geom2).name

                # Check if arm links (not fingertip) touch target
                if 'target' in (geom1_name, geom2_name):
                    other_geom = geom1_name if geom2_name == 'target' else geom2_name
                    if 'link' in other_geom:  # link0 or link1
                        reward_collision = -10.0  # Reduced from -50.0 to encourage exploration
                        arm_collision = True
                        break

        # === COMPONENT 3: Time Penalty (Efficiency Incentive) ===
        # Small penalty to encourage reaching quickly
        reward_time = -0.1

        # === COMPONENT 4: Control Penalty (Smoothness) ===
        # Encourage energy-efficient, smooth movements
        reward_ctrl = -0.01 * np.square(action).sum()

        # === COMPONENT 5: Success Bonus (Sparse, Episode End Only) ===
        # Check if target reached (within 2cm)
        reached = distance < 0.02
        reward_reach = 0.0

        # Compute reward before success bonus
        reward = reward_dist + reward_collision + reward_time + reward_ctrl

        # Early termination on success
        terminated = reached

        # Add success bonus at episode end with time bonus
        if terminated and reached:
            # Base success bonus + bonus for remaining time
            time_bonus = (self.max_episode_steps - self.current_step) * 0.2
            reward_reach = 10.0 + time_bonus
            reward += reward_reach

        # Get observation
        observation = self._get_obs()

        # Info dictionary with all reward components
        info = {
            "distance": distance,
            "reached": reached,
            "arm_collision": arm_collision,
            "reward_dist": reward_dist,
            "reward_reach": reward_reach,
            "reward_collision": reward_collision,
            "reward_time": reward_time,
            "reward_ctrl": reward_ctrl,
            "reward_total": reward,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def reset_model(self):
        """Reset the robot and randomly spawn the red box."""
        # Keep trying until we get a valid spawn (no initial collision)
        max_attempts = 100
        for attempt in range(max_attempts):
            # Random joint angles
            qpos = self.init_qpos + self.np_random.uniform(
                low=-0.1, high=0.1, size=self.model.nq
            )

            # Random box position within reach (0.05 to 0.18m radius)
            while True:
                # Random angle and radius
                angle = self.np_random.uniform(-np.pi, np.pi)
                radius = self.np_random.uniform(0.05, 0.18)

                # Convert to Cartesian coordinates
                self.goal = np.array([radius * np.cos(angle), radius * np.sin(angle)])

                # Ensure within workspace bounds
                if np.linalg.norm(self.goal) <= 0.2:
                    break

            # Set box position (indices 4, 5 correspond to target_x, target_y)
            qpos[-2:] = self.goal

            # Zero velocities
            qvel = self.init_qvel + self.np_random.uniform(
                low=-0.005, high=0.005, size=self.model.nv
            )
            qvel[-2:] = 0

            self.set_state(qpos, qvel)

            # Step physics once to update contacts
            self.do_simulation(np.zeros(2), 1)

            # Check for initial collision between arm links and box
            initial_collision = False
            if self.data.ncon > 0:
                for i in range(self.data.ncon):
                    contact = self.data.contact[i]
                    geom1_name = self.model.geom(contact.geom1).name
                    geom2_name = self.model.geom(contact.geom2).name

                    # Check if arm links (not fingertip) touch target
                    if 'target' in (geom1_name, geom2_name):
                        other_geom = geom1_name if geom2_name == 'target' else geom2_name
                        if 'link' in other_geom:  # link0 or link1
                            initial_collision = True
                            break

            # If no initial collision, we have a valid spawn
            if not initial_collision:
                break

        # If we couldn't find a valid spawn after max_attempts, use the last one anyway
        # (very unlikely to happen)

        # "Detect" box with YOLO (for now, ground truth)
        # In full system, this will be replaced by actual YOLO detection
        self.box_position_2d = self.goal.copy().astype(np.float32)

        # Reset step counter
        self.current_step = 0

        return self._get_obs()

    def _get_obs(self):
        """
        Observation: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ̇1, θ̇2, x_box, y_box]

        Note: x_box, y_box are CONSTANT throughout episode (YOLO detection at start)
        """
        theta = self.data.qpos.flat[:2]

        return np.concatenate([
            np.cos(theta),          # cos(θ1), cos(θ2)
            np.sin(theta),          # sin(θ1), sin(θ2)
            self.data.qvel.flat[:2],  # θ̇1, θ̇2
            self.box_position_2d,    # x_box, y_box (CONSTANT from YOLO)
        ]).astype(np.float32)  # Convert to float32 for MPS compatibility

    def render_top_view(self):
        """Render top-down view for YOLO detection."""
        # Save current camera
        original_camera = self.mujoco_renderer.camera_id

        # Set to top view camera
        self.mujoco_renderer.camera_id = self.model.camera("top_view").id

        # Render
        img = self.mujoco_renderer.render(render_mode="rgb_array")

        # Restore original camera
        self.mujoco_renderer.camera_id = original_camera

        return img
