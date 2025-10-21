"""
YOLO wrapper for Reacher environment.

Integrates YOLO object detection with the Reacher environment:
- Runs YOLO detection once per episode at reset
- Converts image coordinates to world coordinates
- Maintains constant box position throughout episode
- Inherits dynamic observations (joints, velocities) from base env
"""

import numpy as np
import gymnasium as gym
from pathlib import Path
from ultralytics import YOLO


class YOLOReacherWrapper(gym.Wrapper):
    """
    Wraps ReacherBoxEnv to integrate YOLO detection.

    Key features:
    - YOLO detection runs ONCE per episode at reset()
    - Box position is CONSTANT throughout episode (realistic)
    - Other observations (angles, velocities) update normally
    - Supports both YOLO and ground truth modes for validation
    """

    def __init__(
        self,
        env,
        yolo_model_path="results/yolo/train/weights/best.pt",
        use_yolo=True,
        verbose=False
    ):
        """
        Initialize YOLO wrapper.

        Args:
            env: Base ReacherBoxEnv environment
            yolo_model_path: Path to trained YOLO model (.pt file)
            use_yolo: If True, use YOLO detection; if False, use ground truth
            verbose: Print detection information
        """
        super().__init__(env)

        self.use_yolo = use_yolo
        self.verbose = verbose

        # Load YOLO model if needed
        if self.use_yolo:
            model_path = Path(yolo_model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"YOLO model not found: {model_path}")

            if self.verbose:
                print(f"Loading YOLO model from {model_path}")

            self.yolo_model = YOLO(str(model_path))

            if self.verbose:
                print("✓ YOLO model loaded")
        else:
            self.yolo_model = None
            if self.verbose:
                print("Using ground truth (YOLO disabled)")

        # Store detected box position for the episode
        self.detected_box_position = None
        self.ground_truth_position = None

    def reset(self, **kwargs):
        """
        Reset environment and run YOLO detection.

        Returns:
            observation: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ̇1, θ̇2, x_box, y_box]
        """
        # Reset base environment
        obs = self.env.reset(**kwargs)

        # Store ground truth for comparison (access unwrapped env)
        self.ground_truth_position = self.env.unwrapped.goal.copy().astype(np.float32)

        # Run YOLO detection or use ground truth
        if self.use_yolo:
            # Render top-down view using unwrapped env (same as dataset generation)
            img = self.env.unwrapped.render_top_view()

            # Detect box with YOLO
            self.detected_box_position = self._detect_box(img)

            if self.verbose:
                gt = self.ground_truth_position
                det = self.detected_box_position
                error = np.linalg.norm(det - gt)
                print(f"YOLO Detection:")
                print(f"  Ground truth: ({gt[0]:.4f}, {gt[1]:.4f})")
                print(f"  YOLO pred:    ({det[0]:.4f}, {det[1]:.4f})")
                print(f"  Error:        {error*1000:.2f} mm")
        else:
            # Use ground truth (already float32 from line 82)
            self.detected_box_position = self.ground_truth_position.copy()

            if self.verbose:
                print(f"Using ground truth: ({self.detected_box_position[0]:.4f}, "
                      f"{self.detected_box_position[1]:.4f})")

        # Update environment's box position (access unwrapped env)
        self.env.unwrapped.box_position_2d = self.detected_box_position

        # IMPORTANT: Also update env.goal to match the detected visual position
        # This ensures consistency between Mujoco state and what YOLO sees
        self.env.unwrapped.goal = self.detected_box_position.copy()

        # Get updated observation with YOLO coordinates (access unwrapped)
        obs = self.env.unwrapped._get_obs()

        return obs, {}

    def step(self, action):
        """
        Step environment (box position remains constant).

        Args:
            action: Joint torques [τ1, τ2]

        Returns:
            observation: Updated state (with constant box position)
            reward: Shaped reward
            terminated: Episode termination flag
            truncated: Episode truncation flag
            info: Additional information
        """
        # Execute action in base environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Verify box position hasn't changed (sanity check)
        assert np.allclose(self.env.unwrapped.box_position_2d, self.detected_box_position), \
            "Box position changed during episode!"

        return obs, reward, terminated, truncated, info

    def _detect_box(self, img):
        """
        Run YOLO detection and convert to world coordinates.

        Args:
            img: Top-down RGB image from environment

        Returns:
            box_position: [x_world, y_world] in meters
        """
        # Convert RGB to BGR (YOLO was trained on BGR images from cv2)
        import cv2
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Run YOLO inference
        results = self.yolo_model(img_bgr, verbose=False)

        if len(results[0].boxes) == 0:
            # No detection - fallback to ground truth with warning
            print("⚠️  WARNING: YOLO failed to detect box, using ground truth")
            return self.ground_truth_position.copy()  # Already float32

        # Get bbox center in normalized image coordinates [0, 1]
        bbox = results[0].boxes.xywhn[0]  # [x, y, w, h] normalized
        x_img = bbox[0].item()
        y_img = bbox[1].item()

        # Convert to world coordinates
        x_world, y_world = self._image_to_world(x_img, y_img)

        return np.array([x_world, y_world], dtype=np.float32)  # Use float32 for MPS compatibility

    def _image_to_world(self, x_img, y_img):
        """
        Convert normalized image coordinates to world coordinates.

        Coordinate systems:
        - Image: [0, 1] × [0, 1] (normalized, origin top-left)
        - World: Mujoco world coordinates (meters, origin at robot base)

        Args:
            x_img: Normalized x-coordinate [0, 1]
            y_img: Normalized y-coordinate [0, 1]

        Returns:
            (x_world, y_world) in meters

        Note: Camera calibration determined the correct transformation.
        The Y-axis is flipped (image Y increases downward, world Y increases upward).
        """
        # Calibrated transformation (from scripts/calibrate_camera.py)
        # Scale factor: 1.4136 (determined empirically)
        # Y-axis flip: -1 (image Y down = world Y up)
        x_world = (x_img - 0.5) * 1.4136
        y_world = (y_img - 0.5) * -1.4136  # Negative for Y-axis flip

        return x_world, y_world

    def get_detection_info(self):
        """
        Get information about current detection.

        Returns:
            dict with detection info
        """
        return {
            'detected_position': self.detected_box_position.copy(),
            'ground_truth': self.ground_truth_position.copy(),
            'error': np.linalg.norm(self.detected_box_position - self.ground_truth_position),
            'using_yolo': self.use_yolo
        }
