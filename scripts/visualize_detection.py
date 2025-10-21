"""
Visualize YOLO detection with ground truth and observations.

Shows:
- Environment rendering
- YOLO bounding box prediction
- Ground truth box position
- Detected vs ground truth positions
- Observation values
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gymnasium as gym
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import ReacherBoxEnv
from src.environment.wrappers import YOLOReacherWrapper


def detect_red_box_visual(img_rgb):
    """Detect where box actually appears (ground truth using color detection)."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_red1 = np.array([0, 150, 150])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 150])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    valid = [c for c in contours if 100 < cv2.contourArea(c) < 5000]
    if not valid:
        return None

    largest = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    return x, y, w, h


def world_to_image_pixel(x_world, y_world, img_shape):
    """Convert world coordinates to pixel coordinates for visualization."""
    # Inverse of the corrected transformation
    # x_world = (x_img - 0.5) * 1.4136  =>  x_img = x_world / 1.4136 + 0.5
    # y_world = (y_img - 0.5) * -1.4136  =>  y_img = -y_world / 1.4136 + 0.5

    x_img_norm = x_world / 1.4136 + 0.5
    y_img_norm = -y_world / 1.4136 + 0.5  # Flip back

    img_h, img_w = img_shape[:2]
    x_pixel = int(x_img_norm * img_w)
    y_pixel = int(y_img_norm * img_h)

    return x_pixel, y_pixel


def visualize_episode(num_samples=6, use_yolo=True):
    """
    Visualize YOLO detection on multiple samples.

    Args:
        num_samples: Number of samples to visualize
        use_yolo: Use YOLO detection (True) or ground truth (False)
    """
    # Create environment
    env = gym.make("ReacherObjectDetection-v0", render_mode="rgb_array")
    env = YOLOReacherWrapper(env, use_yolo=use_yolo, verbose=False)

    # Load YOLO model for visualization
    yolo_model = YOLO("results/yolo/train/weights/best.pt")

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()


    for i in range(num_samples):
        # Reset environment
        obs, _ = env.reset()

        # Get unwrapped environment to access rendering
        base_env = env.unwrapped

        # Get ground truth position (where Mujoco thinks box is)
        gt_mujoco = base_env.goal.copy()

        # Render image
        img_rgb = base_env.render_top_view()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Detect visual ground truth (where box actually appears)
        gt_bbox = detect_red_box_visual(img_rgb)

        # Run YOLO detection
        results = yolo_model(img_bgr, verbose=False)

        # Get YOLO prediction in world coordinates
        yolo_world_pos = env.detected_box_position.copy()

        # Extract observation components
        cos_theta = obs[:2]
        sin_theta = obs[2:4]
        velocities = obs[4:6]
        box_pos_obs = obs[6:8]

        # Get fingertip position
        fingertip_pos = base_env.get_body_com("fingertip")[:2]

        # Visualize
        ax = axes[i]
        ax.imshow(img_rgb)

        # Draw YOLO bounding box (green rectangle)
        if len(results[0].boxes) > 0:
            bbox = results[0].boxes.xywhn[0]
            img_h, img_w = img_rgb.shape[:2]
            x_center = bbox[0].item() * img_w
            y_center = bbox[1].item() * img_h
            width = bbox[2].item() * img_w
            height = bbox[3].item() * img_h

            x1 = x_center - width / 2
            y1 = y_center - height / 2

            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=3, edgecolor='lime', facecolor='none',
                label='YOLO Detection'
            )
            ax.add_patch(rect)

        # Simple title
        title = f"Sample {i+1}"
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()

    # Save figure
    output_path = f"visualization_detection_{'yolo' if use_yolo else 'gt'}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    plt.show()

    env.close()


def visualize_rollout(num_steps=10, use_yolo=True):
    """
    Visualize a sequence of steps showing the arm moving towards the target.

    Args:
        num_steps: Number of steps to visualize
        use_yolo: Use YOLO detection (True) or ground truth (False)
    """
    # Create environment
    env = gym.make("ReacherObjectDetection-v0", render_mode="rgb_array")
    env = YOLOReacherWrapper(env, use_yolo=use_yolo, verbose=False)

    # Reset
    obs, _ = env.reset()

    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()


    base_env = env.unwrapped
    box_pos_world = env.detected_box_position.copy()

    for step in range(num_steps):
        # Random action (in real case, this would come from policy)
        action = env.action_space.sample()

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)

        # Get positions
        fingertip_pos = base_env.get_body_com("fingertip")[:2]
        distance = info['distance'] * 100  # Convert to cm

        # Render
        img_rgb = base_env.render_top_view()

        # Visualize
        ax = axes[step]
        ax.imshow(img_rgb)

        # Mark target position (green circle)
        target_pixel = world_to_image_pixel(box_pos_world[0], box_pos_world[1], img_rgb.shape)
        ax.plot(target_pixel[0], target_pixel[1], 'o', color='lime', markersize=12,
                markeredgewidth=2, markerfacecolor='none', label='Target')

        # Mark fingertip (magenta dot)
        fingertip_pixel = world_to_image_pixel(fingertip_pos[0], fingertip_pos[1], img_rgb.shape)
        ax.plot(fingertip_pixel[0], fingertip_pixel[1], 'o', color='magenta',
                markersize=8, label='Fingertip')

        # Title
        reached = "✓" if info['reached'] else ""
        title = f"Step {step+1} {reached}"
        ax.set_title(title, fontsize=11)
        ax.axis('off')

        if terminated or truncated:
            break

    plt.tight_layout()

    # Save figure
    output_path = f"visualization_rollout_{'yolo' if use_yolo else 'gt'}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    plt.show()

    env.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize YOLO detection and environment")
    parser.add_argument('--mode', choices=['detection', 'rollout', 'both'], default='both',
                        help='Visualization mode')
    parser.add_argument('--yolo', action='store_true', default=True,
                        help='Use YOLO detection (default: True)')
    parser.add_argument('--gt', action='store_true',
                        help='Use ground truth instead of YOLO')
    parser.add_argument('--samples', type=int, default=6,
                        help='Number of samples for detection mode')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of steps for rollout mode')

    args = parser.parse_args()

    use_yolo = not args.gt

    if args.mode in ['detection', 'both']:
        visualize_episode(num_samples=args.samples, use_yolo=use_yolo)

    if args.mode in ['rollout', 'both']:
        visualize_rollout(num_steps=args.steps, use_yolo=use_yolo)


if __name__ == "__main__":
    main()
