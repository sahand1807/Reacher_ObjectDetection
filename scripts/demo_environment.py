"""
Demo script showing YOLO detection and environment rollout side-by-side.

Left: Static image with YOLO detection
Right: Animated rollout with random actions
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


def world_to_image_pixel(x_world, y_world, img_shape):
    """Convert world coordinates to pixel coordinates."""
    x_img_norm = x_world / 1.4136 + 0.5
    y_img_norm = -y_world / 1.4136 + 0.5

    img_h, img_w = img_shape[:2]
    x_pixel = int(x_img_norm * img_w)
    y_pixel = int(y_img_norm * img_h)

    return x_pixel, y_pixel


def demo_environment(box_x=0.10, box_y=0.15, num_steps=50):
    """
    Demo environment with YOLO detection and rollout.

    Args:
        box_x: Box X position in meters
        box_y: Box Y position in meters
        num_steps: Number of steps to simulate
    """
    # Create environment
    env = gym.make("ReacherObjectDetection-v0", render_mode="rgb_array")
    env = YOLOReacherWrapper(env, use_yolo=True, verbose=False)

    # Load YOLO model
    yolo_model = YOLO("results/yolo/train/weights/best.pt")

    # Reset and set box to specific position
    obs, _ = env.reset()

    # Iteratively adjust Mujoco position until YOLO sees the desired visual position
    print(f"Adjusting box position to achieve visual position ({box_x:.3f}, {box_y:.3f})...")

    mujoco_pos = np.array([box_x, box_y])  # Initial guess

    for iteration in range(10):  # Max 10 iterations
        # Set Mujoco position
        qpos = env.unwrapped.data.qpos.copy()
        qpos[-2:] = mujoco_pos
        qvel = env.unwrapped.data.qvel.copy()
        env.unwrapped.set_state(qpos, qvel)

        # Render and detect
        img_rgb = env.unwrapped.render_top_view()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        results = yolo_model(img_bgr, verbose=False)

        if len(results[0].boxes) > 0:
            bbox = results[0].boxes.xywhn[0]
            x_img = bbox[0].item()
            y_img = bbox[1].item()
            x_detected = (x_img - 0.5) * 1.4136
            y_detected = (y_img - 0.5) * -1.4136

            # Calculate error
            error = np.linalg.norm([x_detected - box_x, y_detected - box_y])

            print(f"  Iteration {iteration+1}: Detected ({x_detected:.3f}, {y_detected:.3f}), error={error*1000:.1f}mm")

            if error < 0.005:  # Within 5mm
                print(f"  ✓ Converged!")
                break

            # Adjust Mujoco position (inverse correction)
            mujoco_pos[0] += (box_x - x_detected) * 0.8  # Damping factor
            mujoco_pos[1] += (box_y - y_detected) * 0.8
        else:
            print(f"  Iteration {iteration+1}: No detection!")
            break

    # Final detection
    img_rgb = env.unwrapped.render_top_view()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results = yolo_model(img_bgr, verbose=False)

    if len(results[0].boxes) > 0:
        bbox = results[0].boxes.xywhn[0]
        x_img = bbox[0].item()
        y_img = bbox[1].item()
        x_world = (x_img - 0.5) * 1.4136
        y_world = (y_img - 0.5) * -1.4136
        env.detected_box_position = np.array([x_world, y_world])
        env.unwrapped.box_position_2d = env.detected_box_position
        env.unwrapped.goal = env.detected_box_position.copy()

        print(f"Final visual position: ({x_world:.3f}, {y_world:.3f})")
        print()

    obs = env.unwrapped._get_obs()

    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 8))
    ax_left = plt.subplot(1, 2, 1)
    ax_right = plt.subplot(1, 2, 2)

    # LEFT: Static detection view
    ax_left.imshow(img_rgb)

    # Draw YOLO bounding box
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
            linewidth=3, edgecolor='lime', facecolor='none'
        )
        ax_left.add_patch(rect)

    ax_left.set_title('YOLO Detection', fontsize=14, fontweight='bold')
    ax_left.axis('off')

    # RIGHT: Animated rollout
    img_display = ax_right.imshow(img_rgb)
    ax_right.set_title('Environment Rollout', fontsize=14, fontweight='bold')
    ax_right.axis('off')

    # Text for observations and actions
    text_str = _format_obs_action(obs, np.array([0.0, 0.0]), 0, 0.0, 0.0)
    text_box = ax_right.text(0.02, 0.98, text_str,
                            transform=ax_right.transAxes,
                            fontsize=10,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            family='monospace')

    plt.tight_layout()
    plt.ion()
    plt.show()

    # Collect frames for GIF
    frames = []

    # Run rollout
    for step in range(num_steps):
        # Random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Get new frame
        img_rgb = env.unwrapped.render_top_view()

        # Update right panel
        img_display.set_data(img_rgb)

        # Update text
        distance = info['distance'] * 100  # cm
        text_str = _format_obs_action(obs, action, step+1, reward, distance)
        text_box.set_text(text_str)

        # Capture frame for GIF BEFORE plt.pause
        fig.canvas.draw()
        fig.canvas.flush_events()
        # Use buffer_rgba for Mac compatibility
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)
        # Convert RGBA to RGB
        frame_rgb = frame[:, :, :3].copy()  # Important: make a copy!
        frames.append(frame_rgb)

        plt.pause(0.05)

        if terminated or truncated:
            break

    plt.ioff()

    # Save as GIF using Pillow
    from PIL import Image
    output_path = "demo_environment.gif"

    print(f"\nCollected {len(frames)} frames")

    if len(frames) == 0:
        print("Warning: No frames collected!")
        return

    # Debug: Check if frames are different
    if len(frames) > 1:
        diff = np.abs(frames[0].astype(float) - frames[-1].astype(float)).mean()
        print(f"Average pixel difference between first and last frame: {diff:.2f}")
        if diff < 1.0:
            print("Warning: Frames appear to be identical!")

    # Convert frames to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,  # 100ms per frame = 10 FPS
        loop=0
    )

    print(f"✓ Demo saved to: {output_path} ({len(pil_frames)} frames)")

    plt.show()
    env.close()


def _format_obs_action(obs, action, step, reward, distance):
    """Format observation and action for display."""
    cos_theta = obs[:2]
    sin_theta = obs[2:4]
    velocities = obs[4:6]
    box_pos = obs[6:8]

    theta1 = np.arctan2(sin_theta[0], cos_theta[0]) * 180 / np.pi
    theta2 = np.arctan2(sin_theta[1], cos_theta[1]) * 180 / np.pi

    text = f"Step: {step}\n"
    text += f"Reward: {reward:6.2f}\n"
    text += f"Distance: {distance:5.1f} cm\n"
    text += f"\n"
    text += f"Joint Angles:\n"
    text += f"  θ₁: {theta1:6.1f}°\n"
    text += f"  θ₂: {theta2:6.1f}°\n"
    text += f"\n"
    text += f"Velocities:\n"
    text += f"  ω₁: {velocities[0]:6.2f}\n"
    text += f"  ω₂: {velocities[1]:6.2f}\n"
    text += f"\n"
    text += f"Box Position:\n"
    text += f"  x: {box_pos[0]:6.3f} m\n"
    text += f"  y: {box_pos[1]:6.3f} m\n"
    text += f"\n"
    text += f"Action:\n"
    text += f"  τ₁: {action[0]:6.2f}\n"
    text += f"  τ₂: {action[1]:6.2f}"

    return text


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Demo environment with YOLO detection")
    parser.add_argument('--box-x', type=float, default=0.10,
                        help='Box X position (default: 0.10)')
    parser.add_argument('--box-y', type=float, default=0.15,
                        help='Box Y position (default: 0.15)')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of steps to simulate (default: 50)')

    args = parser.parse_args()

    print("=" * 70)
    print("Reacher Environment Demo")
    print("=" * 70)
    print(f"Box position: ({args.box_x:.2f}, {args.box_y:.2f})")
    print(f"Steps: {args.steps}")
    print()

    demo_environment(box_x=args.box_x, box_y=args.box_y, num_steps=args.steps)


if __name__ == "__main__":
    main()
