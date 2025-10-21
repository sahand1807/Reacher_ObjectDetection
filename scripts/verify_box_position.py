"""Verify where the box actually appears using color detection (same as dataset generation)."""

import sys
from pathlib import Path
import cv2
import numpy as np
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import ReacherBoxEnv

def detect_red_box_bbox(img):
    """Detect red box using color thresholding (EXACT same method as dataset generation)."""
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Red color range (same as dataset_generator.py lines 71-75)
    lower_red1 = np.array([0, 150, 150])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 150])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Filter and get largest
    img_h, img_w = img.shape[:2]
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 5000:
            valid_contours.append(contour)

    if len(valid_contours) == 0:
        return None

    largest = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Convert to normalized YOLO format
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h

    return np.array([x_center, y_center, width, height])

def image_to_world(x_img, y_img):
    """Convert normalized image coords to world coords."""
    x_world = (x_img - 0.5) * 0.4
    y_world = (y_img - 0.5) * 0.4
    return x_world, y_world

print("=" * 70)
print("Verifying Box Position: Command vs Actual Appearance")
print("=" * 70)

env = gym.make("ReacherObjectDetection-v0", render_mode="rgb_array")

for i in range(5):
    env.reset()

    # Where we THINK the box is (from environment state)
    commanded_pos = env.unwrapped.goal.copy()
    qpos = env.unwrapped.data.qpos.flat
    qpos_target = qpos[-2:]
    mujoco_xpos = env.unwrapped.data.xpos[env.unwrapped.model.body("target").id][:2]

    # Where the box ACTUALLY appears (color detection)
    img_rgb = env.unwrapped.render_top_view()
    bbox = detect_red_box_bbox(img_rgb)

    if bbox is not None:
        actual_pos_img = bbox[:2]  # Normalized image coords
        actual_pos_world = np.array(image_to_world(actual_pos_img[0], actual_pos_img[1]))

        error = np.linalg.norm(actual_pos_world - commanded_pos) * 1000

        print(f"\nTest {i+1}:")
        print(f"  Commanded (env.goal):    ({commanded_pos[0]:7.4f}, {commanded_pos[1]:7.4f})")
        print(f"  qpos[-2:]:               ({qpos_target[0]:7.4f}, {qpos_target[1]:7.4f})")
        print(f"  Mujoco xpos:             ({mujoco_xpos[0]:7.4f}, {mujoco_xpos[1]:7.4f})")
        print(f"  Actual visual appearance:({actual_pos_world[0]:7.4f}, {actual_pos_world[1]:7.4f})")
        print(f"  --> Mismatch: {error:.2f} mm")
    else:
        print(f"\nTest {i+1}: Failed to detect box!")

env.close()

print("\n" + "=" * 70)
print("Conclusion")
print("=" * 70)
print("If there's a large mismatch, it means:")
print("  1. env.goal/qpos/xpos report one position")
print("  2. The box visually appears at a different position")
print("  3. YOLO correctly detects the visual position")
print("  4. The coordinate transformation is wrong!")
