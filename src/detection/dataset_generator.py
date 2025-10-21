"""
Synthetic dataset generator for YOLO training.

Generates top-down view images of Reacher environment with random red box positions
and creates YOLO-format annotations.
"""

import os
import sys
import yaml
import numpy as np
import cv2
from tqdm import tqdm
import gymnasium as gym
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environment import ReacherBoxEnv


class SyntheticDatasetGenerator:
    """Generate synthetic dataset for YOLO red box detection."""

    def __init__(self, config_path="configs/training_config.yaml"):
        """
        Initialize dataset generator.

        Args:
            config_path: Path to training configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.dataset_config = self.config['dataset']
        self.env_config = self.config['environment']

        # Create environment
        self.env = gym.make("ReacherObjectDetection-v0", render_mode="rgb_array")

        # Dataset paths
        self.train_images_dir = Path("data/synthetic/images/train")
        self.train_labels_dir = Path("data/synthetic/labels/train")
        self.val_images_dir = Path("data/synthetic/images/val")
        self.val_labels_dir = Path("data/synthetic/labels/val")

        # Create directories
        for dir_path in [self.train_images_dir, self.train_labels_dir,
                         self.val_images_dir, self.val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_red_box_bbox(self, img):
        """
        Detect red box in image using color thresholding.

        Args:
            img: RGB image (H, W, 3)

        Returns:
            bbox: Normalized YOLO format [x_center, y_center, width, height]
                  or None if not detected
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Pure red color range in HSV (more restrictive to avoid pink/magenta)
        # Red wraps around in HSV, so we need two ranges
        # Lower range for bright reds
        lower_red1 = np.array([0, 150, 150])    # Higher saturation and value
        upper_red1 = np.array([10, 255, 255])
        # Upper range for darker reds
        lower_red2 = np.array([170, 150, 150])  # More restrictive hue range
        upper_red2 = np.array([180, 255, 255])

        # Create masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        # Filter contours by size and position
        img_h, img_w = img.shape[:2]
        center_x, center_y = img_w / 2, img_h / 2

        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # Red box should be small (not the large arena boundary)
            # Expected box size: ~20-60 pixels (2cm box at 1.5m height)
            # Arena boundary is much larger (100+ pixels)
            if 100 < area < 5000:  # Filter by area
                # Check if contour is near center (not at edges like arena)
                cx = x + w / 2
                cy = y + h / 2
                dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)

                # Box should be within central region (not at very edges)
                if dist_from_center < img_w * 0.4:  # Within 40% of image radius
                    valid_contours.append(contour)

        if len(valid_contours) == 0:
            return None

        # Get largest valid contour (should be the box)
        largest_contour = max(valid_contours, key=cv2.contourArea)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Convert to YOLO format (normalized coordinates)
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        width = w / img_w
        height = h / img_h

        return np.array([x_center, y_center, width, height])

    def generate_sample(self, idx, output_images_dir, output_labels_dir):
        """
        Generate a single training sample.

        Args:
            idx: Sample index
            output_images_dir: Directory to save image
            output_labels_dir: Directory to save label

        Returns:
            success: True if sample generated successfully
        """
        # Reset environment with random box position
        self.env.reset()

        # Render top-down view
        img = self.env.unwrapped.render_top_view()

        # Detect red box
        bbox = self.get_red_box_bbox(img)

        if bbox is None:
            return False

        # Save image
        img_path = output_images_dir / f"reacher_{idx:06d}.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Save label (YOLO format: class_id x_center y_center width height)
        label_path = output_labels_dir / f"reacher_{idx:06d}.txt"
        with open(label_path, 'w') as f:
            # Class 0 = red_box
            f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        return True

    def generate_dataset(self):
        """Generate full synthetic dataset (train + validation)."""
        train_size = self.dataset_config['train_size']
        val_size = self.dataset_config['val_size']
        seed = self.dataset_config['seed']

        # Set seed for reproducibility
        np.random.seed(seed)

        print("=" * 70)
        print("Generating Synthetic Dataset for YOLO Training")
        print("=" * 70)

        # Generate training set
        print(f"\n📸 Generating {train_size} training samples...")
        train_success = 0
        train_idx = 0

        with tqdm(total=train_size, desc="Training Set") as pbar:
            while train_success < train_size:
                if self.generate_sample(train_idx, self.train_images_dir, self.train_labels_dir):
                    train_success += 1
                    pbar.update(1)
                train_idx += 1

        print(f"✓ Training set: {train_success}/{train_idx} samples (success rate: {train_success/train_idx*100:.1f}%)")

        # Generate validation set
        print(f"\n📸 Generating {val_size} validation samples...")
        val_success = 0
        val_idx = 0

        with tqdm(total=val_size, desc="Validation Set") as pbar:
            while val_success < val_size:
                if self.generate_sample(val_idx, self.val_images_dir, self.val_labels_dir):
                    val_success += 1
                    pbar.update(1)
                val_idx += 1

        print(f"✓ Validation set: {val_success}/{val_idx} samples (success rate: {val_success/val_idx*100:.1f}%)")

        # Create dataset.yaml for YOLO
        self._create_dataset_yaml()

        # Create visualizations
        self._create_sample_visualizations()

        print("\n" + "=" * 70)
        print("✓ Dataset generation complete!")
        print("=" * 70)
        print(f"\nDataset summary:")
        print(f"  Training samples: {train_success}")
        print(f"  Validation samples: {val_success}")
        print(f"  Total samples: {train_success + val_success}")
        print(f"\nFiles saved to:")
        print(f"  Images: data/synthetic/images/")
        print(f"  Labels: data/synthetic/labels/")
        print(f"  Config: data/synthetic/dataset.yaml")

        self.env.close()

    def _create_dataset_yaml(self):
        """Create YOLO dataset configuration file."""
        dataset_yaml = {
            'path': str(Path('data/synthetic').absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'red_box'
            },
            'nc': 1,  # Number of classes
        }

        yaml_path = Path("data/synthetic/dataset.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)

        print(f"\n✓ YOLO dataset config saved to: {yaml_path}")

    def _create_sample_visualizations(self, num_samples=9):
        """Create visualization of sample images with bounding boxes."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()

        # Load random samples
        image_files = list(self.train_images_dir.glob("*.jpg"))[:num_samples]

        for idx, img_file in enumerate(image_files):
            # Load image
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Load label
            label_file = self.train_labels_dir / (img_file.stem + ".txt")
            with open(label_file, 'r') as f:
                label = f.readline().strip().split()
                class_id, x_center, y_center, width, height = map(float, label)

            # Convert normalized coordinates to pixel coordinates
            img_h, img_w = img.shape[:2]
            x_center_px = x_center * img_w
            y_center_px = y_center * img_h
            width_px = width * img_w
            height_px = height * img_h

            # Calculate top-left corner
            x1 = x_center_px - width_px / 2
            y1 = y_center_px - height_px / 2

            # Plot image
            axes[idx].imshow(img)
            axes[idx].set_title(f"Sample {idx + 1}")
            axes[idx].axis('off')

            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), width_px, height_px,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            axes[idx].add_patch(rect)

        plt.tight_layout()
        viz_path = "data/synthetic/sample_visualizations.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"✓ Sample visualizations saved to: {viz_path}")
        plt.close()


def main():
    """Main entry point."""
    generator = SyntheticDatasetGenerator()
    generator.generate_dataset()


if __name__ == "__main__":
    main()
