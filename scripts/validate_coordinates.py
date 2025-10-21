"""
Validate YOLO coordinate predictions against ground truth.

This script:
1. Loads trained YOLO model
2. Runs inference on validation images
3. Converts image coordinates to world coordinates
4. Compares with ground truth positions
5. Generates accuracy reports and visualizations

Usage:
    python scripts/validate_coordinates.py
    python scripts/validate_coordinates.py --model results/yolo/train/weights/best.pt
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import cv2
from tqdm import tqdm


class CoordinateValidator:
    """Validate YOLO predictions against ground truth world coordinates."""

    def __init__(self, model_path, val_dir="data/synthetic"):
        """
        Initialize validator.

        Args:
            model_path: Path to trained YOLO model
            val_dir: Directory containing validation data
        """
        self.model_path = Path(model_path)
        self.val_dir = Path(val_dir)

        # Load YOLO model
        print(f"Loading YOLO model from {self.model_path}...")
        self.model = YOLO(str(self.model_path))

        # Get validation images and labels
        self.img_dir = self.val_dir / "images" / "val"
        self.label_dir = self.val_dir / "labels" / "val"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Validation images not found: {self.img_dir}")

        self.image_files = sorted(list(self.img_dir.glob("*.jpg")))
        print(f"Found {len(self.image_files)} validation images")

        # Results storage
        self.results = {
            'predictions': [],      # YOLO predictions in world coords
            'ground_truth': [],     # Ground truth world coords
            'image_coords': [],     # YOLO predictions in image coords
            'errors': [],           # Euclidean errors
            'image_names': []
        }

    def image_to_world_coords(self, x_img, y_img, method='linear'):
        """
        Convert image coordinates to world coordinates.

        Args:
            x_img: Normalized image x-coordinate [0, 1]
            y_img: Normalized image y-coordinate [0, 1]
            method: Conversion method ('linear' or 'camera_projection')

        Returns:
            (x_world, y_world) in meters
        """
        if method == 'linear':
            # Linear mapping from image [0, 1] to world [-0.2, 0.2]
            # Workspace is 0.4m diameter (2 * 0.2m reach radius)
            x_world = (x_img - 0.5) * 0.4
            y_world = (y_img - 0.5) * 0.4
            return x_world, y_world

        elif method == 'camera_projection':
            # TODO: Implement proper camera projection using Mujoco camera matrix
            # For now, use linear mapping
            return self.image_to_world_coords(x_img, y_img, method='linear')

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_ground_truth_world_coords(self, label_file):
        """
        Extract ground truth world coordinates from label file.

        YOLO label format: class x_center y_center width height (normalized)
        We need to convert bbox center to world coordinates.

        Args:
            label_file: Path to YOLO label file

        Returns:
            (x_world, y_world) in meters
        """
        with open(label_file, 'r') as f:
            line = f.readline().strip()

        if not line:
            return None

        # Parse YOLO label: class x y w h (normalized)
        parts = line.split()
        class_id = int(parts[0])
        x_img = float(parts[1])  # Normalized [0, 1]
        y_img = float(parts[2])

        # Convert to world coordinates
        x_world, y_world = self.image_to_world_coords(x_img, y_img)

        return x_world, y_world

    def run_validation(self, num_samples=None):
        """
        Run validation on all images.

        Args:
            num_samples: Number of samples to validate (None = all)
        """
        print("\n" + "=" * 70)
        print("Running Coordinate Validation")
        print("=" * 70)

        images_to_process = self.image_files[:num_samples] if num_samples else self.image_files

        print(f"\nProcessing {len(images_to_process)} images...")

        for img_path in tqdm(images_to_process, desc="Validating"):
            # Get corresponding label file
            label_path = self.label_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                print(f"Warning: Label not found for {img_path.name}")
                continue

            # Get ground truth world coordinates
            gt_world = self.get_ground_truth_world_coords(label_path)
            if gt_world is None:
                continue

            # Run YOLO prediction
            results = self.model(str(img_path), verbose=False)

            if len(results[0].boxes) == 0:
                print(f"Warning: No detection for {img_path.name}")
                continue

            # Get predicted bbox center (normalized coordinates)
            bbox = results[0].boxes.xywhn[0]  # [x, y, w, h] normalized
            x_img_pred = bbox[0].item()
            y_img_pred = bbox[1].item()

            # Convert prediction to world coordinates
            x_world_pred, y_world_pred = self.image_to_world_coords(x_img_pred, y_img_pred)

            # Store results
            self.results['predictions'].append([x_world_pred, y_world_pred])
            self.results['ground_truth'].append(list(gt_world))
            self.results['image_coords'].append([x_img_pred, y_img_pred])
            self.results['image_names'].append(img_path.name)

            # Calculate error
            error = np.linalg.norm(
                np.array([x_world_pred, y_world_pred]) - np.array(gt_world)
            )
            self.results['errors'].append(error)

        # Convert to numpy arrays
        self.results['predictions'] = np.array(self.results['predictions'])
        self.results['ground_truth'] = np.array(self.results['ground_truth'])
        self.results['errors'] = np.array(self.results['errors'])

        print(f"\n✓ Validation complete: {len(self.results['errors'])} samples processed")

    def analyze_results(self):
        """Analyze and print validation results."""
        print("\n" + "=" * 70)
        print("Coordinate Validation Results")
        print("=" * 70)

        errors = self.results['errors']

        print(f"\n📊 Error Statistics (meters):")
        print(f"   Mean error:       {np.mean(errors):.6f} m ({np.mean(errors)*1000:.3f} mm)")
        print(f"   Std deviation:    {np.std(errors):.6f} m ({np.std(errors)*1000:.3f} mm)")
        print(f"   Median error:     {np.median(errors):.6f} m ({np.median(errors)*1000:.3f} mm)")
        print(f"   Min error:        {np.min(errors):.6f} m ({np.min(errors)*1000:.3f} mm)")
        print(f"   Max error:        {np.max(errors):.6f} m ({np.max(errors)*1000:.3f} mm)")
        print(f"   95th percentile:  {np.percentile(errors, 95):.6f} m ({np.percentile(errors, 95)*1000:.3f} mm)")

        # Error thresholds
        print(f"\n📈 Accuracy Analysis:")
        threshold_cm = [0.5, 1.0, 2.0, 5.0]
        for thresh in threshold_cm:
            thresh_m = thresh / 100  # Convert cm to m
            accuracy = np.mean(errors < thresh_m) * 100
            print(f"   Within {thresh} cm: {accuracy:.1f}%")

        # Component-wise analysis
        pred = self.results['predictions']
        gt = self.results['ground_truth']

        x_errors = np.abs(pred[:, 0] - gt[:, 0])
        y_errors = np.abs(pred[:, 1] - gt[:, 1])

        print(f"\n📍 Component-wise Errors:")
        print(f"   X-axis mean: {np.mean(x_errors):.6f} m ({np.mean(x_errors)*1000:.3f} mm)")
        print(f"   Y-axis mean: {np.mean(y_errors):.6f} m ({np.mean(y_errors)*1000:.3f} mm)")

        print(f"\n✓ Coordinate transformation validated on {len(errors)} samples")

    def visualize_results(self, output_dir="results/validation"):
        """Generate visualizations of validation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("Generating Visualizations")
        print("=" * 70)

        pred = self.results['predictions']
        gt = self.results['ground_truth']
        errors = self.results['errors']

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # 1. Scatter plot: Predictions vs Ground Truth
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(gt[:, 0], gt[:, 1], c='green', alpha=0.6, label='Ground Truth', s=50)
        ax1.scatter(pred[:, 0], pred[:, 1], c='red', alpha=0.6, label='YOLO Predictions', s=30, marker='x')

        # Draw lines connecting GT to predictions
        for i in range(min(50, len(gt))):  # Limit to 50 for clarity
            ax1.plot([gt[i, 0], pred[i, 0]], [gt[i, 1], pred[i, 1]], 'b-', alpha=0.2, linewidth=0.5)

        circle = plt.Circle((0, 0), 0.2, color='gray', fill=False, linestyle='--', linewidth=1)
        ax1.add_patch(circle)
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_title('Predictions vs Ground Truth (World Coordinates)')
        ax1.legend()
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)

        # 2. Error histogram
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(errors * 1000, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(errors) * 1000, color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors)*1000:.2f} mm')
        ax2.axvline(np.median(errors) * 1000, color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors)*1000:.2f} mm')
        ax2.set_xlabel('Error (mm)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Cumulative error plot
        ax3 = plt.subplot(2, 3, 3)
        sorted_errors = np.sort(errors) * 1000
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        ax3.plot(sorted_errors, cumulative, linewidth=2)
        ax3.axhline(95, color='red', linestyle='--', alpha=0.5, label='95th percentile')
        ax3.axvline(np.percentile(errors, 95) * 1000, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Error (mm)')
        ax3.set_ylabel('Cumulative Percentage (%)')
        ax3.set_title('Cumulative Error Distribution')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 4. X-axis errors
        ax4 = plt.subplot(2, 3, 4)
        x_errors = (pred[:, 0] - gt[:, 0]) * 1000
        ax4.hist(x_errors, bins=50, edgecolor='black', alpha=0.7, color='blue')
        ax4.axvline(0, color='red', linestyle='--', linewidth=2)
        ax4.axvline(np.mean(x_errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(x_errors):.2f} mm')
        ax4.set_xlabel('X Error (mm)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('X-Axis Error Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Y-axis errors
        ax5 = plt.subplot(2, 3, 5)
        y_errors = (pred[:, 1] - gt[:, 1]) * 1000
        ax5.hist(y_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax5.axvline(0, color='red', linestyle='--', linewidth=2)
        ax5.axvline(np.mean(y_errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(y_errors):.2f} mm')
        ax5.set_xlabel('Y Error (mm)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Y-Axis Error Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Accuracy vs threshold
        ax6 = plt.subplot(2, 3, 6)
        thresholds_mm = np.linspace(0, 20, 100)
        accuracies = [np.mean(errors * 1000 < t) * 100 for t in thresholds_mm]
        ax6.plot(thresholds_mm, accuracies, linewidth=2)
        ax6.axhline(95, color='red', linestyle='--', alpha=0.5, label='95% accuracy')
        ax6.set_xlabel('Error Threshold (mm)')
        ax6.set_ylabel('Accuracy (%)')
        ax6.set_title('Accuracy vs Error Threshold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()

        plt.tight_layout()
        plot_path = output_dir / "coordinate_validation.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Visualization saved to: {plot_path}")

        # Create a few sample visualizations with images
        self._visualize_samples(output_dir, num_samples=6)

    def _visualize_samples(self, output_dir, num_samples=6):
        """Visualize individual sample predictions."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Select samples with varying errors
        sorted_indices = np.argsort(self.results['errors'])
        sample_indices = [
            sorted_indices[0],                          # Best
            sorted_indices[len(sorted_indices)//4],     # 25th percentile
            sorted_indices[len(sorted_indices)//2],     # Median
            sorted_indices[3*len(sorted_indices)//4],   # 75th percentile
            sorted_indices[-2],                         # Second worst
            sorted_indices[-1],                         # Worst
        ]

        for idx, sample_idx in enumerate(sample_indices[:num_samples]):
            img_name = self.results['image_names'][sample_idx]
            img_path = self.img_dir / img_name

            # Load and display image
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get coordinates
            pred = self.results['predictions'][sample_idx]
            gt = self.results['ground_truth'][sample_idx]
            error = self.results['errors'][sample_idx]

            # Run YOLO to get bbox
            results = self.model(str(img_path), verbose=False)
            img_with_bbox = results[0].plot()
            img_with_bbox = cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB)

            axes[idx].imshow(img_with_bbox)
            axes[idx].axis('off')

            title = f"Error: {error*1000:.2f} mm\n"
            title += f"GT: ({gt[0]:.4f}, {gt[1]:.4f})\n"
            title += f"Pred: ({pred[0]:.4f}, {pred[1]:.4f})"
            axes[idx].set_title(title, fontsize=8)

        plt.tight_layout()
        sample_path = output_dir / "sample_predictions.png"
        plt.savefig(sample_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Sample predictions saved to: {sample_path}")

    def save_report(self, output_dir="results/validation"):
        """Save detailed validation report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "validation_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("YOLO Coordinate Validation Report\n")
            f.write("=" * 70 + "\n\n")

            f.write("Model Information:\n")
            f.write(f"  YOLO model: {self.model_path}\n")
            f.write(f"  Validation samples: {len(self.results['errors'])}\n\n")

            errors = self.results['errors']

            f.write("Error Statistics (meters):\n")
            f.write(f"  Mean:       {np.mean(errors):.6f} m ({np.mean(errors)*1000:.3f} mm)\n")
            f.write(f"  Std dev:    {np.std(errors):.6f} m ({np.std(errors)*1000:.3f} mm)\n")
            f.write(f"  Median:     {np.median(errors):.6f} m ({np.median(errors)*1000:.3f} mm)\n")
            f.write(f"  Min:        {np.min(errors):.6f} m ({np.min(errors)*1000:.3f} mm)\n")
            f.write(f"  Max:        {np.max(errors):.6f} m ({np.max(errors)*1000:.3f} mm)\n")
            f.write(f"  95th %%ile: {np.percentile(errors, 95):.6f} m ({np.percentile(errors, 95)*1000:.3f} mm)\n\n")

            f.write("Accuracy by Threshold:\n")
            for thresh in [0.5, 1.0, 2.0, 5.0]:
                thresh_m = thresh / 100
                accuracy = np.mean(errors < thresh_m) * 100
                f.write(f"  Within {thresh:3.1f} cm: {accuracy:5.1f}%\n")

            pred = self.results['predictions']
            gt = self.results['ground_truth']
            x_errors = np.abs(pred[:, 0] - gt[:, 0])
            y_errors = np.abs(pred[:, 1] - gt[:, 1])

            f.write("\nComponent-wise Analysis:\n")
            f.write(f"  X-axis mean error: {np.mean(x_errors):.6f} m ({np.mean(x_errors)*1000:.3f} mm)\n")
            f.write(f"  Y-axis mean error: {np.mean(y_errors):.6f} m ({np.mean(y_errors)*1000:.3f} mm)\n\n")

            f.write("Interpretation:\n")
            if np.mean(errors) < 0.005:  # < 5mm
                f.write("  ✓ Excellent coordinate accuracy (< 5mm mean error)\n")
            elif np.mean(errors) < 0.010:  # < 10mm
                f.write("  ✓ Good coordinate accuracy (< 10mm mean error)\n")
            else:
                f.write("  ⚠ Coordinate accuracy needs improvement\n")

            if np.mean(errors < 0.02) > 0.95:  # 95% within 2cm
                f.write("  ✓ Suitable for SAC training (95% within 2cm)\n")
            else:
                f.write("  ⚠ May need calibration for optimal SAC performance\n")

        print(f"✓ Validation report saved to: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate YOLO coordinate predictions against ground truth"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="results/yolo/train/weights/best.pt",
        help="Path to trained YOLO model"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to validate (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/validation",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Run validation
    validator = CoordinateValidator(args.model)
    validator.run_validation(num_samples=args.num_samples)
    validator.analyze_results()
    validator.visualize_results(output_dir=args.output)
    validator.save_report(output_dir=args.output)

    print("\n" + "=" * 70)
    print("✓ Validation Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {args.output}/")
    print("  - validation_report.txt     (detailed metrics)")
    print("  - coordinate_validation.png (analysis plots)")
    print("  - sample_predictions.png    (sample visualizations)")
    print()


if __name__ == "__main__":
    main()
