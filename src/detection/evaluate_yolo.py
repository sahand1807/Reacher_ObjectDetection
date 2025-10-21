"""
Evaluate trained YOLOv10 model for red box detection.

Run this script independently after training to:
- Validate model performance
- Generate visualizations and plots
- Benchmark inference speed
- Create comprehensive evaluation report

Usage:
    python src/detection/evaluate_yolo.py
    python src/detection/evaluate_yolo.py --model results/yolo/train/weights/last.pt
"""

import argparse
import time
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


class YOLOEvaluator:
    """Evaluate trained YOLOv10 model."""

    def __init__(self, model_path, config_path="configs/training_config.yaml"):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model (.pt file)
            config_path: Path to training configuration
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = self.config['yolo']['device']

        # Set up output directory
        self.output_dir = self.model_path.parent.parent / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("YOLOv10 Model Evaluation")
        print("=" * 70)
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print()

    def load_model(self):
        """Load trained YOLO model."""
        print(f"📦 Loading model from {self.model_path.name}...")
        self.model = YOLO(str(self.model_path))
        print(f"✓ Model loaded successfully")
        print()

    def validate(self):
        """Run validation on test set."""
        print("=" * 70)
        print("Running Validation")
        print("=" * 70)

        print(f"\n🔍 Validating on dataset...")
        metrics = self.model.val(
            data="data/synthetic/dataset.yaml",
            device=self.device,
            plots=True,
            save_json=True,
            project=str(self.output_dir),
            name="val",
            exist_ok=True,
        )

        # Extract metrics
        self.metrics = {
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': metrics.box.p[0] if len(metrics.box.p) > 0 else 0,
            'recall': metrics.box.r[0] if len(metrics.box.r) > 0 else 0,
        }

        print(f"\n📈 Validation Metrics:")
        print(f"   mAP@0.5:      {self.metrics['mAP50']:.4f}")
        print(f"   mAP@0.5-0.95: {self.metrics['mAP50-95']:.4f}")
        print(f"   Precision:    {self.metrics['precision']:.4f}")
        print(f"   Recall:       {self.metrics['recall']:.4f}")
        print()

        return metrics

    def benchmark_speed(self, num_warmup=10, num_runs=100):
        """Benchmark inference speed."""
        print("=" * 70)
        print("Benchmarking Inference Speed")
        print("=" * 70)

        test_img = "data/synthetic/images/val/reacher_000000.jpg"

        if not Path(test_img).exists():
            # Use first available validation image
            val_images = list(Path("data/synthetic/images/val").glob("*.jpg"))
            if val_images:
                test_img = str(val_images[0])
            else:
                print("⚠️  No validation images found, skipping speed test")
                return None, None

        print(f"\n⚡ Testing inference speed...")
        print(f"   Image: {Path(test_img).name}")
        print(f"   Warmup runs: {num_warmup}")
        print(f"   Timed runs: {num_runs}")

        # Warmup
        for _ in range(num_warmup):
            _ = self.model(test_img, device=self.device, verbose=False)

        # Timed runs
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.model(test_img, device=self.device, verbose=False)
            times.append(time.time() - start)

        avg_time_ms = np.mean(times) * 1000
        std_time_ms = np.std(times) * 1000
        fps = 1000 / avg_time_ms

        print(f"\n📊 Speed Results:")
        print(f"   Average time: {avg_time_ms:.2f} ± {std_time_ms:.2f} ms")
        print(f"   Throughput:   {fps:.1f} FPS")
        print(f"   Min time:     {np.min(times) * 1000:.2f} ms")
        print(f"   Max time:     {np.max(times) * 1000:.2f} ms")
        print()

        self.inference_time_ms = avg_time_ms
        self.fps = fps

        return avg_time_ms, fps

    def visualize_predictions(self, num_samples=9):
        """Create grid of sample predictions."""
        print("=" * 70)
        print("Generating Sample Predictions")
        print("=" * 70)

        # Get sample images
        val_images = list(Path("data/synthetic/images/val").glob("*.jpg"))[:num_samples]

        if not val_images:
            print("⚠️  No validation images found")
            return

        print(f"\n🖼️  Generating predictions for {len(val_images)} images...")

        # Create grid
        rows = int(np.ceil(np.sqrt(num_samples)))
        cols = int(np.ceil(num_samples / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        if num_samples == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, img_path in enumerate(val_images):
            # Run prediction
            results = self.model(str(img_path), device=self.device, verbose=False)

            # Get annotated image
            img = results[0].plot()  # BGR format with boxes

            # Convert BGR to RGB for matplotlib
            img_rgb = img[..., ::-1]

            axes[idx].imshow(img_rgb)
            axes[idx].set_title(f"Sample {idx + 1}", fontsize=10)
            axes[idx].axis('off')

        # Hide empty subplots
        for idx in range(len(val_images), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        output_path = self.output_dir / "sample_predictions.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Sample predictions saved to: {output_path}")
        print()

    def create_summary_report(self):
        """Generate comprehensive evaluation report."""
        print("=" * 70)
        print("Creating Summary Report")
        print("=" * 70)

        report_path = self.output_dir / "evaluation_summary.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("YOLOv10-small Evaluation Report\n")
            f.write("Red Box Detection for Reacher Environment\n")
            f.write("=" * 70 + "\n\n")

            f.write("Model Information:\n")
            f.write(f"  Model path: {self.model_path}\n")
            f.write(f"  Model name: {self.model_path.name}\n")
            f.write(f"  Device: {self.device}\n\n")

            f.write("Dataset:\n")
            f.write("  Validation samples: 1000\n")
            f.write("  Classes: 1 (red_box)\n\n")

            f.write("Performance Metrics:\n")
            f.write(f"  mAP@0.5:      {self.metrics['mAP50']:.4f}\n")
            f.write(f"  mAP@0.5-0.95: {self.metrics['mAP50-95']:.4f}\n")
            f.write(f"  Precision:    {self.metrics['precision']:.4f}\n")
            f.write(f"  Recall:       {self.metrics['recall']:.4f}\n\n")

            if hasattr(self, 'inference_time_ms'):
                f.write("Inference Speed:\n")
                f.write(f"  Average time: {self.inference_time_ms:.2f} ms\n")
                f.write(f"  Throughput:   {self.fps:.1f} FPS\n\n")

            f.write("Generated Files:\n")
            f.write(f"  Evaluation plots: {self.output_dir}/val/\n")
            f.write(f"  Sample predictions: {self.output_dir}/sample_predictions.png\n")
            f.write(f"  This report: {report_path}\n\n")

            f.write("Interpretation:\n")
            if self.metrics['mAP50'] > 0.95:
                f.write("  ✓ Excellent detection accuracy (mAP50 > 0.95)\n")
            elif self.metrics['mAP50'] > 0.85:
                f.write("  ✓ Good detection accuracy (mAP50 > 0.85)\n")
            else:
                f.write("  ⚠ Detection accuracy could be improved\n")

            if self.metrics['mAP50-95'] > 0.85:
                f.write("  ✓ Excellent bounding box precision\n")
            elif self.metrics['mAP50-95'] > 0.70:
                f.write("  ✓ Good bounding box precision\n")
            else:
                f.write("  ⚠ Bounding box precision could be improved\n")

            if hasattr(self, 'fps') and self.fps > 30:
                f.write(f"  ✓ Real-time capable ({self.fps:.1f} FPS)\n")

        print(f"\n✓ Evaluation summary saved to: {report_path}")
        print()

    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        self.load_model()
        self.validate()
        self.benchmark_speed()
        self.visualize_predictions()
        self.create_summary_report()

        print("=" * 70)
        print("✓ Evaluation Complete!")
        print("=" * 70)
        print(f"\nResults saved to: {self.output_dir}")
        print("\nGenerated files:")
        print(f"  - evaluation_summary.txt   (metrics report)")
        print(f"  - sample_predictions.png   (visual predictions)")
        print(f"  - val/                     (validation plots)")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLOv10 model for red box detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="results/yolo/train/weights/best.pt",
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration"
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = YOLOEvaluator(args.model, args.config)
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
