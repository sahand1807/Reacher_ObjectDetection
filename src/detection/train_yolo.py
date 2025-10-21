"""
Train YOLOv10-small on synthetic Reacher red box dataset.

Uses Apple M2 GPU (MPS) for accelerated training.
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import callbacks
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class YOLOTrainer:
    """Train YOLOv10 for red box detection."""

    def __init__(self, config_path="configs/training_config.yaml"):
        """
        Initialize YOLO trainer.

        Args:
            config_path: Path to training configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.yolo_config = self.config['yolo']

        # Verify device availability
        self.device = self.yolo_config['device']
        if self.device == "mps":
            if not torch.backends.mps.is_available():
                print("⚠️  MPS not available, falling back to CPU")
                self.device = "cpu"
            else:
                print(f"✓ Using Apple M2 GPU (MPS) for training")

        # Output directory
        self.results_dir = Path("results/yolo")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.epoch_pbar = None
        self.batch_pbar = None
        self.current_epoch = 0

    def _on_train_start(self, trainer):
        """Called when training starts."""
        self.epoch_pbar = tqdm(
            total=trainer.epochs,
            desc="Training Progress",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}]'
        )

    def _on_epoch_start(self, trainer):
        """Called at the start of each epoch."""
        self.current_epoch = trainer.epoch + 1
        if self.batch_pbar is not None:
            self.batch_pbar.close()

        # Create new batch progress bar for this epoch
        self.batch_pbar = tqdm(
            total=len(trainer.train_loader),
            desc=f"  Epoch {self.current_epoch}/{trainer.epochs}",
            position=1,
            leave=False,
            bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} batches'
        )

    def _on_batch_end(self, trainer):
        """Called after each training batch."""
        if self.batch_pbar is not None:
            # Update batch progress
            self.batch_pbar.update(1)

            # Update description with loss info
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                try:
                    loss_str = f"box={float(trainer.loss_items[0]):.3f}"
                    self.batch_pbar.set_postfix_str(loss_str)
                except:
                    pass

    def _on_epoch_end(self, trainer):
        """Called at the end of each epoch."""
        if self.batch_pbar is not None:
            self.batch_pbar.close()
            self.batch_pbar = None

        # Update epoch progress
        if self.epoch_pbar is not None:
            self.epoch_pbar.update(1)

            # Update with metrics if available
            try:
                if hasattr(trainer, 'metrics') and trainer.metrics:
                    map_val = trainer.metrics.get('metrics/mAP50(B)', 0)
                    self.epoch_pbar.set_postfix_str(f"mAP50={map_val:.3f}")
            except:
                pass

    def _on_train_end(self, trainer):
        """Called when training ends."""
        if self.batch_pbar is not None:
            self.batch_pbar.close()
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()
        print()  # Clean spacing

    def train(self):
        """Train YOLOv10-small model."""
        print("=" * 70)
        print("Training YOLOv10-small for Red Box Detection")
        print("=" * 70)

        # Load pretrained YOLOv10-small model
        print(f"\n📦 Loading YOLOv10-small model...")
        model = YOLO('yolov10s.pt')  # This will download if not present

        # Set up progress callbacks
        model.add_callback("on_train_start", lambda trainer: self._on_train_start(trainer))
        model.add_callback("on_train_epoch_start", lambda trainer: self._on_epoch_start(trainer))
        model.add_callback("on_train_batch_end", lambda trainer: self._on_batch_end(trainer))
        model.add_callback("on_train_epoch_end", lambda trainer: self._on_epoch_end(trainer))
        model.add_callback("on_train_end", lambda trainer: self._on_train_end(trainer))

        # Training parameters
        print(f"\n⚙️  Training configuration:")
        print(f"   Model: {self.yolo_config['model']}")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {self.yolo_config['epochs']}")
        print(f"   Batch size: {self.yolo_config['batch']}")
        print(f"   Image size: {self.yolo_config['imgsz']}")
        print(f"   Patience: {self.yolo_config['patience']}")
        print(f"   Workers: {self.yolo_config['workers']}")

        # Start training
        print(f"\n🚀 Starting training...")
        print(f"   Dataset: data/synthetic/dataset.yaml")
        print(f"   Output: {self.results_dir}")
        print()

        results = model.train(
            data="data/synthetic/dataset.yaml",
            epochs=self.yolo_config['epochs'],
            imgsz=self.yolo_config['imgsz'],
            batch=self.yolo_config['batch'],
            device=self.device,
            workers=self.yolo_config['workers'],
            patience=self.yolo_config['patience'],
            cache=self.yolo_config.get('cache', False),
            project=str(self.results_dir),
            name="train",
            exist_ok=True,
            verbose=False,  # Suppress default output, use our progress bars
            plots=True,
            # Data augmentation
            hsv_h=self.yolo_config['augment']['hsv_h'],
            hsv_s=self.yolo_config['augment']['hsv_s'],
            hsv_v=self.yolo_config['augment']['hsv_v'],
            degrees=self.yolo_config['augment']['degrees'],
            translate=self.yolo_config['augment']['translate'],
            scale=self.yolo_config['augment']['scale'],
            flipud=self.yolo_config['augment']['flipud'],
            fliplr=self.yolo_config['augment']['fliplr'],
            mosaic=self.yolo_config['augment']['mosaic'],
        )

        print("\n" + "=" * 70)
        print("✓ Training complete!")
        print("=" * 70)

        return results

    def evaluate(self):
        """Evaluate trained model on validation set."""
        print("\n" + "=" * 70)
        print("Evaluating Model Performance")
        print("=" * 70)

        # Load best model
        best_model_path = self.results_dir / "train" / "weights" / "best.pt"
        if not best_model_path.exists():
            print("❌ Best model not found!")
            return

        print(f"\n📊 Loading best model: {best_model_path}")
        model = YOLO(str(best_model_path))

        # Validate
        print(f"\n🔍 Running validation...")
        metrics = model.val(
            data="data/synthetic/dataset.yaml",
            device=self.device,
            plots=True,
            save_json=True,
        )

        # Print metrics
        print(f"\n📈 Performance Metrics:")
        print(f"   mAP@0.5: {metrics.box.map50:.4f}")
        print(f"   mAP@0.5-0.95: {metrics.box.map:.4f}")
        print(f"   Precision: {metrics.box.p[0]:.4f}")
        print(f"   Recall: {metrics.box.r[0]:.4f}")

        # Test inference speed
        print(f"\n⚡ Testing inference speed...")
        test_img = "data/synthetic/images/val/reacher_000000.jpg"

        # Warmup
        for _ in range(10):
            _ = model(test_img, device=self.device, verbose=False)

        # Timed runs
        import time
        times = []
        for _ in range(100):
            start = time.time()
            _ = model(test_img, device=self.device, verbose=False)
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000  # Convert to ms
        fps = 1000 / avg_time

        print(f"   Average inference time: {avg_time:.2f} ms")
        print(f"   Throughput: {fps:.1f} FPS")

        # Create summary report
        self._create_summary_report(metrics, avg_time, fps)

        return metrics

    def _create_summary_report(self, metrics, inference_time_ms, fps):
        """Create a summary report of training results."""
        report_path = self.results_dir / "training_summary.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("YOLOv10-small Training Summary\n")
            f.write("Red Box Detection for Reacher Environment\n")
            f.write("=" * 70 + "\n\n")

            f.write("Dataset:\n")
            f.write("  Training samples: 5000\n")
            f.write("  Validation samples: 1000\n")
            f.write("  Classes: 1 (red_box)\n\n")

            f.write("Model Configuration:\n")
            f.write(f"  Architecture: {self.yolo_config['model']}\n")
            f.write(f"  Device: {self.device}\n")
            f.write(f"  Image size: {self.yolo_config['imgsz']}\n")
            f.write(f"  Batch size: {self.yolo_config['batch']}\n")
            f.write(f"  Epochs: {self.yolo_config['epochs']}\n\n")

            f.write("Performance Metrics:\n")
            f.write(f"  mAP@0.5: {metrics.box.map50:.4f}\n")
            f.write(f"  mAP@0.5-0.95: {metrics.box.map:.4f}\n")
            f.write(f"  Precision: {metrics.box.p[0]:.4f}\n")
            f.write(f"  Recall: {metrics.box.r[0]:.4f}\n\n")

            f.write("Inference Speed:\n")
            f.write(f"  Average time: {inference_time_ms:.2f} ms\n")
            f.write(f"  Throughput: {fps:.1f} FPS\n\n")

            f.write("Files:\n")
            f.write(f"  Best model: results/yolo/train/weights/best.pt\n")
            f.write(f"  Last model: results/yolo/train/weights/last.pt\n")
            f.write(f"  Training curves: results/yolo/train/results.png\n")
            f.write(f"  Confusion matrix: results/yolo/train/confusion_matrix.png\n")

        print(f"\n✓ Training summary saved to: {report_path}")

    def test_predictions(self, num_samples=9):
        """Test model on sample images and visualize predictions."""
        print(f"\n🖼️  Generating prediction samples...")

        # Load best model
        best_model_path = self.results_dir / "train" / "weights" / "best.pt"
        model = YOLO(str(best_model_path))

        # Get sample images
        val_images = list(Path("data/synthetic/images/val").glob("*.jpg"))[:num_samples]

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()

        for idx, img_path in enumerate(val_images):
            # Run prediction
            results = model(str(img_path), device=self.device, verbose=False)

            # Plot results
            img = results[0].plot()  # Annotated image
            axes[idx].imshow(img[..., ::-1])  # BGR to RGB
            axes[idx].set_title(f"Sample {idx + 1}")
            axes[idx].axis('off')

        plt.tight_layout()
        pred_path = self.results_dir / "sample_predictions.png"
        plt.savefig(pred_path, dpi=150, bbox_inches='tight')
        print(f"✓ Sample predictions saved to: {pred_path}")
        plt.close()


def main():
    """Main entry point."""
    trainer = YOLOTrainer()

    # Train model
    results = trainer.train()

    # Evaluate model
    metrics = trainer.evaluate()

    # Generate prediction samples
    trainer.test_predictions()

    print("\n" + "=" * 70)
    print("✓ YOLO Training Pipeline Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
