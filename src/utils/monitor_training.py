"""
Real-time training progress monitor with visual progress bar.
"""

import re
import time
import sys
from pathlib import Path


def parse_training_line(line):
    """Parse YOLO training output line to extract progress."""
    # Pattern: "1/30  2.26G  2.567  13.91  1.642  13  416: 29% ━━━───────── 179/625"
    pattern = r'(\d+)/(\d+)\s+[\d.]+G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+).*?(\d+)%.*?(\d+)/(\d+)'
    match = re.search(pattern, line)

    if match:
        epoch_current = int(match.group(1))
        epoch_total = int(match.group(2))
        box_loss = float(match.group(3))
        cls_loss = float(match.group(4))
        dfl_loss = float(match.group(5))
        batch_percent = int(match.group(6))
        batch_current = int(match.group(7))
        batch_total = int(match.group(8))

        return {
            'epoch': epoch_current,
            'epoch_total': epoch_total,
            'box_loss': box_loss,
            'cls_loss': cls_loss,
            'dfl_loss': dfl_loss,
            'batch_percent': batch_percent,
            'batch_current': batch_current,
            'batch_total': batch_total
        }
    return None


def create_progress_bar(percent, width=50):
    """Create a visual progress bar."""
    filled = int(width * percent / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"|{bar}| {percent}%"


def monitor_training():
    """Monitor training progress from results directory."""
    print("=" * 80)
    print("YOLO Training Monitor - Press Ctrl+C to exit")
    print("=" * 80)

    last_info = None
    start_time = time.time()

    try:
        while True:
            # Read the training output file or check logs
            # For now, we'll just display a formatted update

            if last_info:
                elapsed = time.time() - start_time
                elapsed_min = int(elapsed / 60)
                elapsed_sec = int(elapsed % 60)

                # Calculate overall progress
                epoch_progress = (last_info['epoch'] - 1) * 100 + last_info['batch_percent']
                overall_progress = epoch_progress / last_info['epoch_total']

                # Clear screen (works on Unix/Mac)
                print("\033[2J\033[H")

                print("=" * 80)
                print(f"YOLO Training Progress - Elapsed: {elapsed_min}m {elapsed_sec}s")
                print("=" * 80)

                print(f"\n📊 Epoch {last_info['epoch']}/{last_info['epoch_total']}")
                print(create_progress_bar(last_info['batch_percent'], 60))
                print(f"Batch: {last_info['batch_current']}/{last_info['batch_total']}")

                print(f"\n📈 Overall Progress")
                print(create_progress_bar(int(overall_progress), 60))

                print(f"\n📉 Current Losses:")
                print(f"  Box Loss: {last_info['box_loss']:.4f}")
                print(f"  Cls Loss: {last_info['cls_loss']:.4f}")
                print(f"  DFL Loss: {last_info['dfl_loss']:.4f}")

                print("\n" + "=" * 80)

            time.sleep(5)  # Update every 5 seconds

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)


if __name__ == "__main__":
    monitor_training()
