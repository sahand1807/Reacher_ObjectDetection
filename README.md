# Reacher Object Detection with SAC

Vision-based reinforcement learning system combining YOLOv10 object detection with Soft Actor-Critic (SAC) algorithm to train a 2-link robotic arm (Reacher) to reach a randomly placed red box target.

## Project Overview

### System Architecture
1. **Phase 1: Object Detection**
   - YOLOv10-small trained on synthetic top-down Reacher images
   - Detects red box position at episode start
   - Outputs 2D bounding box center coordinates

2. **Phase 2: RL Training**
   - SAC agent receives hybrid observations: [joint states (dynamic), YOLO detection (constant)]
   - Agent learns to reach box with fingertip (not arm)
   - Custom reward function penalizes arm-target contact

### Key Features
- Custom Mujoco environment (identical to Reacher-v4 + red box)
- Top-down camera for consistent YOLO detection
- Modular, professional codebase
- Comprehensive documentation and performance metrics

## Project Structure

```
Reacher_ObjectDetection/
├── data/
│   ├── synthetic/          # YOLO training data
│   │   ├── images/
│   │   └── labels/
│   └── test/               # YOLO validation data
│       ├── images/
│       └── labels/
├── models/
│   ├── yolo/               # Trained YOLO models
│   └── sac/                # Trained SAC policies
├── src/
│   ├── environment/        # Custom Gym environment
│   │   └── assets/         # Modified Mujoco XML
│   ├── detection/          # YOLO detection module
│   ├── agent/              # SAC training logic
│   └── utils/              # Helper functions
├── docs/                   # Theory and reports
├── tests/                  # Unit tests
├── notebooks/              # Analysis & visualization
└── configs/                # Hyperparameters
```

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Workflow

### Phase 1: YOLO Training & Validation
1. Generate synthetic dataset
2. Train YOLOv10-small
3. Quantify performance (mAP, precision, recall)

### Phase 2: SAC Training
1. Integrate YOLO with custom environment
2. Train SAC agent
3. Evaluate performance

## Authors
Reacher Object Detection Project
