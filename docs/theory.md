# Theoretical Background

This document provides the theoretical foundation for the two main components of this project: object detection using YOLOv10 and reinforcement learning using Soft Actor-Critic (SAC).

---

## Table of Contents

1. [Object Detection with YOLO](#object-detection-with-yolo)
2. [Soft Actor-Critic (SAC)](#soft-actor-critic-sac)
3. [Integration: Vision-Based Reinforcement Learning](#integration-vision-based-reinforcement-learning)
4. [References](#references)

---

## Object Detection with YOLO

### Overview

YOLO (You Only Look Once) is a family of real-time object detection algorithms that treat object detection as a regression problem, directly predicting bounding boxes and class probabilities from full images in a single evaluation.

### YOLOv10 Architecture

YOLOv10 introduces several improvements over previous versions:

1. **NMS-Free Training**: Eliminates the need for Non-Maximum Suppression post-processing
2. **Dual Label Assignment**: Combines one-to-many and one-to-one label assignments during training
3. **Efficiency-Accuracy Driven Model Design**: Optimized architecture for better speed-accuracy tradeoff

#### Key Components

**Backbone**: CSPNet (Cross Stage Partial Network)
- Extracts hierarchical features from input images
- Reduces computational redundancy while maintaining accuracy

**Neck**: PAN (Path Aggregation Network)
- Fuses multi-scale features from the backbone
- Enhances feature propagation across different scales

**Head**: Detection head
- Predicts bounding boxes, objectness scores, and class probabilities
- Uses anchor-free detection for YOLOv10

### Loss Function

The YOLOv10 loss function consists of three components:

```
L_total = λ_box * L_box + λ_cls * L_cls + λ_dfl * L_dfl
```

Where:
- **L_box**: Bounding box regression loss (CIoU - Complete IoU)
- **L_cls**: Classification loss (Binary Cross-Entropy)
- **L_dfl**: Distribution Focal Loss for bounding box quality
- λ_box, λ_cls, λ_dfl: Loss weights

#### Complete IoU (CIoU) Loss

CIoU extends traditional IoU by considering:
- Overlap area
- Center point distance  
- Aspect ratio

```
L_CIoU = 1 - IoU + (ρ²(b, b_gt) / c²) + αv

where:
  IoU = Intersection over Union
  ρ²(b, b_gt) = Euclidean distance between predicted and ground truth centers
  c = diagonal length of the smallest enclosing box
  v = consistency of aspect ratio
  α = weight function
```

### Training Strategy

1. **Data Augmentation**:
   - Horizontal flipping
   - Color jittering (HSV)
   - Mosaic augmentation
   - Translation and scaling

2. **Optimization**:
   - Optimizer: AdamW
   - Learning rate: Cosine annealing with warmup
   - Batch size: Adaptive to hardware

3. **Regularization**:
   - Dropout in classification head
   - Weight decay
   - Label smoothing

---

## Soft Actor-Critic (SAC)

### Overview

Soft Actor-Critic is an off-policy, model-free deep reinforcement learning algorithm that combines:
- **Actor-Critic framework**: Separate policy (actor) and value (critic) networks
- **Maximum entropy RL**: Encourages exploration through entropy maximization
- **Off-policy learning**: Learns from past experiences stored in a replay buffer

### Mathematical Framework

#### Maximum Entropy Objective

SAC maximizes the expected return while maintaining high policy entropy:

```
J(π) = Σ E_(s_t,a_t)~ρ_π [r(s_t, a_t) + α H(π(·|s_t))]

where:
  π = policy
  r(s_t, a_t) = reward function
  α = temperature parameter (controls exploration vs exploitation)
  H(π(·|s_t)) = entropy of the policy at state s_t
  ρ_π = state-action marginal distribution under policy π
```

The entropy term encourages exploration and provides robustness:

```
H(π(·|s)) = -E_a~π[log π(a|s)]
```

#### Soft Q-Function

The soft Q-function incorporates entropy into the value estimation:

```
Q^π(s_t, a_t) = E[Σ γ^k (r(s_{t+k}, a_{t+k}) + α H(π(·|s_{t+k})))]

where:
  γ = discount factor (0 < γ < 1)
```

#### Soft Value Function

```
V^π(s_t) = E_a~π [Q^π(s_t, a) - α log π(a|s_t)]
```

### SAC Components

#### 1. Actor (Policy) Network

The actor is a Gaussian policy that outputs a distribution over actions:

```
π_θ(a|s) = N(μ_θ(s), Σ_θ(s))

where:
  μ_θ(s) = mean predicted by neural network with parameters θ
  Σ_θ(s) = diagonal covariance matrix (learned)
```

**Reparameterization Trick**: For continuous actions, SAC uses:

```
a = tanh(μ_θ(s) + Σ_θ(s) ⊙ ε)

where:
  ε ~ N(0, I)
  ⊙ = element-wise multiplication
  tanh = ensures actions are bounded
```

#### 2. Critic Network (Twin Q-Networks)

SAC uses two Q-networks to mitigate overestimation bias:

```
Q_ψ1(s, a) and Q_ψ2(s, a)

Target Q-value:
y(r, s', d) = r + γ(1 - d)[min(Q_ψ'1(s', ã'), Q_ψ'2(s', ã')) - α log π_θ(ã'|s')]

where:
  ã' ~ π_θ(·|s')
  ψ' = target network parameters (slowly updated)
  d = done flag (episode termination)
```

#### 3. Temperature Parameter (α)

SAC automatically tunes α to balance exploration and exploitation:

```
L(α) = E_a~π_θ [-α log π_θ(a|s) - α H_target]

where:
  H_target = target entropy (usually -dim(A))
  dim(A) = dimensionality of action space
```

### Training Algorithm

**Input**: 
- Empty replay buffer D
- Initial policy parameters θ
- Initial Q-function parameters ψ1, ψ2
- Initial target parameters ψ'1 = ψ1, ψ'2 = ψ2

**For** each iteration:

1. **Collect Experience**:
   ```
   a_t ~ π_θ(·|s_t)
   s_{t+1}, r_t ~ env(s_t, a_t)
   Store (s_t, a_t, r_t, s_{t+1}, d_t) in D
   ```

2. **Sample Mini-batch**:
   ```
   B = {(s_i, a_i, r_i, s'_i, d_i)} ~ D
   ```

3. **Update Critics**:
   ```
   L_Q(ψ_j) = E[(Q_ψj(s, a) - y(r, s', d))²]  for j = 1, 2
   
   ψ_j ← ψ_j - λ_Q ∇_ψj L_Q(ψ_j)
   ```

4. **Update Actor**:
   ```
   L_π(θ) = E_s[E_a~π_θ[α log π_θ(a|s) - min_j Q_ψj(s, a)]]
   
   θ ← θ - λ_π ∇_θ L_π(θ)
   ```

5. **Update Temperature**:
   ```
   L(α) = -E_a~π_θ[α log π_θ(a|s) + α H_target]
   
   α ← α - λ_α ∇_α L(α)
   ```

6. **Update Target Networks** (soft update):
   ```
   ψ'_j ← τ ψ_j + (1 - τ) ψ'_j  for j = 1, 2
   
   where τ << 1 (e.g., 0.005)
   ```

### Advantages of SAC

1. **Sample Efficiency**: Off-policy learning from replay buffer
2. **Stability**: Twin Q-networks reduce overestimation
3. **Robustness**: Entropy maximization provides exploration
4. **Automatic Tuning**: Temperature parameter adapts automatically
5. **Continuous Actions**: Handles high-dimensional continuous action spaces

### Hyperparameters

Key hyperparameters in SAC:

- **γ (discount factor)**: 0.99 - weighs future rewards
- **τ (soft update)**: 0.005 - target network update rate  
- **α (temperature)**: Auto-tuned - exploration-exploitation balance
- **Learning rates**: 
  - Actor: 3e-4
  - Critic: 3e-4
  - Temperature: 3e-4
- **Batch size**: 256 - mini-batch size from replay buffer
- **Buffer size**: 500K - replay buffer capacity
- **Update frequency**: Every timestep

---

## Integration: Vision-Based Reinforcement Learning

### Pipeline Architecture

```
Camera Image → YOLO Detector → Box Coordinates → SAC Agent → Joint Torques → MuJoCo Simulation
                                        ↓
                                  Observation Vector
                                [cos θ1, sin θ1, cos θ2, sin θ2, θ̇1, θ̇2, x_box, y_box]
```

### Two-Stage Training

#### Stage 1: Object Detection

**Goal**: Train YOLO to detect red box with high accuracy

**Process**:
1. Generate synthetic dataset (5000 training, 1000 validation)
2. Train YOLOv10-small for 30 epochs
3. Validate coordinate accuracy (must be < 2cm error)

**Output**: Trained YOLO model provides box position (x, y)

#### Stage 2: Reinforcement Learning

**Goal**: Train SAC agent to reach detected box position

**Process**:
1. At episode start: YOLO detects box → extracts (x, y) coordinates
2. Coordinates remain constant throughout episode (matches real-world: single detection per reach)
3. SAC learns to reach (x, y) using joint torques
4. Reward shaping encourages reaching without arm collisions

### State Space Design

The 8-dimensional state vector encodes:

**Joint Configuration** (4D):
- cos(θ1), sin(θ1): Shoulder joint (trigonometric encoding avoids discontinuity)
- cos(θ2), sin(θ2): Elbow joint

**Joint Velocities** (2D):
- θ̇1, θ̇2: Angular velocities

**Target Location** (2D):
- x_box, y_box: YOLO-detected box coordinates (constant per episode)

### Reward Function Design

Multi-component reward function based on 2024 RL literature:

```
R_total = R_distance + R_collision + R_time + R_control + R_success

Components:
1. R_distance = -10 × ||p_fingertip - p_target||
   - Dense signal for learning
   - Scaled 10x to match Gymnasium Reacher-v5

2. R_collision = -10 if arm links touch target, 0 otherwise
   - Encourages using fingertip, not arm
   - Reduced from initial -50 to allow exploration

3. R_time = -0.1
   - Encourages faster reaching
   - Small penalty to not dominate distance reward

4. R_control = -0.01 × ||a||²
   - Encourages smooth, energy-efficient movements
   - Prevents large, jerky actions

5. R_success = 10 + 0.2 × (50 - t) if ||p_fingertip - p_target|| < 0.02m
   - Sparse bonus only at episode end
   - Time bonus rewards faster reaching
   - Triggers early termination on success
```

### Collision Detection

MuJoCo collision groups prevent invalid behavior:

```xml
Collision Groups:
- Group 1 (arm links): contype="1" conaffinity="4"
- Group 2 (fingertip): contype="2" conaffinity="4"  
- Group 4 (target): contype="4" conaffinity="3"

Result:
- Fingertip CAN touch target (2 & 4 → 3 & 4)
- Arm links CAN touch target (1 & 4 → 3 & 4)
- Contacts detected and penalized in reward
```

### Training Optimizations

1. **Parallel Environments**: 4 vectorized environments for 4× sample efficiency
2. **Initial Collision Check**: Prevents invalid spawn configurations (up to 100 respawns)
3. **MPS Acceleration**: Uses Apple M2 GPU for faster training
4. **Float32 Precision**: Required for MPS compatibility
5. **Periodic Evaluation**: Every 5K steps, saves best model

---

## References

### YOLO and Object Detection

1. Wang, C., et al. (2024). "YOLOv10: Real-Time End-to-End Object Detection"
2. Bochkovskiy, A., Wang, C., & Liao, H. (2020). "YOLOv4: Optimal Speed and Accuracy of Object Detection"
3. Redmon, J., & Farhadi, A. (2018). "YOLOv3: An Incremental Improvement"

### Soft Actor-Critic

1. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
2. Haarnoja, T., et al. (2018). "Soft Actor-Critic Algorithms and Applications"
3. Christodoulou, P. (2019). "Soft Actor-Critic for Discrete Action Settings"

### Reinforcement Learning

1. Sutton, R., & Barto, A. (2018). "Reinforcement Learning: An Introduction" (2nd ed.)
2. Lillicrap, T., et al. (2015). "Continuous Control with Deep Reinforcement Learning" (DDPG)
3. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms"

### Reward Shaping

1. Ng, A., Harada, D., & Russell, S. (1999). "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping"
2. Grzes, M., & Kudenko, D. (2010). "Plan-Based Reward Shaping for Reinforcement Learning"

### MuJoCo and Robot Simulation

1. Todorov, E., Erez, T., & Tassa, Y. (2012). "MuJoCo: A Physics Engine for Model-Based Control"
2. Brockman, G., et al. (2016). "OpenAI Gym" (now Gymnasium)
