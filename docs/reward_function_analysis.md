# Reward Function Analysis and Recommendations

## Executive Summary

The current reward function has **4 critical issues** that may prevent effective learning:
1. **No collision penalty** - arm can pass through target
2. **Progress reward can be gamed** - oscillation exploit
3. **Reaching bonus given repeatedly** - not just on first contact
4. **No time pressure** - agent not incentivized to reach quickly

## Current Implementation Analysis

### Reward Components (from `reacher_box_env.py`)

```python
reward_dist = -distance                              # [-0.2, 0.0]
reward_reach = 10.0 if distance < 0.02 else 0.0     # {0, 10}
reward_ctrl = -0.01 * ||action||^2                   # [-0.02, 0.0]
reward_progress = 0.5 * (prev_distance - distance)  # [-0.1, +0.1]
```

### Issues with Current Design

#### Issue 1: No Collision Detection (CRITICAL)

**MuJoCo XML Setting:**
```xml
<geom contype="0" ...>  <!-- Default: no collision -->
```

**Impact:**
- Arm links can pass through target box without penalty
- Unrealistic behavior that won't transfer to real robot
- Agent may learn to "bash through" rather than carefully reach

**Literature Evidence:**
> "Enormous punishment is imposed on the collision behavior to restrain the potential risk"
> - Scientific Reports, 2025

**Fix Required:**
1. Enable collision detection in MuJoCo XML
2. Add collision penalty term: `-50.0` per collision
3. Or use MuJoCo contact sensor: `self.data.ncon`

#### Issue 2: Progress Reward Vulnerability

**Current:**
```python
reward_progress = 0.5 * (prev_distance - distance)
```

**Problems:**
- Oscillation exploit: Move back and forth to accumulate rewards
- Not potential-based shaping (can change optimal policy)
- Scale conflict with distance reward

**Correct Implementation (Ng et al., 1999):**
```python
# Potential-based reward shaping
Φ(s) = -distance(s)
reward_shaping = γ * Φ(s') - Φ(s)
                = γ * (-distance') - (-distance)
                = γ * (distance - distance')
```

**Your version is missing discount factor γ (0.99):**
```python
# Should be:
reward_progress = 0.99 * (self.prev_distance - distance)
# Not:
reward_progress = 0.5 * (self.prev_distance - distance)
```

**Better approach:** Remove progress reward entirely. The dense distance reward `-distance` already provides strong gradient.

#### Issue 3: Reaching Bonus Accumulation

**Current:**
```python
reward_reach = 10.0 if distance < 0.02 else 0.0  # Every step!
```

**Problem:** Over 50 steps near target = 500.0 total bonus
- Encourages "camping" near target
- Conflicts with "reach as soon as possible" goal

**Fix:** Give bonus only once
```python
# Option A: First-time bonus
if distance < 0.02 and not self.has_reached:
    reward_reach = 10.0
    self.has_reached = True
else:
    reward_reach = 0.0

# Option B: Episode-end bonus (better for speed)
if done and distance < 0.02:
    reward_reach = 10.0
```

#### Issue 4: No Time Pressure

**Requirement:** "reach as soon as possible"

**Current:** No time penalty → agent has no urgency

**Literature:**
> "Dense rewards in enhancing sample efficiency... sparse rewards foster less susceptible policies"
> - Reinforcement Learning Motion Planning Review, 2024

**Options:**

**A. Time Step Penalty (Simple)**
```python
reward_time = -0.1  # Small constant penalty per step
```

**B. Early Termination (Preferred)**
```python
if distance < 0.02:
    terminated = True  # End episode immediately
    reward_reach = 10.0 + (max_steps - current_step) * 0.1  # Bonus for speed
```

**C. Exponential Time Reward (Advanced)**
```python
reward_time = -0.01 * current_step  # Penalty grows with time
```

---

## Recommended Reward Function (Version 2.0)

### Design Philosophy (from 2024 literature)

1. **Dense distance reward** - primary learning signal
2. **Sparse success bonus** - clear objective
3. **Collision penalty** - safety constraint
4. **Time pressure** - efficiency incentive
5. **Control penalty** - smooth motion

### Implementation

```python
def step(self, action):
    # Execute action
    self.do_simulation(action, self.frame_skip)

    # Get positions
    fingertip_pos = self.get_body_com("fingertip")[:2]
    target_pos = self.box_position_2d
    distance = np.linalg.norm(fingertip_pos - target_pos)

    # ===== COMPONENT 1: Dense Distance Reward (Primary Signal) =====
    # Scale by 10x to match Gymnasium Reacher-v5 scale
    reward_dist = -10.0 * distance
    # Typical range: [-1.8, 0.0] for distances [0.18m, 0m]

    # ===== COMPONENT 2: Sparse Success Bonus =====
    # Only given at episode end if successful
    reached = distance < 0.02  # 2cm threshold
    reward_reach = 0.0  # Will be added at episode end

    # ===== COMPONENT 3: Collision Penalty (NEW) =====
    # Check for arm-box collisions
    reward_collision = 0.0
    if self.data.ncon > 0:  # If any contacts exist
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = self.model.geom(contact.geom1).name
            geom2_name = self.model.geom(contact.geom2).name

            # Penalize if arm links touch box (but not fingertip)
            if ('link' in geom1_name or 'link' in geom2_name) and 'target' in (geom1_name + geom2_name):
                reward_collision = -50.0  # Large penalty
                break

    # ===== COMPONENT 4: Time Penalty (NEW) =====
    # Small penalty to encourage speed
    reward_time = -0.1

    # ===== COMPONENT 5: Control Penalty =====
    # Encourage smooth, energy-efficient actions
    reward_ctrl = -0.01 * np.square(action).sum()

    # ===== TOTAL REWARD =====
    reward = reward_dist + reward_collision + reward_time + reward_ctrl

    # Check termination
    terminated = reached  # End episode on success (NEW)

    # Add success bonus at episode end
    if terminated and reached:
        # Bonus scales with remaining time (encourage speed)
        time_bonus = (self.max_steps - self.current_step) * 0.2
        reward_reach = 10.0 + time_bonus
        reward += reward_reach

    # Get observation
    observation = self._get_obs()

    # Info dictionary
    info = {
        "distance": distance,
        "reached": reached,
        "reward_dist": reward_dist,
        "reward_reach": reward_reach,
        "reward_collision": reward_collision,
        "reward_time": reward_time,
        "reward_ctrl": reward_ctrl,
        "reward_total": reward,
    }

    if self.render_mode == "human":
        self.render()

    return observation, reward, terminated, False, info
```

### Reward Component Scales

| Component | Range | Purpose | Weight |
|-----------|-------|---------|--------|
| Distance | [-1.8, 0.0] | Dense guidance | 10x |
| Success | [0, ~15] | Sparse goal | 1x |
| Collision | [-50, 0] | Safety | 50x |
| Time | -0.1 | Efficiency | 0.1x |
| Control | [-0.02, 0] | Smoothness | 0.01x |

**Total typical reward per step:** -2.0 to -0.1 (improving as agent gets closer)
**Episode reward:** -100 to +15 (failure to success)

---

## MuJoCo XML Changes Required

To enable collision detection:

```xml
<!-- In reacher_box.xml -->

<!-- Arm links: Enable collision -->
<geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"
      contype="1" conaffinity="1"/>  <!-- CHANGED: contype=1 -->

<geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"
      contype="1" conaffinity="1"/>  <!-- CHANGED: contype=1 -->

<!-- Fingertip: Enable collision (we want this to touch) -->
<geom contype="2" conaffinity="2" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
      <!-- CHANGED: contype=2, different group -->

<!-- Target box: Enable collision -->
<geom name="target" pos="0 0 0" rgba="1.0 0.0 0.0 1.0" size=".02 .02 .02" type="box"
      contype="3" conaffinity="3"/>  <!-- CHANGED: contype=3 -->
```

**Collision groups:**
- Links: group 1 (can collide with group 3)
- Fingertip: group 2 (can collide with group 3, but we allow it)
- Target: group 3 (can collide with all)

This allows detecting arm-box collision while permitting fingertip-box contact.

---

## Alternative: Simpler "Phase 1" Reward (No Collision Detection)

If you want to start training quickly without XML changes:

```python
# Simplified reward - just fix the main issues
reward_dist = -10.0 * distance           # Scale up
reward_ctrl = -0.01 * np.square(action).sum()
reward_time = -0.1                        # Add time pressure

reward = reward_dist + reward_ctrl + reward_time

# Success bonus only at episode end
if distance < 0.02:
    terminated = True
    reward += 10.0 + (50 - self.current_step) * 0.2

# Remove progress reward entirely (was problematic)
```

**Pros:** Simple, no XML changes, should still learn
**Cons:** Can't detect arm-box collision, may learn unrealistic behavior

---

## Comparison with Literature

### Gymnasium Reacher-v5 (Reference Implementation)

```python
# From Gymnasium source code
reward_dist = -np.linalg.norm(fingertip_pos - target_pos)
reward_ctrl = -np.square(action).sum()
reward = reward_dist + reward_ctrl
```

**Differences:**
- No progress reward
- No reaching bonus
- No time penalty (relies on episode limit)
- Simple and clean

**Success metric:** Solve if mean reward > -3.75 over 100 episodes

### Your Task Differences

Your task is **harder** than Gymnasium Reacher-v5:
1. **YOLO detection noise** - target position has uncertainty
2. **Collision avoidance needed** - must use fingertip, not arm
3. **Speed requirement** - "as soon as possible"

Therefore, your reward needs:
- ✓ Collision penalty (not in Reacher-v5)
- ✓ Success bonus (to overcome YOLO noise)
- ✓ Time pressure (for speed requirement)

---

## Recommendations

### Immediate (Required):
1. **Remove or fix progress reward** - it's exploitable
2. **Scale up distance reward** - currently too weak (-0.2 vs -10.0)
3. **Add early termination** - end episode when reached

### High Priority (Strongly Recommended):
4. **Enable collision detection in XML** - prevent arm bashing
5. **Add collision penalty** - enforce proper reaching
6. **Add time penalty** - encourage speed

### Optional (Nice to Have):
7. **Success bonus only at end** - prevent camping
8. **Time-scaled success bonus** - reward faster reaching
9. **Curriculum learning** - start with close targets, increase difficulty

---

## Testing Plan

After implementing changes:

1. **Sanity check:** Random policy should get reward ~ -50 per episode
2. **Optimal policy:** Optimal should get reward ~ +10 per episode
3. **Learning curve:** Should see improvement within 50k steps
4. **Behavioral check:**
   - Agent uses fingertip, not arm links
   - Agent reaches target consistently
   - Agent reaches quickly (< 30 steps)

---

## References

1. "Reinforcement Learning for Path Planning of Free-Floating Space Robotic Manipulator" - Frontiers in Control Engineering, 2024
2. "A Review on Reinforcement Learning for Motion Planning of Robotic Manipulators" - International Journal of Intelligent Systems, 2024
3. "Deep Reinforcement Learning Trajectory Planning for Robotic Manipulator" - Scientific Reports, 2025
4. "Reinforcement-Learning-Based Path Planning: A Reward Function Strategy" - MDPI Applied Sciences, 2024
5. Ng, A. Y., Harada, D., & Russell, S. (1999). "Policy Invariance Under Reward Shaping." ICML.

---

## Conclusion

The current reward function has significant issues that will likely prevent effective learning:

**Critical:** No collision penalty, progress reward exploitable, reaching bonus accumulates
**Important:** No time pressure, wrong distance scale

**Recommended action:** Implement "Recommended Reward Function (Version 2.0)" above with collision detection enabled.

**Estimated training improvement:**
- Current: May not learn proper behavior (arm bashing)
- With fixes: Should reach 80%+ success rate in 100k-500k steps
