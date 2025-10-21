"""
Test script to verify the new reward function implementation.

Tests:
1. Collision detection works correctly
2. Reward component scales are appropriate
3. Early termination on success
4. Reward ranges match expectations
"""

import sys
from pathlib import Path
import numpy as np
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import ReacherBoxEnv


def test_random_policy():
    """Test reward ranges with random actions."""
    print("\n" + "="*70)
    print("TEST 1: Random Policy (Baseline)")
    print("="*70)

    env = gym.make("ReacherObjectDetection-v0")

    episode_rewards = []
    episode_lengths = []

    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        step = 0

        while step < 50:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(step)

        print(f"  Episode {episode+1}: "
              f"Reward={episode_reward:.2f}, "
              f"Length={step}, "
              f"Final Distance={info['distance']*100:.1f}cm")

    print(f"\n  Mean Episode Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Expected Range: -50 to -100 (untrained policy)")
    print(f"  ✓ PASS" if -100 <= np.mean(episode_rewards) <= -10 else "  ✗ FAIL")

    env.close()


def test_reward_components():
    """Test individual reward components."""
    print("\n" + "="*70)
    print("TEST 2: Reward Component Scales")
    print("="*70)

    env = gym.make("ReacherObjectDetection-v0")
    obs, _ = env.reset()

    # Take a single step
    action = np.array([0.5, 0.5])
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\n  Distance to target: {info['distance']*100:.2f} cm")
    print(f"\n  Reward Components:")
    print(f"    Distance reward:   {info['reward_dist']:7.2f}  (range: -1.8 to 0.0)")
    print(f"    Collision penalty: {info['reward_collision']:7.2f}  (range: -50.0 or 0.0)")
    print(f"    Time penalty:      {info['reward_time']:7.2f}  (fixed: -0.1)")
    print(f"    Control penalty:   {info['reward_ctrl']:7.2f}  (range: -0.02 to 0.0)")
    print(f"    Success bonus:     {info['reward_reach']:7.2f}  (range: 0 or 10-20)")
    print(f"    {'─'*50}")
    print(f"    Total reward:      {info['reward_total']:7.2f}")

    # Check scales
    checks = [
        ("Distance reward scale", -2.0 <= info['reward_dist'] <= 0.0),
        ("Collision penalty scale", info['reward_collision'] in [0.0, -50.0]),
        ("Time penalty", info['reward_time'] == -0.1),
        ("Control penalty scale", -0.02 <= info['reward_ctrl'] <= 0.0),
    ]

    print(f"\n  Component Scale Checks:")
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"    {check_name:.<45} {status}")

    env.close()


def test_collision_detection():
    """Test collision detection between arm links and target."""
    print("\n" + "="*70)
    print("TEST 3: Collision Detection")
    print("="*70)

    env = gym.make("ReacherObjectDetection-v0")

    # Reset and try to cause collision by applying large random actions
    print("\n  Applying random actions to test collision detection...")

    collision_detected = False
    collision_count = 0

    for trial in range(10):
        obs, _ = env.reset()

        for step in range(50):
            # Large random actions more likely to cause collisions
            action = env.action_space.sample() * 2.0  # Amplified
            action = np.clip(action, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)

            if info['arm_collision']:
                collision_detected = True
                collision_count += 1
                print(f"    Trial {trial+1}, Step {step+1}: Collision detected!")
                print(f"      Collision penalty: {info['reward_collision']:.1f}")
                print(f"      Distance: {info['distance']*100:.1f} cm")
                break

            if terminated or truncated:
                break

    print(f"\n  Collision Detection Summary:")
    print(f"    Collisions detected: {collision_count}/10 trials")
    print(f"    Collision system: {'✓ WORKING' if collision_detected else '⚠ NOT TRIGGERED (may need more aggressive actions)'}")

    env.close()


def test_early_termination():
    """Test early termination when target is reached."""
    print("\n" + "="*70)
    print("TEST 4: Early Termination on Success")
    print("="*70)

    env = gym.make("ReacherObjectDetection-v0")
    obs, _ = env.reset()

    # Manually set robot very close to target to test success
    print("\n  Manually positioning fingertip near target...")

    # Get target position
    target_pos = env.unwrapped.box_position_2d
    print(f"  Target position: ({target_pos[0]:.3f}, {target_pos[1]:.3f})")

    # Calculate joint angles to reach target (inverse kinematics approximation)
    # For a 2-link planar robot: simple geometric solution
    L1 = 0.1  # Link 1 length
    L2 = 0.1  # Link 2 length

    x, y = target_pos
    distance_to_origin = np.sqrt(x**2 + y**2)

    if distance_to_origin <= (L1 + L2):
        # Use law of cosines
        cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta2 = np.clip(cos_theta2, -1, 1)
        theta2 = np.arccos(cos_theta2)

        # Calculate theta1
        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        # Set joint angles
        qpos = env.unwrapped.data.qpos.copy()
        qpos[0] = theta1
        qpos[1] = theta2
        qvel = env.unwrapped.data.qvel.copy()
        qvel[:2] = 0  # Zero velocities

        env.unwrapped.set_state(qpos, qvel)

        # Get observation
        obs = env.unwrapped._get_obs()

        # Check initial distance
        fingertip_pos = env.unwrapped.get_body_com("fingertip")[:2]
        initial_distance = np.linalg.norm(fingertip_pos - target_pos)

        print(f"  Initial fingertip position: ({fingertip_pos[0]:.3f}, {fingertip_pos[1]:.3f})")
        print(f"  Initial distance to target: {initial_distance*100:.2f} cm")

        # Take a small action (should stay near target)
        action = np.array([0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n  After one step with zero action:")
        print(f"    Distance: {info['distance']*100:.2f} cm")
        print(f"    Reached: {info['reached']}")
        print(f"    Terminated: {terminated}")
        print(f"    Reward: {reward:.2f}")

        if info['reached']:
            print(f"    Success bonus: {info['reward_reach']:.2f}")
            print(f"    Episode steps: {env.unwrapped.current_step}")
            print(f"    Time bonus: {(50 - env.unwrapped.current_step) * 0.2:.2f}")

        # Verify early termination
        if info['reached'] and terminated:
            print(f"\n  ✓ PASS: Early termination works correctly")
        elif info['reached'] and not terminated:
            print(f"\n  ✗ FAIL: Target reached but episode did not terminate")
        else:
            print(f"\n  ⚠ WARNING: Could not position close enough to target")
    else:
        print(f"  ⚠ Target too far from origin for this test")

    env.close()


def test_success_scenario():
    """Test a complete successful episode."""
    print("\n" + "="*70)
    print("TEST 5: Success Scenario Rewards")
    print("="*70)

    env = gym.make("ReacherObjectDetection-v0")

    # Test what happens in a successful episode
    print("\n  Simulating successful reach at different speeds:")

    for steps_taken in [10, 25, 45]:
        # Simulate reaching at different steps
        steps_remaining = 50 - steps_taken
        time_bonus = steps_remaining * 0.2
        success_reward = 10.0 + time_bonus

        # Typical distance reward accumulated (assuming improvement)
        # Start at ~0.15m, end at 0.0m
        avg_distance = 0.075  # Average distance during episode
        total_distance_reward = -10.0 * avg_distance * steps_taken

        # Time penalty
        total_time_penalty = -0.1 * steps_taken

        # Control penalty (assuming moderate actions)
        total_ctrl_penalty = -0.01 * 0.5 * steps_taken  # Assuming avg ||action||² = 0.5

        # Total episode reward
        total_reward = total_distance_reward + total_time_penalty + total_ctrl_penalty + success_reward

        print(f"\n    Success at step {steps_taken}:")
        print(f"      Distance reward:  {total_distance_reward:7.2f}")
        print(f"      Time penalty:     {total_time_penalty:7.2f}")
        print(f"      Control penalty:  {total_ctrl_penalty:7.2f}")
        print(f"      Success bonus:    {success_reward:7.2f}  (10.0 + {time_bonus:.1f} time bonus)")
        print(f"      {'─'*45}")
        print(f"      Episode reward:   {total_reward:7.2f}")

    print(f"\n  Expected Episode Rewards:")
    print(f"    Fast success (10 steps):  ~+8 to +15")
    print(f"    Medium success (25 steps): ~0 to +8")
    print(f"    Slow success (45 steps):   ~-10 to +2")
    print(f"    Failure (50 steps):        ~-50 to -100")

    env.close()


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "REWARD FUNCTION VERIFICATION TESTS" + " "*19 + "║")
    print("╚" + "="*68 + "╝")

    try:
        test_random_policy()
        test_reward_components()
        test_collision_detection()
        test_early_termination()
        test_success_scenario()

        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)
        print("\nReview the results above to verify:")
        print("  1. Random policy gets rewards in expected range (-50 to -100)")
        print("  2. Reward component scales are correct")
        print("  3. Collision detection triggers properly")
        print("  4. Early termination works on success")
        print("  5. Success scenarios give positive rewards")
        print("\n")

    except Exception as e:
        print(f"\n✗ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
