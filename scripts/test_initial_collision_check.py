"""
Test script to verify initial collision check works correctly.

Tests that box never spawns in contact with arm links.
"""

import sys
from pathlib import Path
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import ReacherBoxEnv

print("="*70)
print("Testing Initial Collision Check")
print("="*70)

env = gym.make("ReacherObjectDetection-v0")

print(f"\nRunning 100 episode resets...")
print("Checking for initial arm-box collisions...\n")

collision_count = 0

for i in range(100):
    obs, info = env.reset()

    # Check if there are any contacts
    if env.unwrapped.data.ncon > 0:
        for j in range(env.unwrapped.data.ncon):
            contact = env.unwrapped.data.contact[j]
            geom1_name = env.unwrapped.model.geom(contact.geom1).name
            geom2_name = env.unwrapped.model.geom(contact.geom2).name

            # Check for arm link - target collision
            if 'target' in (geom1_name, geom2_name):
                other_geom = geom1_name if geom2_name == 'target' else geom2_name
                if 'link' in other_geom:
                    collision_count += 1
                    print(f"  Reset {i+1}: ⚠️  INITIAL COLLISION DETECTED!")
                    print(f"    Collision between: {geom1_name} and {geom2_name}")
                    break

    if (i + 1) % 10 == 0:
        print(f"  Completed {i+1}/100 resets...")

env.close()

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Episodes tested: 100")
print(f"Initial collisions: {collision_count}")
print(f"Success rate: {(100 - collision_count)/100 * 100:.1f}%")

if collision_count == 0:
    print("\n✅ PASS: No initial collisions detected!")
    print("   Initial collision check is working correctly.")
else:
    print(f"\n⚠️  WARNING: {collision_count} initial collisions detected!")
    print("   Initial collision check may not be working properly.")

print("="*70 + "\n")
