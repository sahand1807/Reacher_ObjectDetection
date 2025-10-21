"""
Evaluate trained SAC agent on Reacher Object Detection task.

This script:
1. Loads a trained SAC model
2. Runs evaluation episodes with YOLO or ground truth
3. Computes comprehensive performance metrics
4. Generates rollout GIFs
5. Creates evaluation report
"""

import sys
from pathlib import Path
import os
import numpy as np
import gymnasium as gym
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import ReacherBoxEnv
from src.environment.wrappers import YOLOReacherWrapper
from src.agent.sac_agent import SACAgent


def evaluate_model(
    model_path: str,
    n_episodes: int = 100,
    use_yolo: bool = True,
    save_gifs: bool = False,
    render: bool = False,
    deterministic: bool = True
):
    """
    Evaluate a trained SAC model.

    Args:
        model_path: Path to trained model (.zip file)
        n_episodes: Number of evaluation episodes
        use_yolo: Use YOLO detection (True) or ground truth (False)
        save_gifs: Save rollout GIFs
        render: Display environment during evaluation
        deterministic: Use deterministic policy (recommended for evaluation)
    """
    print("\n" + "="*70)
    print("SAC AGENT EVALUATION - REACHER OBJECT DETECTION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"YOLO: {'Enabled' if use_yolo else 'Disabled (ground truth)'}")
    print(f"Policy: {'Deterministic' if deterministic else 'Stochastic'}")
    print("="*70 + "\n")

    # Create environment (single, not vectorized)
    env = gym.make("ReacherObjectDetection-v0", render_mode="rgb_array" if save_gifs else None)
    env = YOLOReacherWrapper(env, use_yolo=use_yolo, verbose=False)

    print("✓ Environment created")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}\n")

    # Load trained model
    print(f"Loading model from: {model_path}")
    agent = SACAgent(
        env=env,
        model_path=model_path,
        verbose=0
    )
    print("✓ Model loaded\n")

    # Run evaluation
    print("-"*70)
    print(f"Running {n_episodes} evaluation episodes...")
    print("-"*70 + "\n")

    episode_rewards = []
    episode_lengths = []
    successes = []
    final_distances = []
    collision_counts = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        collisions = 0
        frames = [] if save_gifs and episode < 5 else None  # Save first 5 GIFs only

        while not done:
            # Get action from policy
            action, _ = agent.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Track collisions
            if info.get('arm_collision', False):
                collisions += 1

            # Save frame for GIF
            if frames is not None:
                frame = env.unwrapped.render_top_view()
                frames.append(frame)

            if render:
                env.render()

        # Episode ended
        final_distance = info.get('distance', float('inf'))
        success = final_distance < 0.02  # 2cm threshold

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        successes.append(success)
        final_distances.append(final_distance * 100)  # cm
        collision_counts.append(collisions)

        # Print episode summary
        print(f"  Episode {episode+1:3d}/{n_episodes}: "
              f"Reward={episode_reward:7.2f}, "
              f"Length={episode_length:2d}, "
              f"Distance={final_distance*100:5.2f}cm, "
              f"Collisions={collisions:2d}, "
              f"Success={'✓' if success else '✗'}")

        # Save GIF
        if frames is not None and len(frames) > 0:
            save_gif(frames, f"eval_episode_{episode+1}.gif")

    env.close()

    # Compute statistics
    results = {
        'n_episodes': n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': np.mean(successes) * 100,
        'mean_final_distance': np.mean(final_distances),
        'std_final_distance': np.std(final_distances),
        'mean_collisions': np.mean(collision_counts),
        'total_collisions': np.sum(collision_counts),
    }

    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nEpisodes: {results['n_episodes']}")
    print(f"\nPerformance:")
    print(f"  Success Rate:      {results['success_rate']:6.2f}%")
    print(f"  Mean Reward:       {results['mean_reward']:8.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Episode Length: {results['mean_length']:6.2f} ± {results['std_length']:.2f}")
    print(f"\nAccuracy:")
    print(f"  Mean Final Distance: {results['mean_final_distance']:5.2f} ± {results['std_final_distance']:.2f} cm")
    print(f"  Success Threshold:   2.00 cm")
    print(f"  Gap:                 {results['mean_final_distance'] - 2.0:5.2f} cm")
    print(f"\nSafety:")
    print(f"  Mean Collisions/Episode: {results['mean_collisions']:.2f}")
    print(f"  Total Collisions:        {results['total_collisions']}")
    print(f"  Collision Rate:          {results['total_collisions']/(n_episodes*50)*100:.1f}%")

    # Grade performance
    if results['success_rate'] >= 80:
        grade = "A+ Excellent"
    elif results['success_rate'] >= 60:
        grade = "A  Very Good"
    elif results['success_rate'] >= 40:
        grade = "B+ Good"
    elif results['success_rate'] >= 25:
        grade = "B  Acceptable"
    else:
        grade = "C  Needs Improvement"

    print(f"\nOverall Grade: {grade}")
    print("="*70 + "\n")

    # Save report
    save_evaluation_report(results, model_path, use_yolo, deterministic)

    return results


def save_gif(frames, filename, output_dir="results/sac/rollouts"):
    """Save frames as GIF."""
    from PIL import Image
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,  # 100ms per frame
        loop=0
    )

    print(f"    → Saved GIF: {output_path}")


def save_evaluation_report(results, model_path, use_yolo, deterministic):
    """Save evaluation report to text file."""
    os.makedirs("results/sac", exist_ok=True)
    report_path = f"results/sac/evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SAC AGENT EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"YOLO Detection: {'Enabled' if use_yolo else 'Disabled'}\n")
        f.write(f"Policy: {'Deterministic' if deterministic else 'Stochastic'}\n\n")

        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n\n")

        f.write(f"Episodes Evaluated: {results['n_episodes']}\n\n")

        f.write("Performance:\n")
        f.write(f"  Success Rate:        {results['success_rate']:6.2f}%\n")
        f.write(f"  Mean Reward:         {results['mean_reward']:8.2f} ± {results['std_reward']:.2f}\n")
        f.write(f"  Mean Episode Length: {results['mean_length']:6.2f} ± {results['std_length']:.2f}\n\n")

        f.write("Accuracy:\n")
        f.write(f"  Mean Final Distance: {results['mean_final_distance']:5.2f} ± {results['std_final_distance']:.2f} cm\n")
        f.write(f"  Success Threshold:   2.00 cm\n")
        f.write(f"  Gap to Success:      {results['mean_final_distance'] - 2.0:5.2f} cm\n\n")

        f.write("Safety:\n")
        f.write(f"  Mean Collisions/Ep:  {results['mean_collisions']:.2f}\n")
        f.write(f"  Total Collisions:    {results['total_collisions']}\n\n")

    print(f"✓ Evaluation report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained SAC agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate best model with YOLO (100 episodes)
  python scripts/eval_sac_agent.py --model models/sac/best_model.zip

  # Evaluate with ground truth (no YOLO)
  python scripts/eval_sac_agent.py --model models/sac/best_model.zip --no-yolo

  # Quick test (10 episodes)
  python scripts/eval_sac_agent.py --model models/sac/final_model.zip --episodes 10

  # Save GIFs of first 5 episodes
  python scripts/eval_sac_agent.py --model models/sac/best_model.zip --save-gifs

  # Render during evaluation
  python scripts/eval_sac_agent.py --model models/sac/best_model.zip --render --episodes 10
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default='models/sac/best_model.zip',
        help='Path to trained model'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--no-yolo',
        action='store_true',
        help='Use ground truth instead of YOLO detection'
    )
    parser.add_argument(
        '--save-gifs',
        action='store_true',
        help='Save rollout GIFs (first 5 episodes)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment during evaluation'
    )
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Use stochastic policy (default: deterministic)'
    )

    args = parser.parse_args()

    # Check model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("\nAvailable models:")
        model_dir = "models/sac"
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.zip'):
                    print(f"  - {os.path.join(model_dir, f)}")
        return

    # Run evaluation
    evaluate_model(
        model_path=args.model,
        n_episodes=args.episodes,
        use_yolo=not args.no_yolo,
        save_gifs=args.save_gifs,
        render=args.render,
        deterministic=not args.stochastic
    )


if __name__ == "__main__":
    main()
