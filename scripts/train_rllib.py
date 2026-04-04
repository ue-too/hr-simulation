"""Training entry point using Ray RLlib with native multi-agent support.

All horses share a single policy (parameter sharing). Per-episode randomization
of tracks, horse count, genomes, and archetypes creates diverse training.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from horse_racing.rllib_env import HorseRacingRLlibEnv


def find_all_tracks(base: str = ".") -> list[str]:
    """Find all track JSON files."""
    return sorted(glob.glob(os.path.join(base, "tracks", "*.json")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train horse racing RL agents with RLlib")
    parser.add_argument("--tracks", type=str, default=None,
                        help="Comma-separated track paths (default: all tracks/*.json)")
    parser.add_argument("--min-horses", type=int, default=4)
    parser.add_argument("--max-horses", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Parallel rollout workers (0 = single-process)")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--restore", type=str, default=None,
                        help="Path to checkpoint to restore from")
    parser.add_argument("--no-randomize-archetypes", action="store_true")
    parser.add_argument("--no-randomize-genomes", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    # Resolve tracks
    if args.tracks:
        track_paths = args.tracks.split(",")
    else:
        track_paths = find_all_tracks()
    if not track_paths:
        print("No tracks found! Check your tracks/ directory.")
        return

    # Resolve checkpoint dir
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", "multi_agent")
    else:
        args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    # Disable runtime_env so workers use the current Python environment directly.
    ray.init(runtime_env={"working_dir": None})

    env_config = {
        "track_paths": track_paths,
        "min_horse_count": args.min_horses,
        "max_horse_count": args.max_horses,
        "max_steps": args.max_steps,
        "randomize_archetypes": not args.no_randomize_archetypes,
        "randomize_genomes": not args.no_randomize_genomes,
    }

    config = (
        PPOConfig()
        .environment(
            env=HorseRacingRLlibEnv,
            env_config=env_config,
        )
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=1,
        )
        .training(
            train_batch_size=16000,
            minibatch_size=512,
            num_epochs=10,
            lr=3e-4,
            gamma=0.995,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=50.0,
            entropy_coeff=0.005,
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
        )
        .framework("torch")
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
    )

    algo = config.build_algo()

    if args.restore:
        restore_path = str(Path(args.restore).resolve())
        algo.restore(restore_path)
        print(f"Restored from {restore_path}")

    print(f"Tracks: {len(track_paths)} ({', '.join(Path(t).stem for t in track_paths)})")
    print(f"Horses: {args.min_horses}-{args.max_horses}")
    print(f"Archetypes: {'randomized' if not args.no_randomize_archetypes else 'off'}")
    print(f"Workers: {args.num_workers}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print()

    best_reward = float("-inf")
    start_time = time.time()

    for i in range(args.iterations):
        result = algo.train()

        # Extract metrics (Ray 2.40+ nesting under env_runners)
        env_runners = result.get("env_runners", result.get("sampler_results", {}))
        mean_reward = env_runners.get("episode_return_mean",
                       env_runners.get("episode_reward_mean", 0.0))
        max_reward = env_runners.get("episode_return_max",
                      env_runners.get("episode_reward_max", 0.0))
        episode_len = env_runners.get("episode_len_mean", 0.0)
        timesteps = result.get("num_env_steps_sampled_lifetime",
                               env_runners.get("num_env_steps_sampled_lifetime", 0))

        if (i + 1) % args.log_every == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = timesteps / elapsed if elapsed > 0 else 0
            print(
                f"Iter {i + 1:4d} | "
                f"reward: {mean_reward:8.1f} (max: {max_reward:8.1f}) | "
                f"ep_len: {episode_len:6.0f} | "
                f"ts: {timesteps:>10,} | "
                f"{rate:,.0f} ts/s"
            )

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_path = algo.save(args.checkpoint_dir)
            if (i + 1) % args.log_every == 0:
                print(f"  *** New best: {best_reward:.1f}")

        if (i + 1) % args.save_every == 0:
            save_path = algo.save(
                os.path.join(args.checkpoint_dir, f"iter_{i + 1}")
            )
            print(f"  Checkpoint: {save_path}")

    elapsed = time.time() - start_time
    final_path = algo.save(os.path.join(args.checkpoint_dir, "final"))

    print(f"\nTraining complete in {elapsed / 60:.1f} min")
    print(f"Best reward: {best_reward:.1f}")
    print(f"Final checkpoint: {final_path}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
