"""Training entry point using Ray RLlib with native multi-agent support."""

from __future__ import annotations

import argparse
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from horse_racing.rllib_env import HorseRacingRLlibEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train horse racing RL agents with RLlib")
    parser.add_argument("--track", type=str, default="tracks/exp_track_8.json")
    parser.add_argument("--horse-count", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200, help="Training iterations")
    parser.add_argument("--num-workers", type=int, default=2, help="Parallel rollout workers")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/horse_racing_ppo")
    parser.add_argument("--shared-policy", action="store_true", default=True,
                        help="All horses share one policy (default)")
    parser.add_argument("--per-agent-policy", action="store_true",
                        help="Each horse gets its own policy")
    args = parser.parse_args()

    ray.init()

    env_config = {
        "track_path": args.track,
        "horse_count": args.horse_count,
        "max_steps": 5000,
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
            train_batch_size=4000,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
        )
        .framework("torch")
    )

    if args.per_agent_policy:
        # Each horse gets its own policy — allows specialization
        config = config.multi_agent(
            policies={
                f"horse_{i}_policy": PolicySpec() for i in range(args.horse_count)
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"{agent_id}_policy",
        )
    else:
        # Shared policy — all horses train the same network (default, more sample efficient)
        config = config.multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )

    algo = config.build()

    print(f"Training on {args.track} with {args.horse_count} horses")
    print(f"Policy mode: {'per-agent' if args.per_agent_policy else 'shared'}")
    print(f"Workers: {args.num_workers}")
    print()

    best_reward = float("-inf")
    for i in range(args.iterations):
        result = algo.train()

        mean_reward = result["env_runners"]["episode_reward_mean"]
        episode_len = result["env_runners"]["episode_len_mean"]

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"Iter {i + 1:4d} | "
                f"reward: {mean_reward:8.2f} | "
                f"ep_len: {episode_len:7.1f} | "
                f"timesteps: {result['num_env_steps_sampled_lifetime']}"
            )

        if mean_reward > best_reward:
            best_reward = mean_reward
            checkpoint = algo.save(args.checkpoint_dir)
            if (i + 1) % 10 == 0:
                print(f"  New best! Saved to {checkpoint}")

    algo.stop()
    ray.shutdown()
    print(f"\nTraining complete. Best reward: {best_reward:.2f}")


if __name__ == "__main__":
    main()
