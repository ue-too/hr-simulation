"""Training entry point — PPO with SB3 + PettingZoo."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from supersuit import concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1

from horse_racing.multi_agent_env import HorseRacingEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train horse racing RL agents")
    parser.add_argument(
        "--track",
        type=str,
        default="tracks/exp_track_8.json",
        help="Path to track JSON file",
    )
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel envs")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-path", type=str, default="horse_racing_ppo")
    args = parser.parse_args()

    print(f"Setting up training with track: {args.track}")

    env = HorseRacingEnv(track_path=args.track)
    vec_env = pettingzoo_env_to_vec_env_v1(env)
    vec_env = concat_vec_envs_v1(vec_env, num_vec_envs=args.n_envs, base_class="stable_baselines3")

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
    )

    print(f"Starting training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps)
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
