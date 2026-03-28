"""Simple SB3 PPO training — single process, no Ray worker issues."""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from horse_racing.env import HorseRacingSingleEnv


class ProgressCallback(BaseCallback):
    def __init__(self, print_freq=10000):
        super().__init__()
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = sum(ep["r"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
                mean_len = sum(ep["l"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
                print(
                    f"Steps: {self.num_timesteps:>8d} | "
                    f"reward: {mean_reward:8.2f} | "
                    f"ep_len: {mean_len:7.0f}"
                )
        return True


def make_env(track_path: str, max_steps: int):
    def _init():
        return HorseRacingSingleEnv(track_path=track_path, max_steps=max_steps)
    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train horse racing with SB3 PPO")
    parser.add_argument("--track", type=str, default="tracks/curriculum_1_straight.json")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--save-path", type=str, default="checkpoints/straight_sb3")
    parser.add_argument("--max-steps", type=int, default=3000, help="Max steps per episode")
    args = parser.parse_args()

    print(f"Training on {args.track}")
    print(f"Envs: {args.n_envs}, Total timesteps: {args.total_timesteps}")
    print()

    # Use DummyVecEnv (single process) to avoid pickling issues
    env = DummyVecEnv([make_env(args.track, args.max_steps) for _ in range(args.n_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        device="cpu",
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=ProgressCallback(print_freq=10000),
    )

    model.save(args.save_path)
    print(f"\nModel saved to {args.save_path}")
    env.close()


if __name__ == "__main__":
    main()
