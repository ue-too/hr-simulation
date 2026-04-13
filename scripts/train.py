"""SB3 PPO training script for horse racing v2."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from horse_racing.env import HorseRacingSingleEnv


def make_env(track_path: str, horse_count: int, max_steps: int):
    def _init():
        return HorseRacingSingleEnv(
            track_path=track_path,
            horse_count=horse_count,
            max_steps=max_steps,
        )
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train horse racing PPO agent")
    parser.add_argument("--track", type=str, required=True, help="Path to track JSON")
    parser.add_argument("--horses", type=int, default=4, help="Number of horses")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max steps per episode")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--save-freq", type=int, default=50_000, help="Save checkpoint every N steps")
    parser.add_argument("--log-dir", type=str, default="logs", help="TensorBoard log directory")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = SubprocVecEnv([
        make_env(args.track, args.horses, args.max_steps)
        for _ in range(args.n_envs)
    ])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(log_dir),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path=str(save_dir),
        name_prefix="ppo_horse_racing_v2",
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_cb,
    )

    final_path = save_dir / "ppo_horse_racing_v2_final"
    model.save(str(final_path))
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
