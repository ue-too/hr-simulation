"""Single-agent SB3 PPO training against BT opponents.

Usage:
    python scripts/train.py --track tracks/tokyo.json --total-timesteps 1_000_000
    python scripts/train.py --track tracks/tokyo.json --no-bt --total-timesteps 500_000
    python scripts/train.py --restore checkpoints/sb3/best_model.zip --total-timesteps 2_000_000
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env

from horse_racing.engine import EngineConfig
from horse_racing.env import HorseRacingSingleEnv


class LoggingCallback(BaseCallback):
    """Logs episode stats every log_every episodes."""

    def __init__(self, log_every: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self._log_every = log_every
        self._episode_count = 0
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # Check for completed episodes in infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_count += 1
                if self._episode_count % self._log_every == 0:
                    ep = info["episode"]
                    elapsed = time.time() - self._start_time
                    ts = self.num_timesteps
                    rate = ts / elapsed if elapsed > 0 else 0
                    print(
                        f"  ep {self._episode_count:5d} | "
                        f"reward: {ep['r']:8.1f} | "
                        f"len: {ep['l']:6d} | "
                        f"ts: {ts:>10,} | "
                        f"{rate:,.0f} ts/s"
                    )
        return True


def find_all_tracks(base: str = ".") -> list[str]:
    return sorted(glob.glob(os.path.join(base, "tracks", "*.json")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-agent SB3 PPO")
    parser.add_argument("--track", type=str, default=None,
                        help="Track JSON path (default: all tracks)")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel envs")
    parser.add_argument("--horses", type=int, default=4,
                        help="Number of horses per race")
    parser.add_argument("--no-bt", action="store_true",
                        help="Disable BT opponents (zero-action baselines)")
    parser.add_argument("--save-dir", type=str, default="checkpoints/sb3")
    parser.add_argument("--save-every", type=int, default=50_000,
                        help="Save checkpoint every N timesteps")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log every N episodes")
    parser.add_argument("--restore", type=str, default=None,
                        help="Path to SB3 checkpoint to resume from")
    args = parser.parse_args()

    # Resolve tracks
    if args.track:
        track_paths = [args.track]
    else:
        track_paths = find_all_tracks()
        if not track_paths:
            print("No tracks found in tracks/ directory.")
            return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    engine_config = EngineConfig(horse_count=args.horses)
    bt_enabled = not args.no_bt

    # Create vectorized env — each env picks a random track on reset
    def make_env(rank: int):
        def _init():
            # Cycle through tracks based on rank for diversity
            track = track_paths[rank % len(track_paths)]
            return HorseRacingSingleEnv(
                track_path=track,
                config=engine_config,
                bt_opponents=bt_enabled,
            )
        return _init

    vec_env = make_vec_env(
        make_env(0),  # dummy, overridden by env_kwargs
        n_envs=args.n_envs,
        seed=42,
    )
    # Replace with per-rank envs for track diversity
    from stable_baselines3.common.vec_env import SubprocVecEnv
    vec_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])

    # PPO hyperparameters matched to RLlib config
    policy_kwargs = dict(net_arch=[256, 256])

    if args.restore:
        print(f"Restoring from {args.restore}")
        model = PPO.load(
            args.restore,
            env=vec_env,
            device="auto",
        )
        model.learning_rate = 3e-4
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            gamma=0.995,
            gae_lambda=0.95,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            clip_range=0.2,
            ent_coef=0.005,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="auto",
        )

    print(f"Tracks: {len(track_paths)} ({', '.join(Path(t).stem for t in track_paths)})")
    print(f"Horses: {args.horses}")
    print(f"BT opponents: {'on' if bt_enabled else 'off'}")
    print(f"Envs: {args.n_envs}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Save dir: {save_dir}")
    print()

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.save_every // args.n_envs, 1),
        save_path=str(save_dir),
        name_prefix="checkpoint",
    )
    logging_cb = LoggingCallback(log_every=args.log_every)

    start_time = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, logging_cb],
    )
    elapsed = time.time() - start_time

    final_path = save_dir / "final_model"
    model.save(str(final_path))
    print(f"\nTraining complete in {elapsed / 60:.1f} min")
    print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
