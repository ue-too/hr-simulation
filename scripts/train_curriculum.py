"""Curriculum training — progressively harder tracks with transfer learning."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from horse_racing.env import HorseRacingSingleEnv


CURRICULUM = [
    {"track": "tracks/curriculum_1_straight.json", "timesteps": 500_000, "name": "Stage 1: Straight"},
    {"track": "tracks/curriculum_2_gentle_oval.json", "timesteps": 1_000_000, "name": "Stage 2: Gentle oval"},
    {"track": "tracks/curriculum_3_tight_oval.json", "timesteps": 1_000_000, "name": "Stage 3: Tight oval"},
    {"track": "tracks/exp_track_8.json", "timesteps": 2_000_000, "name": "Stage 4: Complex track"},
]


class ProgressCallback(BaseCallback):
    def __init__(self, stage_name: str, print_freq=10000):
        super().__init__()
        self.stage_name = stage_name
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = sum(ep["r"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
                mean_len = sum(ep["l"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
                print(
                    f"  [{self.stage_name}] "
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
    parser = argparse.ArgumentParser(description="Curriculum training for horse racing")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--start-stage", type=int, default=1, help="Start from this stage (1-indexed)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from this model path")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = None

    for i, stage in enumerate(CURRICULUM):
        stage_num = i + 1
        if stage_num < args.start_stage:
            continue

        print(f"\n{'='*60}")
        print(f"{stage['name']}")
        print(f"Track: {stage['track']}")
        print(f"Timesteps: {stage['timesteps']:,}")
        print(f"{'='*60}\n")

        env = DummyVecEnv([make_env(stage["track"], args.max_steps) for _ in range(args.n_envs)])

        if model is None and args.resume:
            print(f"Resuming from {args.resume}")
            model = PPO.load(args.resume, env=env)
        elif model is None:
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
        else:
            # Transfer learning: keep weights, swap environment
            model.set_env(env)

        callback = ProgressCallback(stage["name"])
        model.learn(total_timesteps=stage["timesteps"], callback=callback, reset_num_timesteps=False)

        # Save checkpoint for this stage
        save_path = checkpoint_dir / f"curriculum_stage_{stage_num}"
        model.save(str(save_path))
        print(f"\nSaved {stage['name']} to {save_path}")

        env.close()

    # Export final ONNX
    print(f"\n{'='*60}")
    print("Curriculum complete!")
    print(f"Final model: {checkpoint_dir / 'curriculum_stage_4'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
