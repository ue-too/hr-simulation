#!/usr/bin/env python3
"""Single-stage RL training with eval and early stopping.

Run one curriculum stage at a time, verify results, then continue.

Usage:
    # Train stage 1 from scratch
    python scripts/train_stage.py --stage 1

    # Train stage 2, resuming from stage 1 checkpoint
    python scripts/train_stage.py --stage 2

    # Re-run stage 3 with more timesteps
    python scripts/train_stage.py --stage 3 --timesteps 600000

    # Quick smoke test (50K steps, 1 eval episode)
    python scripts/train_stage.py --stage 1 --timesteps 50000 --eval-episodes 1
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from horse_racing.env import HorseRacingSingleEnv

# ── Stage definitions ────────────────────────────────────────────────
STAGES = [
    {
        "track": "tracks/curriculum_1_straight.json",
        "timesteps": 200_000,
        "max_steps": 1500,
        "gate": 0.90,
        "name": "Stage 1: Straight",
    },
    {
        "track": "tracks/curriculum_2_gentle_oval.json",
        "timesteps": 300_000,
        "max_steps": 7000,
        "gate": 0.80,
        "name": "Stage 2: Gentle oval",
    },
    {
        "track": "tracks/curriculum_3_tight_oval.json",
        "timesteps": 300_000,
        "max_steps": 4000,
        "gate": 0.70,
        "name": "Stage 3: Tight oval",
    },
    {
        "track": "tracks/tokyo.json",
        "timesteps": 400_000,
        "max_steps": 3500,
        "gate": 0.60,
        "name": "Stage 4: Tokyo",
    },
    {
        "track": "tracks/kokura.json",
        "timesteps": 400_000,
        "max_steps": 5500,
        "gate": 0.60,
        "name": "Stage 5: Kokura",
    },
    {
        "track": "tracks/tokyo_2600.json",
        "timesteps": 400_000,
        "max_steps": 6000,
        "gate": 0.50,
        "name": "Stage 6: Tokyo 2600",
    },
    {
        "track": "tracks/hanshin.json",
        "timesteps": 400_000,
        "max_steps": 4000,
        "gate": 0.50,
        "name": "Stage 7: Hanshin",
    },
    {
        "track": "tracks/kyoto.json",
        "timesteps": 400_000,
        "max_steps": 4000,
        "gate": 0.50,
        "name": "Stage 8: Kyoto",
    },
    # ── Skill conditioning stages ──────────────────────────────
    {
        "track": "tracks/tokyo.json",
        "timesteps": 400_000,
        "max_steps": 3500,
        "gate": 0.60,
        "name": "Stage 9: Skills – Tokyo",
        "random_skills": True,
        "min_skills": 1,
        "max_skills": 2,
    },
    {
        "track": "tracks/hanshin.json",
        "timesteps": 400_000,
        "max_steps": 4000,
        "gate": 0.50,
        "name": "Stage 10: Skills – Hanshin",
        "random_skills": True,
        "min_skills": 1,
        "max_skills": 3,
    },
    {
        "track": "tracks/kokura.json",
        "timesteps": 400_000,
        "max_steps": 5500,
        "gate": 0.50,
        "name": "Stage 11: Skills – Kokura",
        "random_skills": True,
        "min_skills": 1,
        "max_skills": 3,
    },
    {
        "track": "tracks/kyoto.json",
        "timesteps": 400_000,
        "max_steps": 4000,
        "gate": 0.50,
        "name": "Stage 12: Skills – Kyoto",
        "random_skills": True,
        "min_skills": 2,
        "max_skills": 3,
    },
]

CHECKPOINT_DIR = Path("checkpoints/baseline")
LOG_DIR = "logs/baseline"


# ── ONNX export ──────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    def __init__(self, sb3_policy):
        super().__init__()
        self.features_extractor = sb3_policy.features_extractor
        self.mlp_extractor = sb3_policy.mlp_extractor
        self.action_net = sb3_policy.action_net

    def forward(self, obs):
        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor(features)
        return self.action_net(latent_pi)


def export_onnx(sb3_model: PPO, output_path: str) -> None:
    wrapper = PolicyNetwork(sb3_model.policy)
    wrapper.eval()
    obs_dim = sb3_model.observation_space.shape[0]
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    torch.onnx.export(
        wrapper, dummy, output_path,
        input_names=["obs"], output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


# ── Callbacks ─────────────────────────────────────────────────────────

class StaminaLoggingCallback(BaseCallback):
    """Track stamina metrics and episode completion for TensorBoard."""

    def __init__(self):
        super().__init__()
        self.final_staminas: deque[float] = deque(maxlen=50)
        self.completions: deque[bool] = deque(maxlen=50)
        self.exhaustion_count: int = 0
        self.episode_count: int = 0

    def _on_step(self) -> bool:
        # Check for completed episodes in the info buffer
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                # Extract terminal observation if available
                terminal_obs = info.get("terminal_observation", None)
                if terminal_obs is not None:
                    # stamina_ratio is at a known index in the obs array
                    # but we can also check via the env
                    pass

        # Log every 10K steps
        if self.n_calls % 10_000 == 0 and self.episode_count > 0:
            if self.final_staminas:
                self.logger.record("stamina/mean_final", np.mean(list(self.final_staminas)))
            if self.completions:
                self.logger.record("stamina/completion_rate", np.mean(list(self.completions)))

        return True


class EarlyStopCallback(BaseCallback):
    """Stop training early if completion gate is met."""

    def __init__(self, gate: float, check_freq: int = 50_000, min_episodes: int = 20):
        super().__init__()
        self.gate = gate
        self.check_freq = check_freq
        self.min_episodes = min_episodes
        self.step_offset: int = 0

    def _on_training_start(self) -> None:
        self.step_offset = self.num_timesteps

    def _on_step(self) -> bool:
        stage_steps = self.num_timesteps - self.step_offset
        if stage_steps > 0 and stage_steps % self.check_freq == 0:
            buf = self.model.ep_info_buffer
            if len(buf) >= self.min_episodes:
                # Check mean episode length vs max_steps as a proxy for completion
                # Completed episodes are shorter than truncated ones
                mean_len = np.mean([ep["l"] for ep in buf])
                mean_r = np.mean([ep["r"] for ep in buf])
                # Log the check
                print(
                    f"  [gate check @ {stage_steps:,}] "
                    f"mean_r={mean_r:.1f} mean_len={mean_len:.0f} "
                    f"(gate={self.gate:.0%})"
                )
        return True


class ProgressCallback(BaseCallback):
    def __init__(self, stage_name: str, total_timesteps: int, print_freq: int = 4096):
        super().__init__()
        self.stage_name = stage_name
        self.total = total_timesteps
        self.print_freq = print_freq
        self.start_time: float | None = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.step_offset = self.num_timesteps

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0 and len(self.model.ep_info_buffer) > 0:
            mean_r = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            elapsed = time.time() - self.start_time
            stage_steps = self.num_timesteps - self.step_offset
            sps = stage_steps / elapsed if elapsed > 0 else 0
            pct = 100 * stage_steps / self.total
            eta = (self.total - stage_steps) / sps if sps > 0 else 0
            print(
                f"  [{self.stage_name}] {pct:5.1f}% | "
                f"steps: {stage_steps:>8,} | reward: {mean_r:8.2f} | "
                f"{sps:.0f} sps | ETA: {eta / 60:.1f}m"
            )
        return True


# ── Eval ──────────────────────────────────────────────────────────────

def run_eval(model: PPO, track_path: str, max_steps: int, episodes: int) -> dict:
    """Run evaluation episodes and return summary stats."""
    completions = []
    rewards = []
    final_staminas = []
    avg_speeds = []

    for ep in range(episodes):
        env = HorseRacingSingleEnv(track_path=track_path, max_steps=max_steps)
        obs, _ = env.reset()
        total_reward = 0.0
        speeds = []

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            o = env.engine.get_observations()[0]
            speeds.append(o["tangential_vel"])

            if terminated or truncated:
                completions.append(o.get("finished", False))
                final_staminas.append(o["stamina_ratio"])
                break

        rewards.append(total_reward)
        avg_speeds.append(np.mean(speeds) if speeds else 0)
        env.close()
        sys.stdout.write("." if completions[-1] else "x")
        sys.stdout.flush()

    print()
    return {
        "completion_rate": np.mean(completions) if completions else 0,
        "mean_reward": np.mean(rewards),
        "mean_final_stamina": np.mean(final_staminas) if final_staminas else 0,
        "mean_avg_speed": np.mean(avg_speeds),
    }


# ── Main ──────────────────────────────────────────────────────────────

def make_env(
    track_path: str,
    max_steps: int,
    random_skills: bool = False,
    min_skills: int = 1,
    max_skills: int = 3,
):
    def _init():
        return Monitor(HorseRacingSingleEnv(
            track_path=track_path,
            max_steps=max_steps,
            random_skills=random_skills,
            min_skills=min_skills,
            max_skills=max_skills,
        ))
    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single curriculum stage")
    parser.add_argument("--stage", type=int, required=True, help="Stage number (1-12)")
    parser.add_argument("--timesteps", type=int, default=None, help="Override timesteps")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint to resume from (default: auto from previous stage)")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Eval episodes after training")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    if args.stage < 1 or args.stage > len(STAGES):
        print(f"Stage must be 1-{len(STAGES)}, got {args.stage}")
        sys.exit(1)

    stage = STAGES[args.stage - 1]
    timesteps = args.timesteps or stage["timesteps"]

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine resume checkpoint
    resume_path = args.resume_from
    if resume_path is None and args.stage > 1:
        prev = CHECKPOINT_DIR / f"stage_{args.stage - 1}.zip"
        if prev.exists():
            resume_path = str(prev)

    print(f"{'=' * 60}")
    print(f"  {stage['name']}")
    print(f"  track: {stage['track']}")
    print(f"  timesteps: {timesteps:,}")
    print(f"  gate: {stage['gate']:.0%} completion")
    print(f"  resume: {resume_path or 'fresh'}")
    print(f"{'=' * 60}")

    # Create environment
    env = SubprocVecEnv([
        make_env(
            stage["track"], stage["max_steps"],
            random_skills=stage.get("random_skills", False),
            min_skills=stage.get("min_skills", 1),
            max_skills=stage.get("max_skills", 3),
        ) for _ in range(args.n_envs)
    ])

    # Create or load model
    if resume_path:
        print(f"  Loading checkpoint: {resume_path}")
        model = PPO.load(resume_path, env=env, device=args.device)
        model.tensorboard_log = LOG_DIR
    else:
        model = PPO(
            "MlpPolicy", env, verbose=0,
            n_steps=2048, batch_size=256, n_epochs=10,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, vf_coef=0.5, ent_coef=0.01,
            device=args.device,
            tensorboard_log=LOG_DIR,
        )

    # Train
    callbacks = [
        ProgressCallback(stage["name"], timesteps),
        EarlyStopCallback(stage["gate"]),
        StaminaLoggingCallback(),
    ]

    start = time.time()
    model.learn(
        total_timesteps=timesteps, callback=callbacks,
        reset_num_timesteps=(resume_path is None),
        tb_log_name=f"stage_{args.stage}",
    )
    elapsed = time.time() - start
    print(f"\n  Training done in {elapsed / 60:.1f} minutes")

    # Save checkpoint + ONNX
    save_path = CHECKPOINT_DIR / f"stage_{args.stage}"
    model.save(str(save_path))
    onnx_path = CHECKPOINT_DIR / f"stage_{args.stage}.onnx"
    export_onnx(model, str(onnx_path))
    print(f"  Checkpoint → {save_path}.zip")
    print(f"  ONNX       → {onnx_path}")

    env.close()

    # Eval
    if args.eval_episodes > 0:
        print(f"\n  Evaluating ({args.eval_episodes} episodes, deterministic)...")
        sys.stdout.write("  ")
        stats = run_eval(model, stage["track"], stage["max_steps"], args.eval_episodes)
        print(f"  Completion: {stats['completion_rate']:.0%}")
        print(f"  Mean reward: {stats['mean_reward']:.1f}")
        print(f"  Final stamina: {stats['mean_final_stamina']:.1%}")
        print(f"  Avg speed: {stats['mean_avg_speed']:.1f} m/s")

        # Gate check
        passed = stats["completion_rate"] >= stage["gate"]
        if passed:
            print(f"\n  GATE PASSED ({stats['completion_rate']:.0%} >= {stage['gate']:.0%})")
            sys.exit(0)
        else:
            print(f"\n  GATE FAILED ({stats['completion_rate']:.0%} < {stage['gate']:.0%})")
            print(f"  Re-run with more timesteps: --timesteps {timesteps * 2}")
            sys.exit(1)


if __name__ == "__main__":
    main()
