"""Phase 1 curriculum training — lightweight local version.

Uses few envs and DummyVecEnv to keep CPU usage low.
Logs to TensorBoard at logs/baseline/.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from horse_racing.env import HorseRacingSingleEnv

N_ENVS = 4
BATCH_SIZE = 256
LOG_DIR = "logs/baseline"

CURRICULUM = [
    {"track": "tracks/curriculum_1_straight.json", "timesteps": 500_000, "max_steps": 1500, "name": "Stage 1: Straight"},
    {"track": "tracks/curriculum_2_gentle_oval.json", "timesteps": 750_000, "max_steps": 7000, "name": "Stage 2: Gentle oval"},
    {"track": "tracks/curriculum_3_tight_oval.json", "timesteps": 750_000, "max_steps": 4000, "name": "Stage 3: Tight oval"},
    {"track": "tracks/tokyo.json", "timesteps": 1_000_000, "max_steps": 3500, "name": "Stage 4: Tokyo"},
    {"track": "tracks/kokura.json", "timesteps": 1_000_000, "max_steps": 5500, "name": "Stage 5: Kokura"},
    {"track": "tracks/tokyo_2600.json", "timesteps": 1_000_000, "max_steps": 6000, "name": "Stage 6: Tokyo 2600"},
    {"track": "tracks/hanshin.json", "timesteps": 1_000_000, "max_steps": 4000, "name": "Stage 7: Hanshin"},
    {"track": "tracks/kyoto.json", "timesteps": 1_000_000, "max_steps": 4000, "name": "Stage 8: Kyoto"},
]


class ProgressCallback(BaseCallback):
    def __init__(self, stage_name: str, total_timesteps: int, stage_idx: int, num_stages: int, print_freq: int = 2048):
        super().__init__()
        self.stage_name = stage_name
        self.total = total_timesteps
        self.print_freq = print_freq
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.start_time: float | None = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.step_offset = self.num_timesteps

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0 and len(self.model.ep_info_buffer) > 0:
            mean_r = sum(ep["r"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
            elapsed = time.time() - self.start_time
            stage_steps = self.num_timesteps - self.step_offset
            sps = stage_steps / elapsed if elapsed > 0 else 0
            pct = 100 * stage_steps / self.total
            overall = 100 * (self.stage_idx + stage_steps / self.total) / self.num_stages
            eta = (self.total - stage_steps) / sps if sps > 0 else 0
            print(
                f"  [{self.stage_name}] {pct:5.1f}% | overall: {overall:4.1f}% | "
                f"steps: {stage_steps:>8,} | reward: {mean_r:8.2f} | "
                f"{sps:.0f} sps | ETA: {eta/60:.1f}m"
            )
        return True


def make_env(track_path: str, max_steps: int):
    def _init():
        return Monitor(HorseRacingSingleEnv(track_path=track_path, max_steps=max_steps))
    return _init


def main() -> None:
    checkpoint_dir = Path("checkpoints/baseline")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 1 curriculum — {N_ENVS} envs, DummyVecEnv (low CPU)")
    print(f"TensorBoard logs → {LOG_DIR}/")
    print(f"Total timesteps: {sum(s['timesteps'] for s in CURRICULUM):,}\n")

    import torch
    import torch.nn as nn

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

    def export_onnx(sb3_model, output_path: str) -> None:
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

    version = "v3"
    models_dir = Path("models") / version
    models_dir.mkdir(parents=True, exist_ok=True)

    model = None
    total_start = time.time()

    for i, stage in enumerate(CURRICULUM):
        stage_num = i + 1
        print(f"{'=' * 60}")
        print(f"{stage['name']} — {stage['timesteps']:,} timesteps")
        print(f"{'=' * 60}")

        env = SubprocVecEnv([make_env(stage["track"], stage["max_steps"]) for _ in range(N_ENVS)])

        if model is None:
            model = PPO(
                "MlpPolicy", env, verbose=1,
                n_steps=2048, batch_size=BATCH_SIZE, n_epochs=10,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, vf_coef=0.5, ent_coef=0.01,
                device="cpu",
                tensorboard_log=LOG_DIR,
            )
        else:
            model.set_env(env)

        callback = ProgressCallback(stage["name"], stage["timesteps"], i, len(CURRICULUM))
        model.learn(
            total_timesteps=stage["timesteps"], callback=callback,
            reset_num_timesteps=False, tb_log_name="curriculum",
        )

        # Save SB3 checkpoint (for resuming) + ONNX (for version control)
        save_path = checkpoint_dir / f"curriculum_stage_{stage_num}"
        model.save(str(save_path))
        onnx_path = models_dir / f"baseline_stage{stage_num}.onnx"
        export_onnx(model, str(onnx_path))
        print(f"  Saved → {save_path}")
        print(f"  ONNX  → {onnx_path}\n")
        env.close()

    # Final model = last stage
    final_onnx = models_dir / "baseline.onnx"
    export_onnx(model, str(final_onnx))
    print(f"Final ONNX → {final_onnx}")

    total_time = time.time() - total_start
    print(f"Curriculum complete in {total_time / 60:.1f} minutes")


if __name__ == "__main__":
    main()
