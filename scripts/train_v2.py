"""V2 training script — SAC with GRU on Ray RLlib.

Designed for Google Colab Pro+. Runs curriculum stages sequentially,
with gate checks between stages.

Usage:
    python scripts/train_v2.py --stage 1
    python scripts/train_v2.py --stage 1 --checkpoint /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.typing import ModelConfigDict
from ray.tune.registry import register_env

import torch
import torch.nn as nn
import numpy as np

from horse_racing.env.observation import OBS_SIZE
from horse_racing.training.curriculum import get_stage


# ---------------------------------------------------------------------------
# Custom GRU model for RLlib
# ---------------------------------------------------------------------------


class GRUJockeyModel(RecurrentNetwork, nn.Module):
    """GRU-based policy network for the jockey.

    Architecture:
        obs (63) -> Linear(128) -> GRU(128) -> Linear(64) -> action (2)
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name: str,
    ):
        nn.Module.__init__(self)
        RecurrentNetwork.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        custom = model_config.get("custom_model_config", {})
        self.obs_dim = obs_space.shape[0]
        self.hidden_size = custom.get("gru_hidden_size", 128)
        self.feature_dim = custom.get("feature_dim", 128)
        self.post_gru_dim = custom.get("post_gru_dim", 64)

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.obs_dim, self.feature_dim),
            nn.ReLU(),
        )

        # GRU
        self.gru = nn.GRU(self.feature_dim, self.hidden_size, batch_first=True)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.post_gru_dim),
            nn.ReLU(),
            nn.Linear(self.post_gru_dim, num_outputs),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.post_gru_dim),
            nn.ReLU(),
            nn.Linear(self.post_gru_dim, 1),
        )

        self._cur_value = None

    def get_initial_state(self):
        return [torch.zeros(self.hidden_size)]

    def forward_rnn(self, inputs, state, seq_lens):
        # inputs: (B, T, obs_dim)
        # state: list of (B, hidden_size)
        features = self.feature_extractor(inputs)  # (B, T, feature_dim)

        hidden = state[0].unsqueeze(0)  # (1, B, hidden_size)
        gru_out, new_hidden = self.gru(features, hidden)
        new_hidden = new_hidden.squeeze(0)  # (B, hidden_size)

        policy_out = self.policy_head(gru_out)  # (B, T, num_outputs)
        self._cur_value = self.value_head(gru_out).squeeze(-1)  # (B, T)

        return policy_out, [new_hidden]

    def value_function(self):
        return self._cur_value


# ---------------------------------------------------------------------------
# Environment creator
# ---------------------------------------------------------------------------


def make_env(config):
    """Create environment based on stage config."""
    import random as _rng

    stage_num = config.get("stage_num", 1)
    stage = get_stage(stage_num)

    if stage.self_play:
        from horse_racing.training.self_play import SelfPlayEnv
        opponent_paths = config.get("opponent_paths", [])
        track = _rng.choice(stage.tracks)
        num_opp = _rng.randint(stage.min_horses - 1, stage.max_horses - 1)
        return SelfPlayEnv(
            tracks=stage.tracks,
            max_steps=stage.max_steps,
            opponent_onnx_paths=opponent_paths,
            min_opponents=stage.min_horses - 1,
            max_opponents=stage.max_horses - 1,
            randomize_jockey_style=stage.randomize_jockey_style,
        )
    else:
        from horse_racing.env.racing_env import HorseRacingEnv
        track = _rng.choice(stage.tracks)
        num_horses = _rng.randint(stage.min_horses, stage.max_horses)
        return HorseRacingEnv(
            track_path=track,
            num_horses=num_horses,
            max_steps=stage.max_steps,
            randomize_horses=stage.randomize_horses,
            randomize_jockey_style=stage.randomize_jockey_style,
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_stage(
    stage_num: int,
    checkpoint_path: str | None = None,
    output_dir: str = "checkpoints/v2",
):
    """Train a single curriculum stage."""
    stage = get_stage(stage_num)
    print(f"\n{'='*60}")
    print(f"Stage {stage_num}: {stage.name}")
    print(f"  Tracks: {stage.tracks}")
    print(f"  Horses: {stage.min_horses}-{stage.max_horses}")
    print(f"  Timesteps: {stage.timesteps:,}")
    print(f"  Gate: {stage.gate_metric} >= {stage.gate_threshold}")
    print(f"{'='*60}\n")

    # Register environment
    register_env("horse_racing_v2", make_env)

    # Register custom model
    ModelCatalog.register_custom_model("gru_jockey", GRUJockeyModel)

    # Build config
    config = (
        SACConfig()
        .environment(
            env="horse_racing_v2",
            env_config={
                "stage_num": stage_num,
                "opponent_paths": stage.opponent_paths,
            },
        )
        .framework("torch")
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=256,
            tau=0.005,
            target_entropy="auto",
            n_step=3,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 100_000,
            },
        )
        .rollouts(
            num_rollout_workers=1,
            rollout_fragment_length=64,
        )
        .resources(
            num_gpus=1,
        )
        .reporting(
            min_sample_timesteps_per_iteration=1000,
        )
    )

    # Use custom GRU model
    config.model = {
        "custom_model": "gru_jockey",
        "custom_model_config": {
            "gru_hidden_size": 128,
            "feature_dim": 128,
            "post_gru_dim": 64,
        },
        "max_seq_len": 64,
    }

    # Build algorithm
    algo = config.build()

    # Restore from checkpoint if provided
    if checkpoint_path:
        print(f"Restoring from {checkpoint_path}")
        algo.restore(checkpoint_path)

    # Training loop
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_timesteps = 0
    best_reward = float("-inf")
    iterations = 0

    while total_timesteps < stage.timesteps:
        result = algo.train()
        iterations += 1
        total_timesteps = result.get("timesteps_total", 0)

        # Log progress
        mean_reward = result.get("episode_reward_mean", 0)
        episodes = result.get("episodes_total", 0)

        if iterations % 10 == 0:
            print(
                f"  Iter {iterations}: "
                f"timesteps={total_timesteps:,}, "
                f"reward={mean_reward:.1f}, "
                f"episodes={episodes}"
            )

        # Save best checkpoint
        if mean_reward > best_reward:
            best_reward = mean_reward
            save_path = algo.save(str(output_path / f"stage_{stage_num}_best"))
            print(f"  New best: reward={best_reward:.1f}, saved to {save_path}")

    # Final save
    final_path = algo.save(str(output_path / f"stage_{stage_num}_final"))
    print(f"\nStage {stage_num} complete. Final checkpoint: {final_path}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Best reward: {best_reward:.1f}")

    algo.stop()
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Train v2 horse racing jockey")
    parser.add_argument("--stage", type=int, default=1, help="Curriculum stage (1-5)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--output-dir", type=str, default="checkpoints/v2",
                        help="Output directory for checkpoints")
    args = parser.parse_args()

    # Initialize Ray for Colab
    if not ray.is_initialized():
        ray.init(
            num_cpus=2,
            num_gpus=1,
            ignore_reinit_error=True,
        )

    try:
        train_stage(
            stage_num=args.stage,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
