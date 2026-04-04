"""Local smoke test for the full training pipeline.

Runs a minimal training loop to verify everything works before Colab.
"""
import sys
import os
import time

# --- Step 1: Verify env works ---
print("=" * 60)
print("Step 1: Environment smoke test")
print("=" * 60)

import numpy as np
from horse_racing.env.racing_env import HorseRacingEnv

env = HorseRacingEnv(
    track_path="tracks/curriculum_1_straight.json",
    num_horses=1,
    max_steps=100,
)
obs, info = env.reset(seed=42)
print(f"  Obs shape: {obs.shape}, dtype: {obs.dtype}")
print(f"  Action space: {env.action_space}")

for i in range(20):
    obs, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0]))
    if terminated:
        break
print(f"  After {i+1} steps: progress={info['progress']:.3f}, stamina={info['stamina_ratio']:.3f}")
print("  PASSED\n")

# --- Step 2: Ray + RLlib setup ---
print("=" * 60)
print("Step 2: Ray + RLlib setup")
print("=" * 60)

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.typing import ModelConfigDict
from ray.tune.registry import register_env
import torch
import torch.nn as nn

from horse_racing.env.observation import OBS_SIZE
from horse_racing.training.curriculum import get_stage

# Custom GRU model
class GRUJockeyModel(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config: ModelConfigDict, name: str, **kwargs):
        nn.Module.__init__(self)
        RecurrentNetwork.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        custom = model_config.get("custom_model_config", {})
        self.obs_dim = obs_space.shape[0]
        self.hidden_size = custom.get("gru_hidden_size", 128)
        self.feature_dim = custom.get("feature_dim", 128)
        self.post_gru_dim = custom.get("post_gru_dim", 64)

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.obs_dim, self.feature_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(self.feature_dim, self.hidden_size, batch_first=True)
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.post_gru_dim),
            nn.ReLU(),
            nn.Linear(self.post_gru_dim, num_outputs),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.post_gru_dim),
            nn.ReLU(),
            nn.Linear(self.post_gru_dim, 1),
        )
        self._cur_value = None

    def get_initial_state(self):
        return [torch.zeros(self.hidden_size)]

    def forward_rnn(self, inputs, state, seq_lens):
        features = self.feature_extractor(inputs)
        hidden = state[0].unsqueeze(0)
        gru_out, new_hidden = self.gru(features, hidden)
        new_hidden = new_hidden.squeeze(0)
        policy_out = self.policy_head(gru_out)
        self._cur_value = self.value_head(gru_out).squeeze(-1)
        return policy_out, [new_hidden]

    def value_function(self):
        return self._cur_value.reshape(-1)

ModelCatalog.register_custom_model("gru_jockey", GRUJockeyModel)
print("  GRU model registered")

# Environment factory
import random as _rng_mod

def make_env(config):
    stage_num = config.get("stage_num", 1)
    stage = get_stage(stage_num)
    base_path = config.get("track_base_path", ".")

    tracks = [f"{base_path}/{t}" for t in stage.tracks]

    if stage.self_play:
        from horse_racing.training.self_play import SelfPlayEnv
        opponent_paths = config.get("opponent_paths", [])
        return SelfPlayEnv(
            tracks=tracks,
            max_steps=stage.max_steps,
            opponent_onnx_paths=opponent_paths,
            min_opponents=stage.min_horses - 1,
            max_opponents=stage.max_horses - 1,
            randomize_jockey_style=stage.randomize_jockey_style,
        )
    else:
        from horse_racing.env.racing_env import HorseRacingEnv
        track = _rng_mod.choice(tracks)
        num_horses = _rng_mod.randint(stage.min_horses, stage.max_horses)
        return HorseRacingEnv(
            track_path=track,
            num_horses=num_horses,
            max_steps=stage.max_steps,
            randomize_horses=stage.randomize_horses,
            randomize_jockey_style=stage.randomize_jockey_style,
        )

register_env("horse_racing_v2", make_env)
print("  Environment factory registered")

# Init Ray
if ray.is_initialized():
    ray.shutdown()

ray.init(num_cpus=2, num_gpus=0, ignore_reinit_error=True)
print("  Ray initialized")
print("  PASSED\n")

# --- Step 3: Build algorithm ---
print("=" * 60)
print("Step 3: Build SAC algorithm with GRU model")
print("=" * 60)

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .environment(
        env="horse_racing_v2",
        env_config={
            "stage_num": 1,
            "track_base_path": ".",
        },
    )
    .framework("torch")
    .training(
        gamma=0.99,
        lr=3e-4,
        train_batch_size_per_learner=512,
        minibatch_size=64,
        num_epochs=6,
        entropy_coeff=0.01,
        clip_param=0.2,
        lambda_=0.95,
    )
    .env_runners(
        num_env_runners=1,
        rollout_fragment_length=64,
    )
    .resources(
        num_gpus=0,
    )
    .reporting(
        min_sample_timesteps_per_iteration=500,
    )
)

config.model = {
    "custom_model": "gru_jockey",
    "custom_model_config": {
        "gru_hidden_size": 128,
        "feature_dim": 128,
        "post_gru_dim": 64,
    },
    "max_seq_len": 64,
}

algo = config.build()
print("  Algorithm built successfully")
print("  PASSED\n")

# --- Step 4: Train a few iterations ---
print("=" * 60)
print("Step 4: Training (3 iterations)")
print("=" * 60)

for i in range(3):
    t0 = time.time()
    result = algo.train()
    elapsed = time.time() - t0
    ts = result.get("timesteps_total", 0)
    reward = result.get("episode_reward_mean", 0)
    episodes = result.get("episodes_total", 0)
    print(f"  Iter {i+1}: timesteps={ts}, reward={reward:.1f}, episodes={episodes}, time={elapsed:.1f}s")

print("  PASSED\n")

# --- Step 5: Save checkpoint ---
print("=" * 60)
print("Step 5: Save and restore checkpoint")
print("=" * 60)

import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = algo.save(tmpdir)
    print(f"  Saved to: {save_path}")

    # Build a new algo and restore
    algo2 = config.build()
    algo2.restore(save_path)
    print("  Restored successfully")

    # Run one more iteration to verify
    result = algo2.train()
    print(f"  Post-restore iter: timesteps={result.get('timesteps_total', 0)}")
    algo2.stop()

print("  PASSED\n")

# --- Step 6: Test compute_single_action ---
print("=" * 60)
print("Step 6: compute_single_action with GRU state")
print("=" * 60)

test_env = make_env({"stage_num": 1, "track_base_path": "."})
obs, info = test_env.reset(seed=0)
state = algo.get_policy().get_initial_state()

for step in range(5):
    action_result = algo.compute_single_action(obs, state=state)
    if isinstance(action_result, tuple):
        action, state, _ = action_result
    else:
        action = action_result
    obs, reward, terminated, truncated, info = test_env.step(action)
    print(f"  Step {step+1}: action={action}, reward={reward:.2f}, progress={info['progress']:.4f}")
    if terminated or truncated:
        break

print("  PASSED\n")

# --- Step 7: ONNX export ---
print("=" * 60)
print("Step 7: ONNX export")
print("=" * 60)

from horse_racing.training.export_onnx import GRUPolicyWrapper, export_onnx

policy = algo.get_policy()
rllib_model = policy.model

wrapper = GRUPolicyWrapper(
    obs_dim=OBS_SIZE,
    hidden_size=128,
    action_dim=2,
    feature_dim=128,
    post_gru_dim=64,
)

# Map weights
wrapper.feature_extractor.weight.data = rllib_model.feature_extractor[0].weight.data.clone()
wrapper.feature_extractor.bias.data = rllib_model.feature_extractor[0].bias.data.clone()
wrapper.gru.load_state_dict(rllib_model.gru.state_dict())
wrapper.post_gru.weight.data = rllib_model.policy_head[0].weight.data.clone()
wrapper.post_gru.bias.data = rllib_model.policy_head[0].bias.data.clone()
wrapper.action_head.weight.data = rllib_model.policy_head[2].weight.data.clone()
wrapper.action_head.bias.data = rllib_model.policy_head[2].bias.data.clone()

with tempfile.TemporaryDirectory() as tmpdir:
    onnx_path = os.path.join(tmpdir, "test_model.onnx")
    export_onnx(wrapper, onnx_path)
    file_size = os.path.getsize(onnx_path)
    print(f"  ONNX file size: {file_size / 1024:.1f} KB")

print("  PASSED\n")

# --- Cleanup ---
algo.stop()
ray.shutdown()

print("=" * 60)
print("ALL STEPS PASSED — training pipeline is working")
print("=" * 60)
