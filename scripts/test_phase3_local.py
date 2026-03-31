"""Local smoke test for Phase 3 self-play training.

Runs a tiny training (5K steps, 1 round, 1 env) to verify the pipeline works.
Uses the Phase 2 ONNX models (or better baseline) as opponents.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import onnx
import onnx.numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from horse_racing.engine import HorseRacingEngine
from horse_racing.reward import ARCHETYPES, compute_reward
from horse_racing.self_play_env import SelfPlayEnv
from horse_racing.types import HorseAction

# ── Config ──────────────────────────────────────────────────────────────
BASELINE_ONNX = "/Users/niuee/dev/track reference/trained jockey/better_baseline_jockey.onnx"
TIMESTEPS_PER_ARCHETYPE = 5_000
TRAINING_TRACKS = [
    "tracks/curriculum_2_gentle_oval.json",
    "tracks/hanshin.json",
]


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


def main():
    print("=" * 60)
    print("Phase 3 Self-Play Local Smoke Test")
    print("=" * 60)

    # ── 1. Load baseline ONNX ──────────────────────────────────────────
    print(f"\n1. Loading baseline ONNX for opponents")
    onnx_model = onnx.load(BASELINE_ONNX)
    onnx_weights = {}
    for init in onnx_model.graph.initializer:
        onnx_weights[init.name] = torch.tensor(
            np.array(onnx.numpy_helper.to_array(init))
        )

    layer0_out = onnx_weights["mlp_extractor.policy_net.0.weight"].shape[0]
    layer1_out = onnx_weights["mlp_extractor.policy_net.2.weight"].shape[0]
    print(f"   Architecture: 26 → {layer0_out} → {layer1_out} → 2")

    # Use baseline as opponent (all 4 opponents use same model for smoke test)
    opponent_paths = [BASELINE_ONNX] * 4
    print(f"   Using baseline as all 4 opponents")

    # ── 2. Test SelfPlayEnv directly ───────────────────────────────────
    print("\n2. Testing SelfPlayEnv...")
    env = SelfPlayEnv(
        tracks=TRAINING_TRACKS,
        max_steps=500,
        opponent_onnx_paths=opponent_paths,
        trainee_archetype="front_runner",
        min_opponents=3,
        max_opponents=7,
    )

    obs, info = env.reset()
    print(f"   Obs shape: {obs.shape}, opponents: {env._num_opponents}")

    total_reward = 0
    total_overtakes = 0
    for step in range(200):
        action = np.array([3.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_overtakes += info["overtakes"]
        if terminated or truncated:
            break

    print(f"   Steps: {step+1}, Reward: {total_reward:.1f}, Overtakes: {total_overtakes}")
    print(f"   Final placement: P{info['placement']}, Finished: {info['finished']}")
    print(f"   SelfPlayEnv works ✓")
    env.close()

    # ── 3. Train one archetype with SB3 ────────────────────────────────
    print(f"\n3. Training front_runner — {TIMESTEPS_PER_ARCHETYPE} steps")

    def make_env():
        return SelfPlayEnv(
            tracks=TRAINING_TRACKS,
            max_steps=3000,
            opponent_onnx_paths=opponent_paths,
            trainee_archetype="front_runner",
            min_opponents=3,
            max_opponents=4,
        )

    vec_env = DummyVecEnv([make_env])

    # Create SB3 model from baseline weights
    from horse_racing.env import HorseRacingSingleEnv
    dummy_env = HorseRacingSingleEnv(track_path="tracks/hanshin.json")
    base_model = PPO(
        "MlpPolicy", dummy_env, verbose=0,
        policy_kwargs={"net_arch": [layer0_out, layer1_out]},
        device="cpu",
    )
    dummy_env.close()

    state_dict = base_model.policy.state_dict()
    for name in onnx_weights:
        if name in state_dict:
            state_dict[name] = onnx_weights[name]
    base_model.policy.load_state_dict(state_dict)

    # Train
    model = PPO(
        "MlpPolicy", vec_env, verbose=0,
        policy_kwargs={"net_arch": [layer0_out, layer1_out]},
        n_steps=512, batch_size=64, n_epochs=5,
        learning_rate=3e-4,
        device="cpu",
    )
    model.policy.load_state_dict(base_model.policy.state_dict())

    model.learn(total_timesteps=TIMESTEPS_PER_ARCHETYPE)
    vec_env.close()

    mean_r = (
        sum(ep["r"] for ep in model.ep_info_buffer) / len(model.ep_info_buffer)
        if model.ep_info_buffer else 0
    )
    print(f"   Mean reward: {mean_r:.1f} ✓")

    # ── 4. Export to ONNX ──────────────────────────────────────────────
    print("\n4. Exporting ONNX...")
    checkpoint_dir = Path("checkpoints/phase3_test")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    wrapper = PolicyNetwork(model.policy)
    wrapper.eval()
    onnx_path = str(checkpoint_dir / "jockey_front_runner.onnx")
    dummy = torch.zeros(1, 102, dtype=torch.float32)
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["obs"], output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    sess = ort.InferenceSession(onnx_path)
    result = sess.run(["action"], {"obs": np.zeros((1, 102), dtype=np.float32)})
    print(f"   Exported: {onnx_path}")
    print(f"   Output: [{result[0][0][0]:.4f}, {result[0][0][1]:.4f}] ✓")

    # ── 5. Quick head-to-head ──────────────────────────────────────────
    print("\n5. Head-to-head (trained vs 3 baselines on Hanshin)...")
    engine = HorseRacingEngine("tracks/hanshin.json")
    trained_sess = ort.InferenceSession(onnx_path)
    baseline_sess = ort.InferenceSession(BASELINE_ONNX)

    for tick in range(2000):
        all_obs = engine.get_observations()
        actions = []
        for i in range(4):
            arr = engine.obs_to_array(all_obs[i]).reshape(1, -1)
            if i == 0:
                action = trained_sess.run(["action"], {"obs": arr})[0][0]
            else:
                action = baseline_sess.run(["action"], {"obs": arr})[0][0]
            actions.append(HorseAction(float(action[0]), float(action[1])))
        engine.step(actions)

    placements = engine.get_placements()
    final_obs = engine.get_observations()
    print(f"   Trained (front_runner): P{placements[0]} | progress={final_obs[0]['track_progress']:.3f}")
    for i in range(1, 4):
        print(f"   Baseline {i}:             P{placements[i]} | progress={final_obs[i]['track_progress']:.3f}")

    print(f"\n{'='*60}")
    print("Phase 3 smoke test PASSED ✓")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
