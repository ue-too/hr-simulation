"""Local smoke test for Phase 2 archetype training.

Runs ~5K steps per archetype with 1 env to verify the pipeline works
before committing to full Colab training.
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
from horse_racing.env import HorseRacingSingleEnv
from horse_racing.reward import ARCHETYPES, compute_reward
from horse_racing.types import HorseAction

# ── Config ──────────────────────────────────────────────────────────────
ONNX_PATH = "/Users/niuee/dev/track reference/trained jockey/better_baseline_jockey.onnx"
TIMESTEPS_PER_ARCHETYPE = 5_000
TRAINING_TRACKS = [
    "tracks/curriculum_2_gentle_oval.json",
    "tracks/curriculum_3_tight_oval.json",
    "tracks/hanshin.json",
]


# ── ArchetypeEnv (same as notebook) ────────────────────────────────────
class ArchetypeEnv(gym.Wrapper):
    def __init__(self, tracks, max_steps, archetype):
        self.tracks = tracks if isinstance(tracks, list) else [tracks]
        self.max_steps = max_steps
        self.archetype = archetype
        env = HorseRacingSingleEnv(
            track_path=random.choice(self.tracks), max_steps=max_steps
        )
        super().__init__(env)

    def step(self, action):
        obs_array, _, terminated, truncated, info = self.env.step(action)
        obs_curr = self.env.engine.get_observations()[0]
        placements = self.env.engine.get_placements()
        finish_order = placements[0] if obs_curr["finished"] else None

        reward = compute_reward(
            self._prev_obs_for_reward, obs_curr, obs_curr["collision"],
            placement=placements[0],
            num_horses=self.env.engine.horse_count,
            finish_order=finish_order,
            archetype=self.archetype,
            prev_placement=self._prev_placement,
        )
        self._prev_placement = placements[0]
        self._prev_obs_for_reward = obs_curr
        return obs_array, reward, terminated, truncated, info

    def reset(self, **kwargs):
        new_track = random.choice(self.tracks)
        self.env = HorseRacingSingleEnv(
            track_path=new_track, max_steps=self.max_steps
        )
        obs, info = self.env.reset(**kwargs)
        self._prev_obs_for_reward = self.env.engine.get_observations()[0]
        self._prev_placement = 1
        return obs, info


# ── ONNX export helper ─────────────────────────────────────────────────
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
    print("Phase 2 Local Smoke Test")
    print("=" * 60)

    # ── 1. Load baseline ONNX ──────────────────────────────────────────
    print(f"\n1. Loading baseline ONNX: {ONNX_PATH}")
    onnx_model = onnx.load(ONNX_PATH)
    onnx_weights = {}
    for init in onnx_model.graph.initializer:
        onnx_weights[init.name] = torch.tensor(
            np.array(onnx.numpy_helper.to_array(init))
        )

    layer0_out = onnx_weights["mlp_extractor.policy_net.0.weight"].shape[0]
    layer1_out = onnx_weights["mlp_extractor.policy_net.2.weight"].shape[0]
    print(f"   Architecture: 26 → {layer0_out} → {layer1_out} → 2")

    # Create base SB3 model and inject weights
    dummy_env = HorseRacingSingleEnv(track_path="tracks/hanshin.json", max_steps=5000)
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
    print("   Phase 1 weights loaded into SB3 model ✓")

    # ── 2. Train each archetype ────────────────────────────────────────
    checkpoint_dir = Path("checkpoints/archetypes_test")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    archetype_models = {}
    total_start = time.time()

    for arch_idx, archetype in enumerate(ARCHETYPES):
        print(f"\n2.{arch_idx+1} [{arch_idx+1}/{len(ARCHETYPES)}] Training: {archetype} — {TIMESTEPS_PER_ARCHETYPE:,} steps")

        env = DummyVecEnv([
            lambda a=archetype: ArchetypeEnv(
                tracks=TRAINING_TRACKS, max_steps=3000, archetype=a
            )
        ])

        model = PPO(
            "MlpPolicy", env, verbose=0,
            policy_kwargs={"net_arch": [layer0_out, layer1_out]},
            n_steps=512, batch_size=64, n_epochs=5,
            learning_rate=1e-4,
            device="cpu",
        )
        model.policy.load_state_dict(base_model.policy.state_dict())

        model.learn(total_timesteps=TIMESTEPS_PER_ARCHETYPE)

        save_path = checkpoint_dir / f"jockey_{archetype}"
        model.save(str(save_path))
        archetype_models[archetype] = model
        env.close()

        # Quick reward check
        mean_r = (
            sum(ep["r"] for ep in model.ep_info_buffer) / len(model.ep_info_buffer)
            if model.ep_info_buffer else 0
        )
        print(f"   Done — mean reward: {mean_r:.1f} ✓")

    train_time = time.time() - total_start
    print(f"\n   All archetypes trained in {train_time:.1f}s")

    # ── 3. Export to ONNX ──────────────────────────────────────────────
    print("\n3. Exporting ONNX models...")
    obs_dim = 102
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)

    for archetype, model in archetype_models.items():
        wrapper = PolicyNetwork(model.policy)
        wrapper.eval()

        onnx_path = str(checkpoint_dir / f"jockey_{archetype}.onnx")
        torch.onnx.export(
            wrapper, dummy, onnx_path,
            input_names=["obs"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17, dynamo=False,
        )

        # Verify
        sess = ort.InferenceSession(onnx_path)
        test = np.zeros((1, obs_dim), dtype=np.float32)
        result = sess.run(["action"], {"obs": test})
        print(f"   {archetype:15s} → output: [{result[0][0][0]:.4f}, {result[0][0][1]:.4f}] ✓")

    # ── 4. Head-to-head eval ───────────────────────────────────────────
    print("\n4. Head-to-head race (Hanshin)...")
    engine = HorseRacingEngine("tracks/hanshin.json")

    sessions = {}
    for arch in ARCHETYPES:
        onnx_path = str(checkpoint_dir / f"jockey_{arch}.onnx")
        sessions[arch] = ort.InferenceSession(onnx_path)

    for tick in range(3000):
        all_obs = engine.get_observations()
        actions = []
        for i, arch in enumerate(ARCHETYPES):
            arr = engine.obs_to_array(all_obs[i]).reshape(1, -1)
            action = sessions[arch].run(["action"], {"obs": arr})[0][0]
            actions.append(HorseAction(float(action[0]), float(action[1])))
        engine.step(actions)

        if tick % 1000 == 999:
            placements = engine.get_placements()
            print(f"   Tick {tick+1}:")
            for i, arch in enumerate(ARCHETYPES):
                o = all_obs[i]
                print(
                    f"     {arch:15s} | P{placements[i]} | "
                    f"progress: {o['track_progress']:.3f} | "
                    f"vel: {o['tangential_vel']:.1f} | "
                    f"stamina: {o['stamina_ratio']:.3f}"
                )

    placements = engine.get_placements()
    final_obs = engine.get_observations()
    print(f"\n   FINAL:")
    for i, arch in enumerate(ARCHETYPES):
        o = final_obs[i]
        status = "FINISHED" if o["finished"] else f"{o['track_progress']:.1%}"
        print(f"     P{placements[i]}: {arch:15s} | {status} | stamina: {o['stamina_ratio']:.3f}")

    print(f"\n{'='*60}")
    print("Smoke test PASSED ✓")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
