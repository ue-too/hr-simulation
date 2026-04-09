"""Transfer SB3 PPO weights into an RLlib PPO model.

Both use [256, 256] ReLU MLP. SB3 outputs [-1, 1] then remaps ×[4.5, 3.0]
for physics actions; RLlib outputs raw physics values directly. This script
handles the weight scaling so both produce identical physics-space actions.

Usage:
    python scripts/transfer_sb3_to_rllib.py --sb3-model path/to/final_model.zip --output path/to/rllib_checkpoint
    python scripts/transfer_sb3_to_rllib.py --discover   # just print state_dict keys
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from stable_baselines3 import PPO

from horse_racing.rllib_env import HorseRacingRLlibEnv

# Action scaling factors (SB3 [-1,1] → physics space)
TANG_SCALE = 4.5
NORM_SCALE = 3.0


def build_rllib_algo(
    track_path: str = "tracks/tokyo.json",
    horse_count: int = 4,
    reward_phase: int = 3,
):
    """Build an RLlib PPO algo with matching architecture (old API stack)."""
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=HorseRacingRLlibEnv,
            env_config={
                "track_paths": [track_path],
                "min_horse_count": horse_count,
                "max_horse_count": horse_count,
                "reward_phase": reward_phase,
            },
        )
        .env_runners(num_env_runners=0)
        .framework("torch")
        .training(
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",  # must match SB3 MlpPolicy default
            },
        )
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *a, **kw: "shared_policy",
        )
    )
    return config.build()


def discover(track_path: str):
    """Print state_dict keys for both SB3 and RLlib models."""
    import gymnasium as gym
    from gymnasium import spaces

    # SB3 dummy
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(-np.inf, np.inf, (111,), np.float32)
            self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        def reset(self, **kw):
            return np.zeros(111, np.float32), {}
        def step(self, a):
            return np.zeros(111, np.float32), 0, True, False, {}

    sb3_model = PPO("MlpPolicy", DummyEnv(), policy_kwargs=dict(net_arch=[256, 256]))
    print("=== SB3 policy state_dict ===")
    for k, v in sb3_model.policy.state_dict().items():
        print(f"  {k}: {list(v.shape)}")

    print()
    algo = build_rllib_algo(track_path)
    model = algo.get_policy("shared_policy").model
    print("=== RLlib model state_dict ===")
    for k, v in model.state_dict().items():
        print(f"  {k}: {list(v.shape)}")

    algo.stop()


def transfer_weights(sb3_model: PPO, rllib_algo, policy_id: str = "shared_policy"):
    """Copy SB3 PPO weights into RLlib model with action scaling.

    Weight mapping (SB3 → RLlib old API stack):
        mlp_extractor.policy_net.0  → _hidden_layers.0._model.0   (256, 111)
        mlp_extractor.policy_net.2  → _hidden_layers.1._model.0   (256, 256)
        action_net                  → _logits._model.0 [rows 0:2] (2, 256) × scale
        log_std                     → _logits._model.0 [rows 2:4] (scaled)
        mlp_extractor.value_net.0   → _value_branch_separate.0._model.0  (256, 111)
        mlp_extractor.value_net.2   → _value_branch_separate.1._model.0  (256, 256)
        value_net                   → _value_branch._model.0              (1, 256)
    """
    sb3_sd = sb3_model.policy.state_dict()
    policy = rllib_algo.get_policy(policy_id)
    rllib_sd = policy.model.state_dict()

    new_sd = {}

    # Actor hidden layers
    new_sd["_hidden_layers.0._model.0.weight"] = sb3_sd["mlp_extractor.policy_net.0.weight"].clone()
    new_sd["_hidden_layers.0._model.0.bias"] = sb3_sd["mlp_extractor.policy_net.0.bias"].clone()
    new_sd["_hidden_layers.1._model.0.weight"] = sb3_sd["mlp_extractor.policy_net.2.weight"].clone()
    new_sd["_hidden_layers.1._model.0.bias"] = sb3_sd["mlp_extractor.policy_net.2.bias"].clone()

    # Value hidden layers (separate in both SB3 and RLlib)
    new_sd["_value_branch_separate.0._model.0.weight"] = sb3_sd["mlp_extractor.value_net.0.weight"].clone()
    new_sd["_value_branch_separate.0._model.0.bias"] = sb3_sd["mlp_extractor.value_net.0.bias"].clone()
    new_sd["_value_branch_separate.1._model.0.weight"] = sb3_sd["mlp_extractor.value_net.2.weight"].clone()
    new_sd["_value_branch_separate.1._model.0.bias"] = sb3_sd["mlp_extractor.value_net.2.bias"].clone()

    # Value head (no scaling needed)
    new_sd["_value_branch._model.0.weight"] = sb3_sd["value_net.weight"].clone()
    new_sd["_value_branch._model.0.bias"] = sb3_sd["value_net.bias"].clone()

    # Action output: RLlib _logits is (4, 256) — rows [0:2] = means, rows [2:4] = log_std
    # SB3 action_net is (2, 256) — action means in [-1, 1] space
    # Scale means by [4.5, 3.0] to convert to physics space
    scale = torch.tensor([TANG_SCALE, NORM_SCALE], dtype=torch.float32)

    logits_weight = torch.zeros(4, 256)
    logits_bias = torch.zeros(4)

    # Means (rows 0-1): scale weights and biases
    logits_weight[0] = sb3_sd["action_net.weight"][0] * TANG_SCALE
    logits_weight[1] = sb3_sd["action_net.weight"][1] * NORM_SCALE
    logits_bias[0] = sb3_sd["action_net.bias"][0] * TANG_SCALE
    logits_bias[1] = sb3_sd["action_net.bias"][1] * NORM_SCALE

    # Log-std (rows 2-3): shift by log(scale) to preserve exploration width
    # SB3 log_std is in [-1, 1] action space; RLlib log_std is in physics space
    # std_physics = std_raw * scale → log_std_physics = log_std_raw + log(scale)
    sb3_log_std = sb3_sd["log_std"]
    logits_weight[2:] = 0.0  # log_std is state-independent in SB3
    logits_bias[2] = sb3_log_std[0] + math.log(TANG_SCALE)
    logits_bias[3] = sb3_log_std[1] + math.log(NORM_SCALE)

    new_sd["_logits._model.0.weight"] = logits_weight
    new_sd["_logits._model.0.bias"] = logits_bias

    # Load into RLlib model
    policy.model.load_state_dict(new_sd)
    # Sync weights to policy
    policy.set_weights(policy.get_weights())

    return new_sd


def verify_transfer(sb3_model: PPO, rllib_algo, policy_id: str = "shared_policy", n_samples: int = 100):
    """Compare outputs of both models on random observations."""
    rng = np.random.default_rng(42)
    obs_batch = rng.standard_normal((n_samples, 111)).astype(np.float32)

    # SB3 forward (deterministic action means)
    sb3_policy = sb3_model.policy
    sb3_policy.eval()
    with torch.no_grad():
        obs_t = torch.from_numpy(obs_batch)
        features = sb3_policy.mlp_extractor.forward_actor(obs_t)
        raw = sb3_policy.action_net(features)  # [-1, 1] ish
        # Apply same remap as ONNX export
        sb3_actions = torch.zeros_like(raw)
        sb3_actions[:, 0] = raw[:, 0] * TANG_SCALE
        sb3_actions[:, 1] = raw[:, 1] * NORM_SCALE

    # RLlib forward
    policy = rllib_algo.get_policy(policy_id)
    model = policy.model
    model.eval()
    with torch.no_grad():
        model_out, _ = model({"obs": obs_t})
        rllib_actions = model_out[:, :2]  # first 2 = means

    # Compare
    diff = (sb3_actions - rllib_actions).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Also compare values
    with torch.no_grad():
        sb3_vf_features = sb3_policy.mlp_extractor.forward_critic(obs_t)
        sb3_values = sb3_policy.value_net(sb3_vf_features).squeeze(-1)

        # RLlib value function (need to call forward first to populate internal state)
        model({"obs": obs_t})
        rllib_values = model.value_function()

    vf_diff = (sb3_values - rllib_values).abs()
    vf_max_diff = vf_diff.max().item()

    print(f"\nVerification ({n_samples} samples):")
    print(f"  Action means — max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e}")
    print(f"  Value function — max diff: {vf_max_diff:.2e}")
    print(f"  {'PASS' if max_diff < 1e-5 else 'FAIL'}: action diff {'<' if max_diff < 1e-5 else '>'} 1e-5")
    print(f"  {'PASS' if vf_max_diff < 1e-4 else 'FAIL'}: value diff {'<' if vf_max_diff < 1e-4 else '>'} 1e-4")

    return max_diff < 1e-5 and vf_max_diff < 1e-4


def main():
    parser = argparse.ArgumentParser(description="Transfer SB3 PPO weights to RLlib")
    parser.add_argument("--sb3-model", type=str, help="Path to SB3 .zip model")
    parser.add_argument("--output", type=str, default="checkpoints/rllib_from_sb3",
                        help="Output path for RLlib checkpoint")
    parser.add_argument("--track", type=str, default="tracks/tokyo.json")
    parser.add_argument("--reward-phase", type=int, default=3)
    parser.add_argument("--discover", action="store_true",
                        help="Just print state_dict keys and exit")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification after transfer")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, num_cpus=2)

    if args.discover:
        discover(args.track)
        ray.shutdown()
        return

    if not args.sb3_model:
        parser.error("--sb3-model is required (unless --discover)")

    # Load SB3 model
    sb3_path = str(Path(args.sb3_model).resolve())
    print(f"Loading SB3 model: {sb3_path}")
    sb3_model = PPO.load(sb3_path)

    # Build RLlib algo
    print("Building RLlib algo...")
    algo = build_rllib_algo(args.track, reward_phase=args.reward_phase)

    # Transfer weights
    print("Transferring weights...")
    transfer_weights(sb3_model, algo)
    print("Weights transferred.")

    # Verify
    if args.verify:
        ok = verify_transfer(sb3_model, algo)
        if not ok:
            print("\nWARNING: Verification failed — check weight mapping")

    # Save checkpoint
    output_path = str(Path(args.output).resolve())
    save_result = algo.save(output_path)
    print(f"\nRLlib checkpoint saved: {save_result}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
