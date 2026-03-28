"""Export a trained RLlib PPO model to ONNX for browser inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec


class RLlibPolicyWrapper(nn.Module):
    """Wraps the RLlib RLModule's forward_inference into a simple obs→action network."""

    def __init__(self, rl_module):
        super().__init__()
        self.rl_module = rl_module

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        result = self.rl_module.forward_inference({"obs": obs})
        # PPO outputs action_dist_inputs (mean + log_std for Gaussian)
        # Take only the first 2 values (action means)
        return result["action_dist_inputs"][:, :2]


def main():
    parser = argparse.ArgumentParser(description="Export RLlib model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to RLlib checkpoint directory")
    parser.add_argument("--policy", type=str, default="shared_policy",
                        help="Policy ID to export")
    parser.add_argument("--output", type=str, default="checkpoints/horse_jockey_rllib.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--track", type=str, default="tracks/curriculum_1_straight.json")
    args = parser.parse_args()

    ray.init(runtime_env={"working_dir": None})

    from horse_racing.rllib_env import HorseRacingRLlibEnv

    # Recreate the config that was used for training
    config = (
        PPOConfig()
        .environment(
            env=HorseRacingRLlibEnv,
            env_config={"track_path": args.track},
        )
        .env_runners(num_env_runners=0)
        .framework("torch")
        .multi_agent(
            policies={args.policy: PolicySpec()},
            policy_mapping_fn=lambda agent_id, *a, **kw: args.policy,
        )
    )

    algo = config.build_algo()
    checkpoint_path = str(Path(args.checkpoint).resolve())
    algo.restore(checkpoint_path)
    print(f"Restored from {checkpoint_path}")

    # Get the RL module
    module = algo.get_module(args.policy)
    module.eval()

    # Test forward pass
    obs_dim = 26
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        result = module.forward_inference({"obs": dummy})
        action_dist = result["action_dist_inputs"]
        print(f"Module output shape: {action_dist.shape}")
        print(f"Action means: {action_dist[0, :2].numpy()}")

    # Wrap for ONNX export
    wrapper = RLlibPolicyWrapper(module)
    wrapper.eval()

    with torch.no_grad():
        test_out = wrapper(dummy)
        print(f"Wrapper output: {test_out.numpy()}")

    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={
            "obs": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    # Verify
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(output_path))
    test_obs = np.zeros((1, obs_dim), dtype=np.float32)
    result = session.run(None, {"obs": test_obs})
    print(f"\nONNX verification:")
    print(f"  Output: {result[0]}")

    file_size = output_path.stat().st_size
    print(f"\nExported to {output_path} ({file_size / 1024:.1f} KB)")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
