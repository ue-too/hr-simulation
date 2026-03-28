"""Export a trained SB3 PPO model to ONNX for browser inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO


class PolicyNetwork(nn.Module):
    """Wraps the SB3 policy's MLP to produce deterministic actions."""

    def __init__(self, sb3_policy):
        super().__init__()
        # SB3's MlpPolicy has: features_extractor → mlp_extractor → action_net
        self.features_extractor = sb3_policy.features_extractor
        self.mlp_extractor = sb3_policy.mlp_extractor
        self.action_net = sb3_policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor(features)
        # action_net outputs the mean of the Gaussian policy
        action_mean = self.action_net(latent_pi)
        return action_mean


def main():
    parser = argparse.ArgumentParser(description="Export SB3 model to ONNX")
    parser.add_argument("--model", type=str, default="checkpoints/straight_sb3",
                        help="Path to SB3 model zip")
    parser.add_argument("--output", type=str, default="checkpoints/horse_jockey.onnx",
                        help="Output ONNX file path")
    args = parser.parse_args()

    print(f"Loading model from {args.model}")
    model = PPO.load(args.model)
    policy = model.policy

    # Print network architecture
    print(f"Observation space: {model.observation_space.shape}")
    print(f"Action space: {model.action_space.shape}")
    print(f"Policy network:")
    print(f"  Features extractor: {policy.features_extractor}")
    print(f"  MLP extractor: {policy.mlp_extractor}")
    print(f"  Action net: {policy.action_net}")

    # Wrap for clean ONNX export
    wrapper = PolicyNetwork(policy)
    wrapper.eval()

    # Create dummy input
    obs_dim = model.observation_space.shape[0]
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)

    # Verify forward pass
    with torch.no_grad():
        test_output = wrapper(dummy)
        print(f"  Output shape: {test_output.shape}")
        print(f"  Test output: {test_output.numpy()}")

    # Export to ONNX
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
        dynamo=False,  # Use legacy exporter to embed all weights in the .onnx file
    )

    # Verify the exported model
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(output_path))
    test_obs = np.zeros((1, obs_dim), dtype=np.float32)
    result = session.run(None, {"obs": test_obs})
    print(f"\nONNX verification:")
    print(f"  Input shape: {test_obs.shape}")
    print(f"  Output shape: {result[0].shape}")
    print(f"  Output: {result[0]}")

    file_size = output_path.stat().st_size
    print(f"\nExported to {output_path} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
