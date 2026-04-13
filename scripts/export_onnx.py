"""Export a trained SB3 PPO model to ONNX format."""

import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch
from stable_baselines3 import PPO

from horse_racing.action import NUM_ACTIONS
from horse_racing.core.observation import OBS_SIZE


def export(checkpoint_path: str, output_path: str):
    model = PPO.load(checkpoint_path)
    policy = model.policy

    # Extract the action network
    # SB3 MlpPolicy: features_extractor → mlp_extractor → action_net
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.features_extractor = policy.features_extractor
            self.mlp_extractor = policy.mlp_extractor
            self.action_net = policy.action_net

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            features = self.features_extractor(obs)
            latent_pi, _ = self.mlp_extractor(features)
            return self.action_net(latent_pi)

    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    dummy_input = torch.randn(1, OBS_SIZE)
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={
            "obs": {0: "batch"},
            "actions": {0: "batch"},
        },
        opset_version=17,
    )

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Compare outputs
    session = ort.InferenceSession(output_path)
    test_obs = np.random.randn(4, OBS_SIZE).astype(np.float32)

    with torch.no_grad():
        torch_out = wrapper(torch.from_numpy(test_obs)).numpy()
    onnx_out = session.run(["actions"], {"obs": test_obs})[0]

    max_diff = np.max(np.abs(torch_out - onnx_out))
    print(f"Exported to {output_path}")
    print(f"Output shape: {onnx_out.shape} (expected: (4, {NUM_ACTIONS}))")
    print(f"Max difference (PyTorch vs ONNX): {max_diff:.2e}")
    assert max_diff < 1e-5, f"Output mismatch: {max_diff}"
    print("Validation passed!")


def main():
    parser = argparse.ArgumentParser(description="Export SB3 model to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to SB3 checkpoint (.zip)")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX path")
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
