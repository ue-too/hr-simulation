"""Export a trained RLlib SAC model with GRU to ONNX for browser inference.

The exported model takes two inputs:
  - obs: (1, obs_dim) float32 — the observation vector
  - hidden_in: (1, gru_size) float32 — GRU hidden state from previous tick

And produces two outputs:
  - action: (1, 2) float32 — [effort, lane]
  - hidden_out: (1, gru_size) float32 — GRU hidden state for next tick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from horse_racing.env.observation import OBS_SIZE


class GRUPolicyWrapper(nn.Module):
    """Wraps the RLlib policy network for clean ONNX export.

    Expects the policy to have:
    - A feature extractor (linear layer)
    - A GRU layer
    - An action output head (linear layer)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_size: int,
        action_dim: int,
        feature_dim: int = 128,
        post_gru_dim: int = 64,
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Linear(obs_dim, feature_dim)
        self.gru = nn.GRU(feature_dim, hidden_size, batch_first=True)
        self.post_gru = nn.Linear(hidden_size, post_gru_dim)
        self.action_head = nn.Linear(post_gru_dim, action_dim)
        self.activation = nn.ReLU()

    def forward(
        self, obs: torch.Tensor, hidden_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # obs: (batch, obs_dim)
        # hidden_in: (batch, hidden_size) -> reshape for GRU: (1, batch, hidden_size)
        features = self.activation(self.feature_extractor(obs))
        features = features.unsqueeze(1)  # (batch, 1, feature_dim) for single timestep
        hidden = hidden_in.unsqueeze(0)  # (1, batch, hidden_size)

        gru_out, hidden_out = self.gru(features, hidden)
        gru_out = gru_out.squeeze(1)  # (batch, hidden_size)
        hidden_out = hidden_out.squeeze(0)  # (batch, hidden_size)

        post = self.activation(self.post_gru(gru_out))
        action = torch.tanh(self.action_head(post))  # [-1, 1] for effort and lane

        return action, hidden_out


def load_weights_from_rllib(wrapper: GRUPolicyWrapper, checkpoint_path: str) -> None:
    """Load weights from an RLlib checkpoint into the wrapper.

    RLlib stores model weights in a specific format. This function maps
    the RLlib weight names to our wrapper's parameter names.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # RLlib weight mapping depends on the exact model config
    # This is a template — adapt based on actual checkpoint structure
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "worker" in checkpoint:
        # RLlib 2.x format
        policy_state = checkpoint["worker"]["policy_state"]["default_policy"]
        state_dict = policy_state.get("model", policy_state)
    else:
        state_dict = checkpoint

    # Try direct load first
    try:
        wrapper.load_state_dict(state_dict, strict=False)
        print("Loaded weights directly")
    except Exception as e:
        print(f"Direct load failed ({e}), attempting manual mapping...")
        # Manual mapping would go here based on actual RLlib layer names
        raise


def export_onnx(
    wrapper: GRUPolicyWrapper,
    output_path: str | Path,
    obs_dim: int = OBS_SIZE,
    hidden_size: int = 128,
) -> None:
    """Export the wrapper to ONNX format."""
    wrapper.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    dummy_hidden = torch.zeros(1, hidden_size, dtype=torch.float32)

    # Verify forward pass
    with torch.no_grad():
        action, hidden_out = wrapper(dummy_obs, dummy_hidden)
        print(f"Forward pass OK: action={action.numpy()}, hidden_out shape={hidden_out.shape}")

    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_hidden),
        str(output_path),
        input_names=["obs", "hidden_in"],
        output_names=["action", "hidden_out"],
        dynamic_axes={
            "obs": {0: "batch"},
            "hidden_in": {0: "batch"},
            "action": {0: "batch"},
            "hidden_out": {0: "batch"},
        },
        opset_version=17,
    )

    # Verify ONNX
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(output_path))
    test_obs = np.zeros((1, obs_dim), dtype=np.float32)
    test_hidden = np.zeros((1, hidden_size), dtype=np.float32)
    result = session.run(None, {"obs": test_obs, "hidden_in": test_hidden})

    print(f"ONNX verification:")
    print(f"  Inputs: obs {test_obs.shape}, hidden_in {test_hidden.shape}")
    print(f"  Outputs: action {result[0].shape}, hidden_out {result[1].shape}")
    print(f"  Action: {result[0]}")

    file_size = output_path.stat().st_size
    print(f"Exported to {output_path} ({file_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Export RLlib model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to RLlib checkpoint")
    parser.add_argument("--output", type=str, default="models/v2/jockey.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="GRU hidden state size")
    args = parser.parse_args()

    wrapper = GRUPolicyWrapper(
        obs_dim=OBS_SIZE,
        hidden_size=args.hidden_size,
        action_dim=2,
    )

    load_weights_from_rllib(wrapper, args.checkpoint)
    export_onnx(wrapper, args.output, hidden_size=args.hidden_size)


if __name__ == "__main__":
    main()
