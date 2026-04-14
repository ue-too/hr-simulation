"""Self-play opponent: uses a frozen PPO model snapshot as an opponent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..action import NUM_ACTIONS, decode_action
from ..core.observation import OBS_SIZE, build_observations
from .scripted import Strategy

if TYPE_CHECKING:
    from ..core.race import Race
    from ..core.types import Horse, InputState


class SelfPlayStrategy(Strategy):
    """Opponent that runs inference from a frozen PPO policy.

    The policy network is extracted as a simple callable (features → logits)
    so we don't need to depend on stable_baselines3 at import time.
    """

    def __init__(self, predict_fn, race_ref: Race, horse_id: int):
        self._predict_fn = predict_fn
        self._race_ref = race_ref
        self._horse_id = horse_id

    def act(self, progress: float) -> int:
        # Not used — act_continuous is always available.
        return 0

    def act_continuous(self, horse: Horse) -> InputState | None:
        from ..core.types import InputState

        all_obs = build_observations(self._race_ref)
        obs = all_obs[self._horse_id].astype(np.float32)
        action = self._predict_fn(obs)
        tang, norm = decode_action(action)
        return InputState(tang, norm)


def make_self_play_predict(model_path: str):
    """Load a PPO checkpoint and return a predict function.

    Returns a callable: obs (np.ndarray[OBS_SIZE]) -> int (action index).
    """
    import torch
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    policy = model.policy
    policy.eval()

    @torch.no_grad()
    def predict(obs: np.ndarray) -> int:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        features = policy.features_extractor(obs_tensor)
        latent_pi, _ = policy.mlp_extractor(features)
        logits = policy.action_net(latent_pi)
        return int(logits.argmax(dim=-1).item())

    return predict
