"""Behavioral cloning from race recordings — pretrain policy before RL."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .action import NORMAL_LEVELS, NUM_ACTIONS, NUM_NORMAL, TANGENTIAL_LEVELS


def _snap_to_index(value: float, levels: list[float]) -> int:
    """Snap a continuous value to the nearest discrete level index."""
    best = 0
    best_dist = abs(value - levels[0])
    for i, lev in enumerate(levels):
        d = abs(value - lev)
        if d < best_dist:
            best_dist = d
            best = i
    return best


def _encode_action(tangential: float, normal: float) -> int:
    """Map continuous (tangential, normal) to a flat discrete action index."""
    ti = _snap_to_index(tangential, TANGENTIAL_LEVELS)
    ni = _snap_to_index(normal, NORMAL_LEVELS)
    return ti * NUM_NORMAL + ni


def extract_demonstrations(
    recording_path: str | Path,
    player_horse_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (observations, actions) from a race recording.

    Args:
        recording_path: Path to a race JSON recording with obs data.
        player_horse_id: Which horse to extract. If None, auto-detect
            the horse with the most input variation (likely the player).

    Returns:
        (obs_array, action_array) where obs is (N, OBS_SIZE) float32
        and actions is (N,) int64.
    """
    with open(recording_path) as f:
        race = json.load(f)

    frames = race["frames"]

    # Auto-detect player horse by input variation
    if player_horse_id is None:
        best_id = 0
        best_var = 0
        for hid in range(race["horseCount"]):
            vals = set()
            for frame in frames[:500]:
                inp = frame.get("inputs", {}).get(str(hid), {})
                if inp:
                    vals.add((inp.get("t", 0), inp.get("n", 0)))
            if len(vals) > best_var:
                best_var = len(vals)
                best_id = hid
        player_horse_id = best_id

    obs_list = []
    action_list = []

    for frame in frames:
        horse = None
        for h in frame["horses"]:
            if h["id"] == player_horse_id:
                horse = h
                break
        if horse is None or horse.get("finished", False):
            continue
        if "obs" not in horse:
            continue

        inp = frame.get("inputs", {}).get(str(player_horse_id), {})
        if not inp:
            continue

        obs_list.append(np.array(horse["obs"], dtype=np.float32))
        action_list.append(_encode_action(inp["t"], inp["n"]))

    return np.array(obs_list), np.array(action_list, dtype=np.int64)


def extract_from_multiple(
    recording_paths: list[str | Path],
    player_horse_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and concatenate demonstrations from multiple recordings."""
    all_obs = []
    all_actions = []
    for path in recording_paths:
        obs, actions = extract_demonstrations(path, player_horse_id)
        if len(obs) > 0:
            all_obs.append(obs)
            all_actions.append(actions)
    return np.concatenate(all_obs), np.concatenate(all_actions)


def pretrain_bc(
    model,
    obs: np.ndarray,
    actions: np.ndarray,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[float]:
    """Pretrain a Stable Baselines3 PPO model's policy via behavioral cloning.

    Args:
        model: A stable_baselines3.PPO model instance.
        obs: Observations array (N, OBS_SIZE).
        actions: Action indices array (N,).
        epochs: Number of BC training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for BC.
        device: Torch device.

    Returns:
        List of per-epoch average loss values.
    """
    policy = model.policy
    policy.train()

    # Extract the action network: features_extractor → mlp_extractor → action_net
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    act_tensor = torch.tensor(actions, dtype=torch.long, device=device)

    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        count = 0
        for batch_obs, batch_act in loader:
            features = policy.features_extractor(batch_obs)
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)

            loss = criterion(logits, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_obs)
            count += len(batch_obs)

        avg_loss = epoch_loss / count
        losses.append(avg_loss)
        print(f"  BC epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    policy.eval()
    return losses
