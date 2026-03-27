"""Reward function module — modular reward shaping for experimentation."""

from __future__ import annotations


def compute_reward(
    obs_prev: dict,
    obs_curr: dict,
    collision_occurred: bool,
) -> float:
    """Compute the per-step reward for a single horse.

    Args:
        obs_prev: Observation dict from previous step.
        obs_curr: Observation dict from current step.
        collision_occurred: Whether the horse collided this tick.

    Returns:
        Scalar reward.
    """
    reward = 0.0

    # Forward progress (primary signal)
    reward += 1.0 * (obs_curr["track_progress"] - obs_prev["track_progress"])

    # Speed bonus
    max_spd = obs_curr["effective_max_speed"]
    if max_spd > 1e-6:
        reward += 0.1 * (obs_curr["tangential_vel"] / max_spd)

    # Lane-holding penalty
    reward -= 0.05 * abs(obs_curr["displacement"])

    # Collision penalty
    if collision_occurred:
        reward -= 1.0

    # Stamina management — penalty for running on empty
    if obs_curr["stamina_ratio"] < 0.15:
        reward -= 0.1

    # Finish bonus
    if obs_curr["track_progress"] >= 1.0:
        reward += 100.0

    return reward
