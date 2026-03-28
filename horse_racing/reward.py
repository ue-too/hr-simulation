"""Reward function module — modular reward shaping for competitive racing."""

from __future__ import annotations

# Finish order bonuses: 1st place gets the most, 4th gets the least
FINISH_ORDER_BONUS = [100.0, 50.0, 25.0, 10.0]


def compute_reward(
    obs_prev: dict,
    obs_curr: dict,
    collision_occurred: bool,
    placement: int = 1,
    num_horses: int = 4,
    finish_order: int | None = None,
) -> float:
    """Compute the per-step reward for a single horse.

    Args:
        obs_prev: Observation dict from previous step.
        obs_curr: Observation dict from current step.
        collision_occurred: Whether the horse collided this tick.
        placement: Current race placement (1 = first, num_horses = last).
        num_horses: Total number of horses in the race.
        finish_order: If horse just finished, its finish position (1-indexed).
            None if not finished this step.

    Returns:
        Scalar reward.
    """
    reward = 0.0

    # Forward progress — primary signal, scaled to dominate
    reward += 100.0 * (obs_curr["track_progress"] - obs_prev["track_progress"])

    # Speed efficiency — small bonus for going fast relative to capability
    max_spd = obs_curr["effective_max_speed"]
    if max_spd > 1e-6:
        reward += 0.05 * (obs_curr["tangential_vel"] / max_spd)

    # Placement bonus — per-tick incentive to be ahead of others
    if num_horses > 1:
        reward += 0.02 * (num_horses - placement) / (num_horses - 1)

    # Lane-holding penalty — reduced from 0.05, less punitive on curves
    reward -= 0.01 * abs(obs_curr["displacement"])

    # Collision penalty — reduced, some bumping is strategic with push traits
    if collision_occurred:
        reward -= 0.5

    # Stamina crisis — only penalize severe depletion
    if obs_curr["stamina_ratio"] < 0.10:
        reward -= 0.2

    # Cornering stress — penalize exceeding grip threshold
    cornering_margin = obs_curr.get("cornering_margin", float("inf"))
    if cornering_margin < 0:
        reward -= 0.01 * abs(cornering_margin)

    # Finish order bonus — rewards racing position, not just finishing
    if finish_order is not None:
        idx = min(finish_order - 1, len(FINISH_ORDER_BONUS) - 1)
        reward += FINISH_ORDER_BONUS[idx]

    return reward
