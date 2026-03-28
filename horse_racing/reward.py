"""Reward function module — tuned for competitive racing with speed incentives."""

from __future__ import annotations

# Finish order bonuses: 1st place gets the most, 4th gets the least
FINISH_ORDER_BONUS = [500.0, 300.0, 150.0, 50.0]


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

    # Forward progress — dominant signal
    reward += 1000.0 * (obs_curr["track_progress"] - obs_prev["track_progress"])

    # Speed bonus — rewards going fast relative to max capability
    max_spd = obs_curr["effective_max_speed"]
    if max_spd > 1e-6:
        reward += 0.3 * (obs_curr["tangential_vel"] / max_spd)

    # Speed above cruise bonus — rewards pushing beyond auto-cruise
    cruise_spd = obs_curr["effective_cruise_speed"]
    vel = obs_curr["tangential_vel"]
    if max_spd > cruise_spd + 1e-6 and vel > cruise_spd:
        reward += 0.2 * (vel - cruise_spd) / (max_spd - cruise_spd)

    # Alive penalty — creates time pressure to finish faster
    reward -= 0.1

    # Placement bonus — per-tick incentive to be ahead of others
    if num_horses > 1:
        reward += 0.1 * (num_horses - placement) / (num_horses - 1)

    # Collision penalty — reduced, some bumping is strategic
    if collision_occurred:
        reward -= 0.3

    # Stamina crisis — only penalize severe depletion
    if obs_curr["stamina_ratio"] < 0.10:
        reward -= 0.1

    # Finish order bonus — large terminal reward for racing position
    if finish_order is not None:
        idx = min(finish_order - 1, len(FINISH_ORDER_BONUS) - 1)
        reward += FINISH_ORDER_BONUS[idx]

    return reward
