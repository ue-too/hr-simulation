"""Reward function — delta-progress + finish bonus + finishing speed + overtake + collision."""

from __future__ import annotations

_FINISH_BONUS = {1: 10.0, 2: 5.0, 3: 2.0}
FINISHING_SPEED_BONUS = 5.0
DEPLETION_PENALTY = 15.0
DEPLETION_THRESHOLD = 0.6  # speed_ratio below this triggers penalty
OVERTAKE_BONUS = 0.5
RAIL_COLLISION_PENALTY = 0.002


def compute_reward(
    prev_progress: float,
    curr_progress: float,
    finish_order: int | None,
    finishing_speed: float = 0.0,
    cruise_speed: float = 13.0,
    overtakes: int = 0,
    rail_contact: bool = False,
) -> float:
    """Compute reward for one step.

    Args:
        prev_progress: Track progress at previous tick [0, 1].
        curr_progress: Track progress at current tick [0, 1].
        finish_order: Finishing position (1-based) if horse finished, else None.
        finishing_speed: Horse's tangential velocity when crossing the finish.
        cruise_speed: Horse's base cruise speed (for normalization).
        overtakes: Number of opponents overtaken this step (positive = passed them).
        rail_contact: Whether the horse touched a rail this tick.
    """
    reward = curr_progress - prev_progress
    reward += OVERTAKE_BONUS * overtakes
    if rail_contact:
        reward -= RAIL_COLLISION_PENALTY
    if finish_order is not None:
        reward += _FINISH_BONUS.get(finish_order, 0.0)
        speed_ratio = finishing_speed / cruise_speed if cruise_speed > 0 else 0.0
        # Bonus for finishing fast
        reward += FINISHING_SPEED_BONUS * speed_ratio
        # Penalty for finishing depleted (below threshold)
        if speed_ratio < DEPLETION_THRESHOLD:
            reward -= DEPLETION_PENALTY * (DEPLETION_THRESHOLD - speed_ratio)
    return reward
