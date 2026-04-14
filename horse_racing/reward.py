"""Reward function — delta-progress + finish bonus + stamina efficiency + overtake + collision."""

from __future__ import annotations

_FINISH_BONUS = {1: 10.0, 2: 5.0, 3: 2.0}
STAMINA_EFFICIENCY_BONUS = 3.0
OVERTAKE_BONUS = 0.1
RAIL_COLLISION_PENALTY = 0.005


def compute_reward(
    prev_progress: float,
    curr_progress: float,
    finish_order: int | None,
    current_stamina: float = 1.0,
    max_stamina: float = 100.0,
    overtakes: int = 0,
    rail_contact: bool = False,
) -> float:
    """Compute reward for one step.

    Args:
        prev_progress: Track progress at previous tick [0, 1].
        curr_progress: Track progress at current tick [0, 1].
        finish_order: Finishing position (1-based) if horse finished, else None.
        current_stamina: Horse's current stamina.
        max_stamina: Horse's maximum stamina.
        overtakes: Number of opponents overtaken this step (positive = passed them).
        rail_contact: Whether the horse touched a rail this tick.
    """
    reward = curr_progress - prev_progress
    reward += OVERTAKE_BONUS * overtakes
    if rail_contact:
        reward -= RAIL_COLLISION_PENALTY
    if finish_order is not None:
        reward += _FINISH_BONUS.get(finish_order, 0.0)
        stamina_pct = current_stamina / max_stamina if max_stamina > 0 else 0.0
        reward += STAMINA_EFFICIENCY_BONUS * (1.0 - stamina_pct)
    return reward
