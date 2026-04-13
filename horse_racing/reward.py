"""Reward function — delta-progress + finish bonus + time bonus + exhaustion penalty."""

from __future__ import annotations

_FINISH_BONUS = {1: 10.0, 2: 5.0, 3: 2.0}
EXHAUSTION_PENALTY = -0.002
PAR_TICKS = 2200          # ~cruise finish time; finishing faster earns a bonus
TIME_BONUS_MAX = 10.0     # bonus at tick 0; linearly decays to 0 at PAR_TICKS


def compute_reward(
    prev_progress: float,
    curr_progress: float,
    finish_order: int | None,
    current_stamina: float = 1.0,
    finish_tick: int | None = None,
) -> float:
    """Compute reward for one step.

    Args:
        prev_progress: Track progress at previous tick [0, 1].
        curr_progress: Track progress at current tick [0, 1].
        finish_order: Finishing position (1-based) if horse finished, else None.
        current_stamina: Horse's current stamina (0 = exhausted).
        finish_tick: Tick number when the horse finished (for time bonus).
    """
    reward = curr_progress - prev_progress
    if current_stamina <= 0:
        reward += EXHAUSTION_PENALTY
    if finish_order is not None:
        reward += _FINISH_BONUS.get(finish_order, 0.0)
    if finish_tick is not None:
        reward += TIME_BONUS_MAX * max(0.0, 1.0 - finish_tick / PAR_TICKS)
    return reward
