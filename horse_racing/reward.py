"""Reward function — delta-progress + finish bonus."""

_FINISH_BONUS = {1: 10.0, 2: 5.0, 3: 2.0}


def compute_reward(
    prev_progress: float,
    curr_progress: float,
    finish_order: int | None,
) -> float:
    """Compute reward for one step.

    Args:
        prev_progress: Track progress at previous tick [0, 1].
        curr_progress: Track progress at current tick [0, 1].
        finish_order: Finishing position (1-based) if horse finished, else None.
    """
    reward = curr_progress - prev_progress
    if finish_order is not None:
        reward += _FINISH_BONUS.get(finish_order, 0.0)
    return reward
