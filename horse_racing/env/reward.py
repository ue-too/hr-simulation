"""Reward function — minimal, outcome-focused with light jockey style nudges.

5 core components:
  1. placement_reward: per-tick incentive to be ahead
  2. finish_bonus: big one-time reward for top 4 finishers
  3. progress_nudge: small forward incentive (prevents stalling)
  4. alive_penalty: time pressure
  5. collision_penalty: discourage bumping
"""

from __future__ import annotations

from horse_racing.core.stamina import StaminaState
from horse_racing.core.types import JockeyStyle

# Finish bonuses by placement (1st through 4th)
FINISH_ORDER_BONUS: list[float] = [50.0, 30.0, 15.0, 5.0]


def compute_reward(
    placement: int,
    num_horses: int,
    progress: float,
    prev_progress: float,
    collided: bool,
    finished: bool,
    jockey_style: JockeyStyle,
    stamina: StaminaState,
    leading_at_25: bool = False,
    leading_at_50: bool = False,
    positions_gained_final_25: int = 0,
) -> float:
    """Compute reward for one tick.

    Args:
        placement: Current race position (0-indexed).
        num_horses: Total horses in race.
        progress: Current track progress [0, 1].
        prev_progress: Previous tick's progress.
        collided: Whether horse collided this tick.
        finished: Whether horse just finished the race.
        jockey_style: Jockey style parameters.
        stamina: Current stamina state.
        leading_at_25: Whether horse was leading at 25% progress.
        leading_at_50: Whether horse was leading at 50% progress.
        positions_gained_final_25: Positions gained since 75% progress.
    """
    reward = 0.0

    # 1. Placement reward: being ahead is good
    if num_horses > 1:
        reward += 0.5 * (num_horses - 1 - placement) / (num_horses - 1)

    # 2. Finish bonus
    if finished and placement < len(FINISH_ORDER_BONUS):
        reward += FINISH_ORDER_BONUS[placement]

    # 3. Progress nudge — main learning signal, must dominate alive penalty
    delta_progress = progress - prev_progress
    reward += 200.0 * delta_progress

    # 4. Alive penalty (light time pressure — must be << progress reward)
    reward -= 0.01

    # 5. Collision penalty
    if collided:
        reward -= 1.0

    # --- Jockey style nudges (light, overridable) ---

    # Front-runner nudge: bonus for leading at checkpoints
    if jockey_style.tactical_bias < -0.3:  # front-runner biased
        if leading_at_25:
            reward += 0.1
        if leading_at_50:
            reward += 0.1

    # Closer nudge: bonus for gaining positions in final stretch
    if jockey_style.tactical_bias > 0.3:  # closer biased
        if progress > 0.75 and positions_gained_final_25 > 0:
            reward += 0.1 * positions_gained_final_25

    # Risk tolerance: conservative jockeys penalized for depleting stamina early
    if jockey_style.risk_tolerance < 0.3:  # conservative
        if stamina.ratio < 0.2 and progress < 0.8:
            reward -= 0.05

    return reward
