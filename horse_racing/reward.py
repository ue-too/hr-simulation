"""Reward function — progress + pacing + speed band + positioning + smoothness + finish."""

from __future__ import annotations

_FINISH_BONUS = {1: 10.0, 2: 5.0, 3: 2.0}
FINISHING_SPEED_BONUS = 5.0
DEPLETION_PENALTY = 15.0
DEPLETION_THRESHOLD = 0.6  # speed_ratio below this triggers penalty
OVERTAKE_BONUS = 0.5
RAIL_COLLISION_PENALTY = 0.002
PACING_BONUS = 0.002  # per-step reward for stamina above ideal curve
BOXING_PENALTY = 0.004  # per-step penalty for being stuck behind a slower horse

# Speed efficiency: reward cruising in first 70%, free to sprint in final 30%
SPEED_BAND_BONUS = 0.003
SPEED_BAND_LOW = 0.65   # of cruise speed
SPEED_BAND_HIGH = 0.80  # of cruise speed
SPEED_BAND_PHASE = 0.70  # reward only applies before this progress

# Tactical positioning: stalk early, kick late
POSITION_BONUS = 0.002
POSITION_STALK_PHASE = 0.50   # reward pack position before this
POSITION_KICK_PHASE = 0.75    # reward leading position after this

# Smoothness: penalize rapid action oscillation
SMOOTHNESS_PENALTY = 0.001


def _ideal_stamina(progress: float) -> float:
    """Ideal stamina fraction at a given progress.

    Linear ramp: 1.0 at start → 0.05 at finish.
    """
    return max(0.0, 1.0 - 0.95 * progress)


def compute_reward(
    prev_progress: float,
    curr_progress: float,
    finish_order: int | None,
    finishing_speed: float = 0.0,
    cruise_speed: float = 13.0,
    overtakes: int = 0,
    rail_contact: bool = False,
    stamina_frac: float = 1.0,
    speed_ratio: float = 0.0,
    rank: int = 1,
    horse_count: int = 4,
    prev_tang: float = 0.0,
    curr_tang: float = 0.0,
    prev_norm: float = 0.0,
    curr_norm: float = 0.0,
    boxed: bool = False,
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
        stamina_frac: Current stamina as fraction of max [0, 1].
        speed_ratio: Current speed / cruise speed.
        rank: Current placement (1 = leading).
        horse_count: Total horses in race.
        prev_tang: Previous step's tangential action.
        curr_tang: Current step's tangential action.
        prev_norm: Previous step's normal action.
        curr_norm: Current step's normal action.
    """
    reward = curr_progress - prev_progress
    reward += OVERTAKE_BONUS * overtakes
    if rail_contact:
        reward -= RAIL_COLLISION_PENALTY
    if boxed:
        reward -= BOXING_PENALTY

    # --- Per-step pacing: stamina above ideal curve ---
    ideal = _ideal_stamina(curr_progress)
    if stamina_frac >= ideal:
        reward += PACING_BONUS
    else:
        reward -= PACING_BONUS * (ideal - stamina_frac) / max(ideal, 0.01)

    # --- Speed efficiency band (first 70% only) ---
    if curr_progress < SPEED_BAND_PHASE:
        if SPEED_BAND_LOW <= speed_ratio <= SPEED_BAND_HIGH:
            reward += SPEED_BAND_BONUS
        elif speed_ratio > SPEED_BAND_HIGH:
            overshoot = speed_ratio - SPEED_BAND_HIGH
            reward -= SPEED_BAND_BONUS * min(overshoot / 0.2, 1.0)

    # --- Tactical positioning: stalk early, kick late ---
    if curr_progress < POSITION_STALK_PHASE:
        # Early race: reward being in the pack (2nd-4th), penalize leading
        if 2 <= rank <= min(4, horse_count):
            reward += POSITION_BONUS
        elif rank == 1:
            reward -= POSITION_BONUS * 0.5  # mild penalty for leading early
    elif curr_progress >= POSITION_KICK_PHASE:
        # Final stretch: reward leading positions
        if rank <= 2:
            reward += POSITION_BONUS

    # --- Smoothness: penalize action oscillation ---
    tang_flip = (curr_tang - prev_tang)
    # Penalize large tangential changes (e.g., full push to brake)
    if abs(tang_flip) > 0.5:
        reward -= SMOOTHNESS_PENALTY * (abs(tang_flip) - 0.5)
    # Lighter penalty for lateral flips
    norm_flip = abs(curr_norm - prev_norm)
    if norm_flip > 1.0:
        reward -= SMOOTHNESS_PENALTY * 0.5 * (norm_flip - 1.0)

    # --- Finish bonuses ---
    if finish_order is not None:
        reward += _FINISH_BONUS.get(finish_order, 0.0)
        fin_speed_ratio = finishing_speed / cruise_speed if cruise_speed > 0 else 0.0
        reward += FINISHING_SPEED_BONUS * fin_speed_ratio
        if fin_speed_ratio < DEPLETION_THRESHOLD:
            reward -= DEPLETION_PENALTY * (DEPLETION_THRESHOLD - fin_speed_ratio)
    return reward
