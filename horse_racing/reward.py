"""Reward function module — tuned for competitive racing with jockey archetypes.

Archetypes are derived from analysis of 79,447 HKJC race runs. Each archetype
shapes the jockey's racing strategy through reward bonuses, while the horse's
physical attributes (speed, stamina, weight) remain separate.
"""

from __future__ import annotations

# Finish order bonuses: 1st place gets the most, 4th gets the least
FINISH_ORDER_BONUS = [500.0, 300.0, 150.0, 50.0]

# Archetype names
ARCHETYPES = ["front_runner", "stalker", "closer", "presser"]


def compute_reward(
    obs_prev: dict,
    obs_curr: dict,
    collision_occurred: bool,
    placement: int = 1,
    num_horses: int = 4,
    finish_order: int | None = None,
    archetype: str | None = None,
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
        archetype: Jockey racing style. One of "front_runner", "stalker",
            "closer", "presser", or None for no archetype shaping.

    Returns:
        Scalar reward.
    """
    reward = 0.0
    progress = obs_curr["track_progress"]

    # Forward progress — accelerating reward: per-meter reward increases
    # as horse approaches finish (1x at start → 3x near finish)
    delta_progress = progress - obs_prev["track_progress"]
    reward += 1000.0 * delta_progress * (1.0 + 2.0 * progress)

    # Speed bonus — rewards going fast relative to max capability
    max_spd = obs_curr["effective_max_speed"]
    if max_spd > 1e-6:
        reward += 0.3 * (obs_curr["tangential_vel"] / max_spd)

    # Speed above cruise bonus — rewards pushing beyond auto-cruise
    cruise_spd = obs_curr["effective_cruise_speed"]
    vel = obs_curr["tangential_vel"]
    if max_spd > cruise_spd + 1e-6 and vel > cruise_spd:
        reward += 0.2 * (vel - cruise_spd) / (max_spd - cruise_spd)

    # Alive penalty — decreasing as horse progresses
    reward -= 0.05 * (1.0 - progress)

    # Near-finish bonus — extra incentive in the final stretch
    if progress > 0.9:
        reward += 0.5

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

    # Archetype-specific reward shaping
    if archetype:
        reward += _archetype_bonus(
            archetype, obs_prev, obs_curr, placement, num_horses, progress,
        )

    return reward


# ---------------------------------------------------------------------------
# Archetype reward bonuses — derived from HKJC pace analysis
# ---------------------------------------------------------------------------

def _archetype_bonus(
    archetype: str,
    obs_prev: dict,
    obs_curr: dict,
    placement: int,
    num_horses: int,
    progress: float,
) -> float:
    """Compute archetype-specific reward bonus.

    Each archetype rewards different positioning and speed patterns at
    different stages of the race, calibrated from real HKJC data.
    """
    if archetype == "front_runner":
        return _front_runner_bonus(obs_curr, placement, num_horses, progress)
    elif archetype == "stalker":
        return _stalker_bonus(obs_curr, placement, num_horses, progress)
    elif archetype == "closer":
        return _closer_bonus(obs_prev, obs_curr, placement, num_horses, progress)
    elif archetype == "presser":
        return _presser_bonus(obs_curr, placement, num_horses, progress)
    return 0.0


def _front_runner_bonus(
    obs: dict, placement: int, num_horses: int, progress: float,
) -> float:
    """Front-runner: fast start, lead throughout.

    From HKJC data: speed 1.23x early, fading to 0.83x late.
    Position: leads (0.25) throughout, drifts to 0.34 by end.
    Win rate: 12.4%.
    """
    bonus = 0.0

    # Early race (0-50%): strong bonus for leading
    if progress < 0.5:
        if placement == 1:
            bonus += 0.3
        elif placement == 2:
            bonus += 0.1

        # Bonus for pushing above cruise early
        vel = obs["tangential_vel"]
        cruise = obs["effective_cruise_speed"]
        if vel > cruise:
            bonus += 0.15

    # Mid race (50-75%): bonus for maintaining lead
    elif progress < 0.75:
        if placement <= 2:
            bonus += 0.2

    # Late race (75-100%): no penalty for fading — front-runners naturally slow
    else:
        if placement == 1:
            bonus += 0.1

    return bonus


def _stalker_bonus(
    obs: dict, placement: int, num_horses: int, progress: float,
) -> float:
    """Stalker: steady pace, sits near front, moves to lead late.

    From HKJC data: speed ~1.0x throughout (most consistent).
    Position: near front (0.24) throughout, slight drift.
    Win rate: 13.0% (highest).
    """
    bonus = 0.0

    # Early race (0-25%): bonus for 2nd-3rd position
    if progress < 0.25:
        if 2 <= placement <= 3:
            bonus += 0.25
        elif placement == 1:
            bonus -= 0.1  # penalty for leading too early

    # Mid race (25-75%): bonus for staying near front
    elif progress < 0.75:
        if placement <= 3:
            bonus += 0.15

        # Bonus for conserving stamina (staying near cruise, not over-pushing)
        vel = obs["tangential_vel"]
        cruise = obs["effective_cruise_speed"]
        max_spd = obs["effective_max_speed"]
        # Reward being in the efficient speed zone (cruise to cruise+30% of gap)
        target_ceil = cruise + 0.3 * (max_spd - cruise)
        if cruise <= vel <= target_ceil:
            bonus += 0.1

    # Late race (75-100%): big bonus for taking the lead
    else:
        if placement == 1:
            bonus += 0.4
        elif placement == 2:
            bonus += 0.15

    return bonus


def _closer_bonus(
    obs_prev: dict, obs_curr: dict, placement: int, num_horses: int, progress: float,
) -> float:
    """Closer: starts at back, conserves energy, makes big late move.

    From HKJC data: speed 0.96x early, builds to 1.03x late.
    Position: starts at 0.76, moves to 0.66 by end.
    Win rate: 3.2% (low, but spectacular when it works).
    """
    bonus = 0.0

    # Early race (0-50%): bonus for conserving stamina, no penalty for being back
    if progress < 0.5:
        # Reward high stamina conservation
        if obs_curr["stamina_ratio"] > 0.8:
            bonus += 0.15
        # No penalty for being at the back
        if placement >= num_horses - 1:
            bonus += 0.05

    # Late race (75-100%): big bonus for gaining positions
    elif progress > 0.75:
        # Bonus for each position gained this step
        prev_progress = obs_prev["track_progress"]
        if prev_progress > 0.7:  # only count gains in late race
            # Higher speed = more reward (the "kick")
            vel = obs_curr["tangential_vel"]
            max_spd = obs_curr["effective_max_speed"]
            if max_spd > 1e-6:
                bonus += 0.3 * (vel / max_spd)

        # Big bonus for being in the top 2 late
        if placement <= 2:
            bonus += 0.5
        elif placement <= 3:
            bonus += 0.2

    # Mid race (50-75%): start building speed
    else:
        vel = obs_curr["tangential_vel"]
        cruise = obs_curr["effective_cruise_speed"]
        if vel > cruise:
            bonus += 0.1

    return bonus


def _presser_bonus(
    obs: dict, placement: int, num_horses: int, progress: float,
) -> float:
    """Presser: pushes pace consistently, wears down the field.

    From HKJC data: speed 1.21x early, fading to 0.85x late.
    Position: starts mid-back (0.76), pushes through pack.
    Win rate: 3.2% (exhausting strategy, sometimes pays off).
    """
    bonus = 0.0

    # Reward consistently high speed throughout
    vel = obs["tangential_vel"]
    cruise = obs["effective_cruise_speed"]
    max_spd = obs["effective_max_speed"]

    if vel > cruise and max_spd > cruise + 1e-6:
        # The presser always wants to be above cruise
        above_cruise_ratio = (vel - cruise) / (max_spd - cruise)
        bonus += 0.2 * above_cruise_ratio

    # Early race (0-50%): bonus for pushing hard from any position
    if progress < 0.5:
        if vel > cruise * 1.1:  # 10% above cruise
            bonus += 0.15

    # Mid race (50-75%): bonus for maintaining pace pressure
    elif progress < 0.75:
        if vel > cruise:
            bonus += 0.1
        # Bonus for moving up through the field
        if placement <= num_horses // 2:
            bonus += 0.1

    # Late race (75-100%): bonus for still having gas
    else:
        if obs["stamina_ratio"] > 0.2 and vel > cruise:
            bonus += 0.2  # still pushing with stamina left = strong finish

    return bonus
