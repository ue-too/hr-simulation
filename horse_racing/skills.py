"""Composable jockey skill reward functions.

Each skill adds a small bonus/penalty (roughly 0.1-0.3 magnitude) that shapes
the agent's behavior when that skill flag is active in the observation vector.
Skills are composable: multiple can be active simultaneously, and their bonuses
sum. Archetype presets map to specific skill combinations.
"""

from __future__ import annotations

from horse_racing.types import SKILL_IDS

# ---------------------------------------------------------------------------
# Archetype → skill presets
# ---------------------------------------------------------------------------

ARCHETYPE_SKILL_PRESETS: dict[str, list[str]] = {
    "front_runner": ["pace_pressure", "sprint_timing"],
    "stalker": ["drafting_exploit", "sprint_timing"],
    "closer": ["stamina_management", "sprint_timing", "overtake"],
    "presser": ["pace_pressure", "stamina_management"],
}


# ---------------------------------------------------------------------------
# Individual skill bonus functions
# ---------------------------------------------------------------------------


def _stamina_management_bonus(
    obs_curr: dict,
    obs_prev: dict,
    placement: int,
    num_horses: int,
) -> float:
    """Efficient stamina budgeting vs linear baseline.

    Rewards conserving stamina above the linear drain curve and penalizes
    overspending. Active all race.
    """
    bonus = 0.0
    progress = obs_curr["track_progress"]
    stamina = obs_curr["stamina_ratio"]

    # Linear baseline: stamina should be at least (1 - progress)
    expected = 1.0 - progress
    margin = stamina - expected

    # Reward being above the linear baseline
    if margin > 0.0:
        bonus += 0.15 * min(margin, 0.3)

    # Overspend penalty is handled by the base reward (reward.py:162-163).
    # Only add the skill-unique signal: bonus for having reserves near finish.
    if progress > 0.85 and stamina > 0.2:
        bonus += 0.1 * min(stamina, 0.4)

    return bonus


def _sprint_timing_bonus(
    obs_curr: dict,
    obs_prev: dict,
    placement: int,
    num_horses: int,
) -> float:
    """Late-race speed bursts; penalize early sprinting.

    Rewards pushing above cruise after 75% progress, penalizes sprinting
    before 60%.
    """
    bonus = 0.0
    progress = obs_curr["track_progress"]
    stamina = obs_curr["stamina_ratio"]
    vel = obs_curr["tangential_vel"]
    cruise = obs_curr["effective_cruise_speed"]
    max_spd = obs_curr["effective_max_speed"]

    if max_spd <= cruise + 1e-6:
        return 0.0

    above_cruise_ratio = max((vel - cruise) / (max_spd - cruise), 0.0)

    # Penalize sprinting too early
    if progress < 0.6 and above_cruise_ratio > 0.3:
        bonus -= 0.05

    # Reward sprinting in final quarter
    if progress > 0.75:
        bonus += 0.25 * above_cruise_ratio * max(stamina, 0.1)

    # Extra bonus for acceleration after 80%
    if progress > 0.8:
        prev_vel = obs_prev.get("tangential_vel", vel)
        accel = vel - prev_vel
        if accel > 0:
            bonus += 0.1 * min(accel / 2.0, 1.0)

    return bonus


def _overtake_bonus(
    obs_curr: dict,
    obs_prev: dict,
    placement: int,
    num_horses: int,
    prev_placement: int | None = None,
) -> float:
    """Close on and pass opponents.

    Rewards gaining positions and closing tangential gaps. Penalizes
    losing positions.
    """
    bonus = 0.0

    # Position change reward/penalty
    if prev_placement is None:
        prev_placement = placement
    if placement < prev_placement:
        positions_gained = prev_placement - placement
        bonus += 0.2 * positions_gained
    elif placement > prev_placement:
        positions_lost = placement - prev_placement
        bonus -= 0.1 * positions_lost

    # Reward closing tangential gap to the horse ahead
    # relatives[0] is the nearest horse ahead (sorted by progress)
    relatives = obs_curr.get("relatives", [])
    if relatives and len(relatives) > 0:
        tang_gap = relatives[0][0]  # tangential offset to nearest
        if tang_gap > 0:  # horse ahead
            prev_relatives = obs_prev.get("relatives", [])
            if prev_relatives and len(prev_relatives) > 0:
                prev_tang_gap = prev_relatives[0][0]
                if prev_tang_gap > 0:
                    closing = prev_tang_gap - tang_gap
                    if closing > 0:
                        bonus += 0.1 * min(closing / 5.0, 1.0)

    return bonus


def _drafting_exploit_bonus(
    obs_curr: dict,
    obs_prev: dict,
    placement: int,
    num_horses: int,
) -> float:
    """Stay in draft position early, break away late.

    Rewards drafting modifier being active before 75%, rewards high speed
    after 75% (breaking away from the pack).
    """
    bonus = 0.0
    progress = obs_curr["track_progress"]
    active_mods = obs_curr.get("active_modifiers", set())

    if progress < 0.75:
        # Reward being in drafting position
        if "drafting" in active_mods:
            bonus += 0.15

        # Reward close following position (tang gap to horse ahead is small)
        relatives = obs_curr.get("relatives", [])
        if relatives and len(relatives) > 0:
            tang_gap = relatives[0][0]
            if 0 < tang_gap < 10.0:
                bonus += 0.1 * (1.0 - tang_gap / 10.0)
    else:
        # Late race: reward high speed (breaking away)
        vel = obs_curr["tangential_vel"]
        max_spd = obs_curr["effective_max_speed"]
        if max_spd > 1e-6:
            bonus += 0.2 * (vel / max_spd)

    return bonus


def _cornering_line_bonus(
    obs_curr: dict,
    obs_prev: dict,
    placement: int,
    num_horses: int,
) -> float:
    """Inside lines and high-speed cornering.

    Rewards taking the inside line on curves and maintaining speed through
    them. Only active on curves (curvature > 0).
    """
    curvature = obs_curr.get("curvature", 0.0)
    if curvature <= 0:
        return 0.0

    bonus = 0.0
    displacement = obs_curr.get("displacement", 0.0)
    vel = obs_curr["tangential_vel"]
    cruise = obs_curr["effective_cruise_speed"]

    # Reward inside displacement (negative = inside)
    if displacement < 0:
        bonus += 0.2 * min(abs(displacement), 5.0) / 5.0 * curvature * 10.0
    else:
        # Penalize being wide
        bonus -= 0.1 * min(displacement, 5.0) / 5.0 * curvature * 10.0

    # Reward maintaining speed through curves
    if cruise > 1e-6 and vel > cruise * 0.9:
        bonus += 0.1 * min(vel / cruise, 1.5) - 0.1

    # Clamp total
    return max(-0.15, min(bonus, 0.3))


def _pace_pressure_bonus(
    obs_curr: dict,
    obs_prev: dict,
    placement: int,
    num_horses: int,
) -> float:
    """Sustained high speed above cruise.

    Rewards consistently pushing above cruise speed. Penalizes coasting
    below cruise. Active all race.
    """
    bonus = 0.0
    vel = obs_curr["tangential_vel"]
    cruise = obs_curr["effective_cruise_speed"]
    max_spd = obs_curr["effective_max_speed"]
    stamina = obs_curr["stamina_ratio"]

    if max_spd <= cruise + 1e-6:
        return 0.0

    if vel > cruise:
        above_ratio = (vel - cruise) / (max_spd - cruise)
        bonus += 0.2 * above_ratio

        # Extra bonus when stamina is healthy (sustainable pressure)
        if stamina > 0.4:
            bonus += 0.1 * above_ratio
    else:
        # Penalize coasting below cruise
        if cruise > 1e-6:
            below_ratio = (cruise - vel) / cruise
            bonus -= 0.1 * min(below_ratio, 0.5)

    return bonus


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_SKILL_FN = {
    "stamina_management": _stamina_management_bonus,
    "sprint_timing": _sprint_timing_bonus,
    "overtake": _overtake_bonus,
    "drafting_exploit": _drafting_exploit_bonus,
    "cornering_line": _cornering_line_bonus,
    "pace_pressure": _pace_pressure_bonus,
}


def compute_skill_bonus(
    active_skills: set[str],
    obs_curr: dict,
    obs_prev: dict,
    placement: int,
    num_horses: int,
    prev_placement: int | None = None,
) -> float:
    """Sum bonuses for all active skills.

    Args:
        active_skills: Set of active skill IDs.
        obs_curr: Current observation dict.
        obs_prev: Previous observation dict.
        placement: Current 1-indexed placement.
        num_horses: Total horses in the race.
        prev_placement: Previous tick's placement (1-indexed). If None,
            defaults to current placement (no position change detected).

    Returns:
        Combined skill bonus (unclamped — individual skills self-limit).
    """
    total = 0.0
    for sid in active_skills:
        fn = _SKILL_FN.get(sid)
        if fn is not None:
            if fn is _overtake_bonus:
                total += fn(obs_curr, obs_prev, placement, num_horses, prev_placement)
            else:
                total += fn(obs_curr, obs_prev, placement, num_horses)
    return total
