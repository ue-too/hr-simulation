"""Reward function module — tuned for competitive racing with jockey archetypes.

Archetypes are derived from analysis of 79,447 HKJC race runs. Each archetype
shapes the jockey's racing strategy through reward bonuses, while the horse's
physical attributes (speed, stamina, weight) remain separate.

All per-tick reward components are normalized to roughly 0.1–1.0 range so that
shaping signals (stamina, collisions, lateral drift) are audible alongside the
dominant forward-progress reward.
"""

from __future__ import annotations

# Finish order bonuses: 1st place gets the most, 4th gets the least
FINISH_ORDER_BONUS = [50.0, 30.0, 15.0, 5.0]

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
    prev_placement: int | None = None,
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
    stamina = obs_curr["stamina_ratio"]
    vel = obs_curr["tangential_vel"]
    max_spd = obs_curr["effective_max_speed"]
    cruise_spd = obs_curr["effective_cruise_speed"]

    # ── Forward progress ─────────────────────────────────────────────
    # Accelerating reward: 1x at start → 3x near finish.
    # Scaled by path_efficiency so wider curve lines yield less reward.
    # ~0.15 per tick at start, ~0.45 per tick near finish at cruise speed.
    delta_progress = progress - obs_prev["track_progress"]
    efficiency = obs_curr.get("path_efficiency", 1.0)
    reward += 200.0 * delta_progress * (1.0 + 2.0 * progress) * efficiency

    # ── Speed bonus ──────────────────────────────────────────────────
    # Rewards going fast relative to max capability.
    # Scaled by stamina so exhausted horses aren't rewarded for redlining.
    # ~0.18 per tick at cruise, ~0.24 at max speed.
    if max_spd > 1e-6:
        reward += 0.3 * (vel / max_spd) * max(stamina, 0.3)

    # Rewards pushing beyond auto-cruise. Also scaled by stamina.
    # 0 at cruise, ~0.15 at max speed with full stamina.
    if max_spd > cruise_spd + 1e-6 and vel > cruise_spd:
        reward += 0.2 * (vel - cruise_spd) / (max_spd - cruise_spd) * max(stamina, 0.3)

    # ── Lateral drift penalty ────────────────────────────────────────
    # Penalize sideways velocity — real jockeys keep a clean line.
    # ~0.0 on straights, up to -0.15 when drifting hard on curves.
    normal_vel = obs_curr["normal_vel"]
    reward -= 0.05 * abs(normal_vel)

    # ── Alive penalty ────────────────────────────────────────────────
    # Increases with progress to encourage finishing quickly rather than
    # stalling near the end. Light early (agent still learning basics).
    # -0.0 at start, -0.1 near finish.
    reward -= 0.1 * progress

    # ── Near-finish bonus ────────────────────────────────────────────
    # Scaled by stamina so horses with reserves (realistic "kick") are
    # rewarded more than depleted ones. ~0.1-0.5 per tick in final 10%.
    if progress > 0.9:
        reward += 0.5 * max(stamina, 0.1)

    # ── Placement bonus ──────────────────────────────────────────────
    # Per-tick incentive to be ahead of others. ~0.0-0.3 per tick.
    if num_horses > 1:
        reward += 0.3 * (num_horses - placement) / (num_horses - 1)

    # ── Collision penalty ────────────────────────────────────────────
    # High enough that bumping doesn't pay off from placement gains.
    if collision_occurred:
        reward -= 1.0

    # ── Stamina management ───────────────────────────────────────────
    # Proportional penalty from 30% (matching exhaustion threshold).
    # 0 at 30%, -0.3 at 0%. Also reward good conservation above 50%.
    if stamina < 0.30:
        reward -= 0.3 * (1.0 - stamina / 0.30)
    elif stamina > 0.50:
        reward += 0.05

    # ── Finish order bonus ───────────────────────────────────────────
    # Large terminal reward for racing position (scaled down from 500
    # to match the reduced per-tick scale).
    if finish_order is not None:
        idx = min(finish_order - 1, len(FINISH_ORDER_BONUS) - 1)
        reward += FINISH_ORDER_BONUS[idx]

    # ── Overtaking bonus ─────────────────────────────────────────────
    # Rewards gaining positions. Scaled by stamina so reckless overtakes
    # that burn reserves are worth less.
    if prev_placement is not None and placement < prev_placement:
        positions_gained = prev_placement - placement
        reward += 0.8 * positions_gained * max(stamina, 0.3)

    # ── Archetype-specific reward shaping ────────────────────────────
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
