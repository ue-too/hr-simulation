"""Reward function module — tuned for competitive racing with jockey archetypes.

Archetypes are derived from analysis of 79,447 HKJC race runs. Each archetype
shapes the jockey's racing strategy through reward bonuses, while the horse's
physical attributes (speed, stamina, weight) remain separate.

All per-tick reward components are normalized to roughly 0.1–0.2 range so that
no single signal dominates. Placement, progress, stamina, and cornering are
all audible to the agent.
"""

from __future__ import annotations

from horse_racing.types import TRACK_HALF_WIDTH

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
    delta_progress = progress - obs_prev["track_progress"]
    raw_efficiency = obs_curr.get("path_efficiency", 1.0)
    # Amplify efficiency: inside line (eff > 1) boosted, outside (eff < 1) penalized more
    efficiency = 1.0 + 2.0 * (raw_efficiency - 1.0)
    reward += 200.0 * delta_progress * (1.0 + 2.0 * progress) * efficiency

    # ── Speed bonus ──────────────────────────────────────────────────
    # Rewards going fast relative to max capability.
    # Scaled by stamina so exhausted horses aren't rewarded for redlining.
    if max_spd > 1e-6:
        reward += 0.3 * (vel / max_spd) * max(stamina, 0.1)

    # Rewards pushing beyond auto-cruise. Also scaled by stamina.
    if max_spd > cruise_spd + 1e-6 and vel > cruise_spd:
        reward += 0.2 * (vel - cruise_spd) / (max_spd - cruise_spd) * max(stamina, 0.1)

    # ── Cornering line bonus ─────────────────────────────────────────
    # Reward taking the inside line on curves. Negative displacement
    # means the horse is inside its entry lane = shorter path.
    curvature = obs_curr.get("curvature", 0.0)
    if curvature > 0:
        displacement = obs_curr.get("displacement", 0.0)
        capped_inside = min(max(-displacement, 0.0), 3.0)
        reward += 3.0 * capped_inside * curvature
        # Penalty for being OUTSIDE on curves (positive displacement)
        if displacement > 0:
            reward -= 1.5 * min(displacement / TRACK_HALF_WIDTH, 1.0) * curvature * 60.0
        reward += 0.3 * efficiency

        # Reward actively steering inward when not already deep inside
        normal_vel = obs_curr.get("normal_vel", 0.0)
        if displacement > -1.5 and normal_vel < 0:
            reward += 0.5 * min(abs(normal_vel), 2.0)

    # ── Straight-segment stability ────────────────────────────────────
    # Penalize lateral velocity on straights to prevent drifting.
    # The horse should hold its line from the previous curve exit.
    if curvature <= 0:
        normal_vel = obs_curr.get("normal_vel", 0.0)
        reward -= 0.15 * min(abs(normal_vel), 2.0)

    # ── Alive penalty ────────────────────────────────────────────────
    # Strong time pressure so finishing faster outweighs accumulating
    # per-tick bonuses. Scales with progress: light early, heavy late.
    reward -= 0.2 * progress

    # ── Near-finish bonus ────────────────────────────────────────────
    # Scaled by stamina so horses with reserves (realistic "kick") are
    # rewarded more than depleted ones.
    if progress > 0.9:
        reward += 0.5 * max(stamina, 0.1)

    # ── Placement bonus ──────────────────────────────────────────────
    # Per-tick incentive to be ahead of others. Strong enough to make
    # overtaking (via steering) worthwhile.
    if num_horses > 1:
        reward += 0.4 * (num_horses - placement) / (num_horses - 1)

    # ── Collision penalty ────────────────────────────────────────────
    # High enough that bumping doesn't pay off from placement gains.
    if collision_occurred:
        reward -= 2.0

    # ── Stamina management ───────────────────────────────────────────
    # Proportional penalty from 40% (above exhaustion threshold to give
    # the agent time to react). Strong enough to overcome placement bonus.
    if stamina < 0.40:
        reward -= 0.4 * (1.0 - stamina / 0.40)
    elif stamina > 0.50:
        reward += 0.1 * stamina

    # ── Pacing bonus ─────────────────────────────────────────────────
    # Reward cruising efficiently in a healthy stamina zone.
    if 0.40 < stamina < 0.70 and abs(vel - cruise_spd) < 1.0:
        reward += 0.08

    # ── Finish order bonus ───────────────────────────────────────────
    # Large terminal reward for racing position.
    if finish_order is not None:
        idx = min(finish_order - 1, len(FINISH_ORDER_BONUS) - 1)
        reward += FINISH_ORDER_BONUS[idx]

    # ── Overtaking bonus ─────────────────────────────────────────────
    # Strong reward for gaining positions — the main incentive for
    # lateral movement and strategic racing.
    if prev_placement is not None and placement < prev_placement:
        positions_gained = prev_placement - placement
        reward += 1.5 * positions_gained * max(stamina, 0.3)

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

    # Late race (75-100%): bonus for holding on, but penalize dying lead
    else:
        if placement == 1:
            bonus += 0.1
        # Penalty for leading while nearly exhausted — unrealistic
        if placement == 1 and obs["stamina_ratio"] < 0.2:
            bonus -= 0.15

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
    stamina = obs["stamina_ratio"]

    # Early race (0-25%): bonus for 2nd-3rd position
    if progress < 0.25:
        if 2 <= placement <= 3:
            bonus += 0.25
        elif placement == 1:
            bonus -= 0.1  # penalty for leading too early

    # Mid race (25-75%): bonus for staying near front with good stamina
    elif progress < 0.75:
        if placement <= 3:
            bonus += 0.15

        # Bonus for conserving stamina (staying near cruise, not over-pushing)
        # Only if stamina is healthy — otherwise the agent already made a mistake
        vel = obs["tangential_vel"]
        cruise = obs["effective_cruise_speed"]
        max_spd = obs["effective_max_speed"]
        target_ceil = cruise + 0.3 * (max_spd - cruise)
        if cruise <= vel <= target_ceil and stamina > 0.50:
            bonus += 0.15

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
    stamina = obs_curr["stamina_ratio"]

    # Early race (0-50%): bonus for conserving stamina, no penalty for being back
    if progress < 0.5:
        if stamina > 0.8:
            bonus += 0.15
        if placement >= num_horses - 1:
            bonus += 0.05

    # Mid race (50-75%): gradually build speed while maintaining reserves
    elif progress < 0.75:
        vel = obs_curr["tangential_vel"]
        cruise = obs_curr["effective_cruise_speed"]
        max_spd = obs_curr["effective_max_speed"]

        # Reward building speed above cruise, scaled by how deep into mid-race
        ramp = (progress - 0.5) / 0.25  # 0 at 50%, 1 at 75%
        if vel > cruise and max_spd > cruise + 1e-6:
            bonus += 0.15 * ramp * (vel - cruise) / (max_spd - cruise)

        # Reward maintaining stamina reserves for the kick
        if stamina > 0.50:
            bonus += 0.1

    # Late race (75-100%): big bonus for gaining positions
    else:
        prev_progress = obs_prev["track_progress"]
        if prev_progress > 0.7:
            vel = obs_curr["tangential_vel"]
            max_spd = obs_curr["effective_max_speed"]
            if max_spd > 1e-6:
                bonus += 0.3 * (vel / max_spd)

        if placement <= 2:
            bonus += 0.5
        elif placement <= 3:
            bonus += 0.2

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
    stamina = obs["stamina_ratio"]

    vel = obs["tangential_vel"]
    cruise = obs["effective_cruise_speed"]
    max_spd = obs["effective_max_speed"]

    # Reward consistently high speed throughout, scaled by stamina
    if vel > cruise and max_spd > cruise + 1e-6:
        above_cruise_ratio = (vel - cruise) / (max_spd - cruise)
        bonus += 0.2 * above_cruise_ratio * max(stamina, 0.2)

    # Early race (0-50%): bonus for pushing hard from any position
    if progress < 0.5:
        if vel > cruise * 1.1:
            bonus += 0.15

    # Mid race (50-75%): bonus for maintaining pace pressure
    elif progress < 0.75:
        if vel > cruise:
            bonus += 0.1
        if placement <= num_horses // 2:
            bonus += 0.1

    # Late race (75-100%): bonus for still having gas
    else:
        if stamina > 0.2 and vel > cruise:
            bonus += 0.2

    return bonus
