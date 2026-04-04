"""Stamina system — drain-only energy pool with speed-efficiency curve.

The key mechanic: stamina drain scales quadratically with speed above the
horse's efficiency_speed. This creates natural pacing trade-offs — pushing
hard costs disproportionately more energy.
"""

from __future__ import annotations

from dataclasses import dataclass

from horse_racing.core.horse import HorseProfile

# Base drain constants
BASE_DRAIN_RATE: float = 0.001  # constant drain per tick (distance tax)
EXCESS_DRAIN_RATE: float = 0.003  # quadratic drain per (m/s above efficiency)^2
LATERAL_DRAIN_RATE: float = 0.0005  # drain from lateral movement
CORNERING_EXCESS_DRAIN: float = 0.001  # drain from exceeding cornering grip

# Exhaustion thresholds
FATIGUE_THRESHOLD: float = 0.3  # below this, horse starts degrading
CRITICAL_THRESHOLD: float = 0.15  # below this, severe performance loss


@dataclass
class StaminaState:
    """Mutable stamina state for a single horse."""

    current: float
    maximum: float

    @property
    def ratio(self) -> float:
        return max(0.0, self.current / self.maximum) if self.maximum > 0 else 0.0

    @property
    def is_fatigued(self) -> bool:
        return self.ratio < FATIGUE_THRESHOLD

    @property
    def is_critical(self) -> bool:
        return self.ratio < CRITICAL_THRESHOLD

    def drain(self, amount: float) -> None:
        self.current = max(0.0, self.current - amount)


def create_stamina(profile: HorseProfile) -> StaminaState:
    return StaminaState(current=profile.stamina_pool, maximum=profile.stamina_pool)


def compute_drain(
    profile: HorseProfile,
    current_speed: float,
    lateral_speed: float,
    cornering_excess: float,
    dt: float,
) -> float:
    """Compute stamina drain for one physics tick.

    Args:
        profile: Horse physical attributes.
        current_speed: Current forward speed (m/s).
        lateral_speed: Absolute lateral velocity (m/s).
        cornering_excess: How much centripetal force exceeds grip (>=0).
        dt: Time step (seconds).

    Returns:
        Stamina to drain this tick.
    """
    efficiency_speed = profile.efficiency_speed
    inv_efficiency = 1.0 / profile.stamina_efficiency

    # Base drain: always present when moving
    drain = BASE_DRAIN_RATE * current_speed * inv_efficiency

    # Quadratic excess drain: pushing beyond comfortable speed
    excess = max(0.0, current_speed - efficiency_speed)
    drain += EXCESS_DRAIN_RATE * excess * excess * inv_efficiency

    # Lateral movement drain
    drain += LATERAL_DRAIN_RATE * abs(lateral_speed)

    # Cornering excess drain
    if cornering_excess > 0:
        drain += CORNERING_EXCESS_DRAIN * cornering_excess

    return drain * dt


def apply_fatigue(
    stamina: StaminaState,
    base_forward_force: float,
    base_lateral_response: float,
) -> tuple[float, float]:
    """Scale forces based on fatigue level.

    Returns:
        (scaled_forward_force, scaled_lateral_response)
    """
    ratio = stamina.ratio

    if ratio >= FATIGUE_THRESHOLD:
        return base_forward_force, base_lateral_response

    if ratio >= CRITICAL_THRESHOLD:
        # Linear degradation between critical and fatigue thresholds
        t = (ratio - CRITICAL_THRESHOLD) / (FATIGUE_THRESHOLD - CRITICAL_THRESHOLD)
        forward_scale = 0.4 + 0.6 * t  # drops to 40% at critical
        lateral_scale = 0.5 + 0.5 * t  # drops to 50% at critical
    else:
        # Below critical: severe degradation
        t = ratio / CRITICAL_THRESHOLD
        forward_scale = 0.2 + 0.2 * t  # drops to 20% at zero
        lateral_scale = 0.3 + 0.2 * t  # drops to 30% at zero

    return base_forward_force * forward_scale, base_lateral_response * lateral_scale
