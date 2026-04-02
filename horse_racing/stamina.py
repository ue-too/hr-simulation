"""Stamina depletion and exhaustion logic.

Fixed pool with no recovery — stamina only decreases over a race.
Drain accounts for both forward and lateral movement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from horse_racing.attributes import CoreAttributes, TRAIT_RANGES
from horse_racing.modifiers import ActiveModifier
from horse_racing.types import (
    CORNERING_DRAIN_RATE,
    GRIP_FORCE_BASELINE,
    LATERAL_STEERING_DRAIN_RATE,
    LATERAL_VELOCITY_DRAIN_RATE,
    OVERDRIVE_DRAIN_RATE,
    SPEED_DRAIN_RATE,
    STAMINA_DRAIN_RATE,
)


@dataclass
class HorseRuntimeState:
    current_stamina: float = 100.0
    base_attributes: CoreAttributes = field(default_factory=CoreAttributes)
    active_modifiers: list[ActiveModifier] = field(default_factory=list)


def update_stamina(
    state: HorseRuntimeState,
    eff: CoreAttributes,
    extra_tangential: float,
    extra_normal: float,
    current_speed: float,
    tangential_vel: float,
    normal_vel: float,
    turn_radius: float,
) -> float:
    """Update stamina based on current forces and return new stamina value.

    Fixed pool — no recovery. All drain is multiplied by the horse's
    drain_rate_mult attribute (lower = more efficient).
    """
    drain = 0.0

    # Drain from jockey pushing forward
    if extra_tangential > 0:
        drain += abs(extra_tangential) * STAMINA_DRAIN_RATE

    # Drain from jockey steering laterally
    if abs(extra_normal) > 0:
        drain += abs(extra_normal) * LATERAL_STEERING_DRAIN_RATE

    # Drain from exceeding cruise speed
    if current_speed > eff.cruise_speed:
        drain += (current_speed - eff.cruise_speed) * OVERDRIVE_DRAIN_RATE

    # Drain from cornering beyond grip threshold
    if turn_radius < 1e6:
        required_force = tangential_vel**2 / turn_radius
        tolerated_force = eff.cornering_grip * GRIP_FORCE_BASELINE
        if required_force > tolerated_force:
            drain += (required_force - tolerated_force) * CORNERING_DRAIN_RATE

    # Distance tax — every meter traveled costs stamina
    drain += current_speed * SPEED_DRAIN_RATE

    # Lateral velocity tax — sustained sideways movement costs stamina
    drain += abs(normal_vel) * LATERAL_VELOCITY_DRAIN_RATE

    # Apply per-horse drain efficiency
    drain *= eff.drain_rate_mult

    state.current_stamina = max(0.0, state.current_stamina - drain)
    return state.current_stamina


def apply_exhaustion(
    eff: CoreAttributes,
    current_stamina: float,
    max_stamina: float,
) -> CoreAttributes:
    """Degrade effective attributes when stamina is low."""
    if max_stamina < 1e-6:
        return eff

    ratio = current_stamina / max_stamina
    result = CoreAttributes(
        cruise_speed=eff.cruise_speed,
        max_speed=eff.max_speed,
        forward_accel=eff.forward_accel,
        turn_accel=eff.turn_accel,
        cornering_grip=eff.cornering_grip,
        stamina=eff.stamina,
        drain_rate_mult=eff.drain_rate_mult,
        weight=eff.weight,
    )

    if ratio < 0.30:
        result.forward_accel = max(
            TRAIT_RANGES["forward_accel"][0],
            result.forward_accel * (ratio / 0.30),
        )

    if ratio < 0.20:
        lerp = ratio / 0.20
        result.max_speed = result.cruise_speed + (result.max_speed - result.cruise_speed) * lerp
        result.max_speed = max(TRAIT_RANGES["max_speed"][0], result.max_speed)

    if ratio < 0.25:
        result.turn_accel = max(
            TRAIT_RANGES["turn_accel"][0],
            result.turn_accel * (0.5 + 0.5 * (ratio / 0.25)),
        )

    return result
