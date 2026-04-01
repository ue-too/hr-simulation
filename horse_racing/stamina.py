"""Stamina depletion, recovery, and exhaustion logic."""

from __future__ import annotations

from dataclasses import dataclass, field

from horse_racing.attributes import CoreAttributes
from horse_racing.modifiers import ActiveModifier
from horse_racing.types import (
    CORNERING_DRAIN_RATE,
    GRIP_FORCE_BASELINE,
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
    current_speed: float,
    tangential_vel: float,
    turn_radius: float,
) -> float:
    """Update stamina based on current forces and return new stamina value."""
    drain = 0.0

    # Drain from jockey pushing forward
    if extra_tangential > 0:
        drain += abs(extra_tangential) * STAMINA_DRAIN_RATE

    # Drain from exceeding cruise speed
    if current_speed > eff.cruise_speed:
        drain += (current_speed - eff.cruise_speed) * OVERDRIVE_DRAIN_RATE

    # Drain from cornering beyond grip threshold
    if turn_radius < 1e6:
        required_force = tangential_vel**2 / turn_radius
        tolerated_force = eff.cornering_grip * GRIP_FORCE_BASELINE
        if required_force > tolerated_force:
            drain += (required_force - tolerated_force) * CORNERING_DRAIN_RATE

    # Drain proportional to speed — makes every meter traveled cost stamina.
    # Shorter paths (inside line) drain less total stamina over a race.
    drain += current_speed * SPEED_DRAIN_RATE

    # Recovery: always applies, but reduced when draining (prevents
    # binary on/off exploit where agent alternates push/coast ticks).
    if drain > 0:
        net = drain - eff.stamina_recovery * 0.25
        if net > 0:
            state.current_stamina = max(0, state.current_stamina - net)
        else:
            state.current_stamina = min(eff.stamina, state.current_stamina - net)
    else:
        state.current_stamina = min(eff.stamina, state.current_stamina + eff.stamina_recovery)

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
        stamina_recovery=eff.stamina_recovery,
        weight=eff.weight,
    )

    if ratio < 0.30:
        result.forward_accel *= ratio / 0.30

    if ratio < 0.20:
        lerp = ratio / 0.20
        result.max_speed = result.cruise_speed + (result.max_speed - result.cruise_speed) * lerp

    if ratio < 0.15:
        result.turn_accel *= 0.75

    return result
