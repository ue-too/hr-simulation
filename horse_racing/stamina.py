"""Stamina depletion and exhaustion logic — physics redesign.

Two pools per horse:
  * aerobic (state.current_stamina)  — main fuel, slowly drained by speed
  * burst   (state.burst_pool)        — small reserve for late kicks

The engine populates ``state.is_frontmost`` and ``state.is_drafting`` each
tick before calling :func:`update_stamina`. Lead penalty and draft recovery
both depend on those flags.

See horse_racing/types.py for constant definitions and the high-level
description of the redesigned mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from horse_racing.attributes import CoreAttributes, TRAIT_RANGES
from horse_racing.modifiers import ActiveModifier
from horse_racing.types import (
    BURST_DRAIN_K,
    BURST_EMPTY_CLAMP,
    BURST_K,
    BURST_RECOVERY_K,
    CLIFF_ACCEL_MULT,
    CLIFF_CRUISE_MULT,
    CLIFF_THRESHOLD,
    CORNERING_DRAIN_RATE,
    DRAFT_RECOVERY_CRUISE_BUFFER,
    DRAFT_RECOVERY_K,
    GRIP_FORCE_BASELINE,
    LATERAL_STEERING_DRAIN_RATE,
    LATERAL_VELOCITY_DRAIN_RATE,
    LEAD_K,
    LEAD_STAMINA_EXP,
    LEAD_STAMINA_REF,
    SPEED_DRAIN_RATE,
    STAMINA_DRAIN_RATE,
)


@dataclass
class HorseRuntimeState:
    current_stamina: float = 100.0
    base_attributes: CoreAttributes = field(default_factory=CoreAttributes)
    active_modifiers: list[ActiveModifier] = field(default_factory=list)

    # Burst pool — kick reserve, separate from aerobic stamina.
    burst_pool: float = 0.0
    burst_max: float = 0.0

    # Race-context flags written by the engine each tick before update_stamina.
    is_frontmost: bool = False
    is_drafting: bool = False


def compute_burst_max(attrs: CoreAttributes) -> float:
    """Burst pool size from a horse's attributes (band × stamina × BURST_K)."""
    band = max(0.1, attrs.max_speed - attrs.cruise_speed)
    return BURST_K * band * (attrs.stamina / 100.0)


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
    """Update aerobic + burst stamina. Returns new aerobic value.

    The engine must populate ``state.is_frontmost`` and ``state.is_drafting``
    before calling. ``state.burst_pool`` and ``state.burst_max`` must be
    initialized (engine does this in reset).
    """
    drain = 0.0

    # Push cost — every unit of forward push costs aerobic, even when speed
    # is already capped at max. This makes "mash the pedal" strictly worse
    # than "push just enough to reach max."
    if extra_tangential > 0:
        drain += extra_tangential * STAMINA_DRAIN_RATE

    # Lateral steering input
    if abs(extra_normal) > 0:
        drain += abs(extra_normal) * LATERAL_STEERING_DRAIN_RATE

    # Cornering beyond grip
    if turn_radius < 1e6:
        required_force = tangential_vel * tangential_vel / turn_radius
        tolerated_force = eff.cornering_grip * GRIP_FORCE_BASELINE
        if required_force > tolerated_force:
            drain += (required_force - tolerated_force) * CORNERING_DRAIN_RATE

    # Distance tax — every meter traveled costs aerobic
    drain += current_speed * SPEED_DRAIN_RATE

    # Lateral drift tax
    drain += abs(normal_vel) * LATERAL_VELOCITY_DRAIN_RATE

    # Per-horse efficiency
    drain *= eff.drain_rate_mult

    # Lead penalty — non-linearly scaled by stamina so stayers get cubic relief
    if state.is_frontmost and current_speed > eff.cruise_speed:
        excess = current_speed - eff.cruise_speed
        stam_factor = (LEAD_STAMINA_REF / max(1.0, eff.stamina)) ** LEAD_STAMINA_EXP
        drain += LEAD_K * excess * stam_factor

    state.current_stamina = max(0.0, state.current_stamina - drain)

    # Draft recovery — only when at or near cruise (not while pushing)
    if state.is_drafting and current_speed <= eff.cruise_speed + DRAFT_RECOVERY_CRUISE_BUFFER:
        recovery = DRAFT_RECOVERY_K / max(0.5, eff.drain_rate_mult)
        max_aerobic = state.base_attributes.stamina if state.base_attributes else 100.0
        state.current_stamina = min(max_aerobic, state.current_stamina + recovery)

    # Burst pool dynamics
    if current_speed > eff.cruise_speed:
        excess = current_speed - eff.cruise_speed
        state.burst_pool = max(0.0, state.burst_pool - excess * excess * BURST_DRAIN_K)
    elif current_speed < eff.cruise_speed - DRAFT_RECOVERY_CRUISE_BUFFER:
        state.burst_pool = min(state.burst_max, state.burst_pool + BURST_RECOVERY_K)

    return state.current_stamina


def apply_exhaustion(
    eff: CoreAttributes,
    state: HorseRuntimeState,
    max_stamina: float,
) -> CoreAttributes:
    """Cliff collapse + burst-empty clamp.

    Replaces the legacy progressive lerps with binary state changes:
      * cliff: aerobic ratio ≤ CLIFF_THRESHOLD → cruise drops to 40%, max
        drops to cruise + BURST_EMPTY_CLAMP, forward_accel drops to 20%.
      * burst clamp: when burst_pool ≤ 0, max_speed clamps to cruise + 0.5
        (no kick available). Only ratchets max_speed downward.
    """
    if max_stamina < 1e-6:
        return eff

    result = CoreAttributes(
        cruise_speed=eff.cruise_speed,
        max_speed=eff.max_speed,
        forward_accel=eff.forward_accel,
        turn_accel=eff.turn_accel,
        cornering_grip=eff.cornering_grip,
        stamina=eff.stamina,
        drain_rate_mult=eff.drain_rate_mult,
        weight=eff.weight,
        pushing_power=eff.pushing_power,
        push_resistance=eff.push_resistance,
    )

    ratio = state.current_stamina / max_stamina
    if ratio <= CLIFF_THRESHOLD:
        result.cruise_speed = eff.cruise_speed * CLIFF_CRUISE_MULT
        result.max_speed = result.cruise_speed + BURST_EMPTY_CLAMP
        result.forward_accel = eff.forward_accel * CLIFF_ACCEL_MULT
        return result

    if state.burst_pool <= 0.0:
        clamp = eff.cruise_speed + BURST_EMPTY_CLAMP
        if clamp < result.max_speed:
            result.max_speed = clamp

    return result
