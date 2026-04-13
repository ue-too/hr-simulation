"""Binary exhaustion model — mirrors TS exhaustion.ts."""

import dataclasses

from .attributes import CoreAttributes
from .types import Horse

EXHAUSTION_DECAY = 0.95

_CRUISE_SPEED_FLOOR_RATIO = 0.4
_MAX_SPEED_FLOOR_RATIO = 0.55
_FORWARD_ACCEL_FLOOR_RATIO = 0.15
_TURN_ACCEL_FLOOR_RATIO = 0.3


def apply_exhaustion(horse: Horse) -> CoreAttributes:
    """Resolve effective attributes based on stamina state."""
    base = horse.base_attributes

    if horse.current_stamina > 0:
        return dataclasses.replace(base)

    eff = horse.effective_attributes
    floor_cruise = base.cruise_speed * _CRUISE_SPEED_FLOOR_RATIO
    floor_max_speed = base.cruise_speed * _MAX_SPEED_FLOOR_RATIO
    floor_forward_accel = base.forward_accel * _FORWARD_ACCEL_FLOOR_RATIO
    floor_turn_accel = base.turn_accel * _TURN_ACCEL_FLOOR_RATIO

    return dataclasses.replace(
        base,
        cruise_speed=floor_cruise + (eff.cruise_speed - floor_cruise) * EXHAUSTION_DECAY,
        max_speed=floor_max_speed + (eff.max_speed - floor_max_speed) * EXHAUSTION_DECAY,
        forward_accel=floor_forward_accel + (eff.forward_accel - floor_forward_accel) * EXHAUSTION_DECAY,
        turn_accel=floor_turn_accel + (eff.turn_accel - floor_turn_accel) * EXHAUSTION_DECAY,
    )
