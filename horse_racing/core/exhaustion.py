"""Gradual exhaustion model — sigmoid-knee degradation curve."""

import dataclasses
import math

from .attributes import CoreAttributes
from .types import Horse

KNEE = 0.10
FLOOR = 0.45
K = 10

# Max speed degrades faster — push ceiling shrinks before cruise does
MAX_SPEED_KNEE = 0.25
MAX_SPEED_FLOOR = 0.40
MAX_SPEED_K = 10


def effective_ratio(
    stamina_pct: float,
    knee: float = KNEE,
    floor: float = FLOOR,
    k: float = K,
) -> float:
    """Map stamina percentage (0-1) to stat multiplier (floor-1.0).

    Uses a sigmoid curve centered on `knee`:
    - Above knee: stats degrade gently
    - Below knee: stats drop steeply
    - At 0 stamina: stats bottom out at `floor`
    """
    raw = 1.0 / (1.0 + math.exp(-k * (stamina_pct - knee)))
    sig_at_1 = 1.0 / (1.0 + math.exp(-k * (1.0 - knee)))
    sig_at_0 = 1.0 / (1.0 + math.exp(-k * (0.0 - knee)))
    normalized = (raw - sig_at_0) / (sig_at_1 - sig_at_0)
    return floor + (1.0 - floor) * normalized


def apply_exhaustion(horse: Horse) -> CoreAttributes:
    """Compute effective attributes based on current stamina level.

    max_speed uses an aggressive curve so the push ceiling shrinks
    faster than cruise — the late-race "kick" comes from holding pace
    while exhausted opponents fade, not from supernatural acceleration.
    """
    base = horse.base_attributes
    stamina_pct = horse.current_stamina / base.max_stamina if base.max_stamina > 0 else 0.0
    stamina_pct = max(0.0, min(1.0, stamina_pct))
    ratio = effective_ratio(stamina_pct)
    max_speed_ratio = effective_ratio(stamina_pct, MAX_SPEED_KNEE, MAX_SPEED_FLOOR, MAX_SPEED_K)
    return dataclasses.replace(
        base,
        cruise_speed=base.cruise_speed * ratio,
        max_speed=base.max_speed * max_speed_ratio,
        forward_accel=base.forward_accel * ratio,
        turn_accel=base.turn_accel * ratio,
    )
