"""CoreAttributes dataclass and trait ranges — mirrors TS attributes.ts."""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class CoreAttributes:
    cruise_speed: float
    max_speed: float
    forward_accel: float
    turn_accel: float
    cornering_grip: float
    max_stamina: float
    drain_rate_mult: float
    weight: float


TRAIT_RANGES: dict[str, tuple[float, float]] = {
    "cruise_speed": (8, 18),
    "max_speed": (15, 25),
    "forward_accel": (0.5, 1.5),
    "turn_accel": (0.5, 1.5),
    "cornering_grip": (0.5, 1.5),
    "max_stamina": (50, 150),
    "drain_rate_mult": (0.7, 1.3),
    "weight": (400, 600),
}

# Base force caps in m/s² (scaled by forwardAccel / turnAccel)
F_T_MAX = 5
F_N_MAX = 3

# Defaults (midpoints of trait ranges)
_DEFAULTS = {
    "cruise_speed": 13.0,
    "max_speed": 20.0,
    "forward_accel": 1.0,
    "turn_accel": 1.0,
    "cornering_grip": 1.0,
    "max_stamina": 100.0,
    "drain_rate_mult": 1.0,
    "weight": 500.0,
}


def create_default_attributes() -> CoreAttributes:
    return CoreAttributes(**_DEFAULTS)


def create_randomized_attributes(jitter: float = 0.10) -> CoreAttributes:
    """Create attributes with ±jitter variation around defaults.

    Each trait is independently scaled by a uniform factor in
    [1-jitter, 1+jitter]. For drain_rate_mult the jitter direction
    is inverted so that "better" means lower drain (consistent with
    other traits where higher = better).

    Args:
        jitter: Fractional variation (0.10 = ±10%).
    """
    def vary(base: float) -> float:
        return base * random.uniform(1 - jitter, 1 + jitter)

    return CoreAttributes(
        cruise_speed=vary(_DEFAULTS["cruise_speed"]),
        max_speed=vary(_DEFAULTS["max_speed"]),
        forward_accel=vary(_DEFAULTS["forward_accel"]),
        turn_accel=vary(_DEFAULTS["turn_accel"]),
        cornering_grip=vary(_DEFAULTS["cornering_grip"]),
        max_stamina=vary(_DEFAULTS["max_stamina"]),
        drain_rate_mult=vary(_DEFAULTS["drain_rate_mult"]),
        weight=vary(_DEFAULTS["weight"]),
    )
