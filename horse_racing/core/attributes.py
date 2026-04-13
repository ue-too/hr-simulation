"""CoreAttributes dataclass and trait ranges — mirrors TS attributes.ts."""

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


def create_default_attributes() -> CoreAttributes:
    return CoreAttributes(
        cruise_speed=13,
        max_speed=20,
        forward_accel=1.0,
        turn_accel=1.0,
        cornering_grip=1.0,
        max_stamina=100,
        drain_rate_mult=1.0,
        weight=500,
    )
