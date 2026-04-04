"""Horse profile — physical capabilities only, no strategy or personality."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class HorseProfile:
    """A horse's physical attributes. The horse is just a body."""

    top_speed: float  # 16-20 m/s, absolute speed ceiling
    acceleration: float  # 0.5-1.5, force scaling for reaching target speed
    stamina_pool: float  # 60-150, total energy budget (drain only)
    stamina_efficiency: float  # 0.7-1.3, how cheaply it sustains speed
    cornering_grip_left: float  # 0.5-1.5, lateral grip on left turns
    cornering_grip_right: float  # 0.5-1.5, lateral grip on right turns
    weight: float  # 430-550 kg, mass for collisions and inertia
    climbing_power: float  # 0.5-1.5, uphill/downhill performance

    @property
    def efficiency_speed(self) -> float:
        """Speed the horse can sustain cheaply (75% of top speed)."""
        return self.top_speed * 0.75


# Trait ranges for random generation
TRAIT_RANGES: dict[str, tuple[float, float]] = {
    "top_speed": (16.0, 20.0),
    "acceleration": (0.5, 1.5),
    "stamina_pool": (60.0, 150.0),
    "stamina_efficiency": (0.7, 1.3),
    "cornering_grip_left": (0.5, 1.5),
    "cornering_grip_right": (0.5, 1.5),
    "weight": (430.0, 550.0),
    "climbing_power": (0.5, 1.5),
}


def random_horse(rng: random.Random | None = None) -> HorseProfile:
    """Generate a horse with uniformly random traits."""
    r = rng or random.Random()
    return HorseProfile(**{
        name: r.uniform(lo, hi)
        for name, (lo, hi) in TRAIT_RANGES.items()
    })


def default_horse() -> HorseProfile:
    """A neutral horse with all traits at midpoint."""
    return HorseProfile(**{
        name: (lo + hi) / 2
        for name, (lo, hi) in TRAIT_RANGES.items()
    })
