"""Per-horse core attributes, trait ranges, and effective-attribute resolution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from horse_racing.modifiers import ActiveModifier

# ---------------------------------------------------------------------------
# Core attributes (10 traits)
# ---------------------------------------------------------------------------

TRAIT_RANGES: dict[str, tuple[float, float]] = {
    "cruise_speed": (8.0, 18.0),
    "max_speed": (15.0, 25.0),
    "forward_accel": (0.5, 1.5),
    "turn_accel": (0.5, 1.5),
    "cornering_grip": (0.5, 1.5),
    "stamina": (50.0, 150.0),
    "stamina_recovery": (0.5, 2.0),
    "weight": (400.0, 600.0),
    "pushing_power": (0.0, 1.0),
    "push_resistance": (0.0, 1.0),
}

TRAIT_NAMES: list[str] = list(TRAIT_RANGES.keys())


@dataclass
class CoreAttributes:
    cruise_speed: float = 13.0
    max_speed: float = 20.0
    forward_accel: float = 1.0
    turn_accel: float = 1.0
    cornering_grip: float = 1.0
    stamina: float = 100.0
    stamina_recovery: float = 1.0
    weight: float = 500.0
    pushing_power: float = 0.5
    push_resistance: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Modifier resolution
# ---------------------------------------------------------------------------


def resolve_effective(
    base: CoreAttributes,
    active_modifiers: list[ActiveModifier],
) -> CoreAttributes:
    """Apply flat and percentage bonuses from active modifiers, clamp to trait ranges."""
    from horse_racing.modifiers import MODIFIER_REGISTRY

    flat_bonuses: dict[str, float] = defaultdict(float)
    pct_bonuses: dict[str, float] = defaultdict(float)

    for mod in active_modifiers:
        defn = MODIFIER_REGISTRY[mod.id]
        for effect in defn.effects:
            if effect.flat:
                flat_bonuses[effect.target] += effect.flat * mod.strength
            if effect.pct:
                pct_bonuses[effect.target] += effect.pct * mod.strength

    result: dict[str, float] = {}
    for trait in TRAIT_NAMES:
        base_val = getattr(base, trait)
        effective = (base_val + flat_bonuses[trait]) * (1.0 + pct_bonuses[trait])
        lo, hi = TRAIT_RANGES[trait]
        result[trait] = max(lo, min(hi, effective))

    return CoreAttributes(**result)
