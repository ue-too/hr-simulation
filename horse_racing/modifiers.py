"""Modifier definitions, condition functions, and the 8 built-in modifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ActiveModifier:
    id: str
    strength: float  # 0–1, from genome


@dataclass
class ModifierEffect:
    target: str  # trait name
    flat: float = 0.0  # flat bonus (scaled by strength)
    pct: float = 0.0  # percentage bonus (scaled by strength)


@dataclass
class ModifierDefinition:
    id: str
    description: str
    condition: Callable[..., bool]
    effects: list[ModifierEffect] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Condition context — passed to condition functions each tick
# ---------------------------------------------------------------------------


@dataclass
class ModifierContext:
    horse_index: int
    positions: list[tuple[float, float]]  # all horses
    velocities: list[tuple[float, float]]
    track_progress: list[float]  # fraction [0,1] per horse
    current_stamina: float
    max_stamina: float
    track_surface: str = "dry"  # "dry", "wet", "heavy"


# ---------------------------------------------------------------------------
# Condition helpers
# ---------------------------------------------------------------------------


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _is_behind(
    my_pos: tuple[float, float],
    my_vel: tuple[float, float],
    other_pos: tuple[float, float],
    threshold: float,
) -> bool:
    """Check if other horse is ahead and within threshold distance."""
    dx = other_pos[0] - my_pos[0]
    dy = other_pos[1] - my_pos[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist > threshold or dist < 1e-6:
        return False
    # Dot with velocity direction
    speed = (my_vel[0] ** 2 + my_vel[1] ** 2) ** 0.5
    if speed < 1e-6:
        return False
    dot = (dx * my_vel[0] + dy * my_vel[1]) / (dist * speed)
    return dot > 0.5  # roughly ahead


def _horses_within(ctx: ModifierContext, radius: float) -> int:
    """Count horses within radius of the current horse (excluding self)."""
    my_pos = ctx.positions[ctx.horse_index]
    count = 0
    for i, pos in enumerate(ctx.positions):
        if i == ctx.horse_index:
            continue
        if _distance(my_pos, pos) <= radius:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Condition functions
# ---------------------------------------------------------------------------


def cond_drafting(ctx: ModifierContext) -> bool:
    my_pos = ctx.positions[ctx.horse_index]
    my_vel = ctx.velocities[ctx.horse_index]
    for i, pos in enumerate(ctx.positions):
        if i == ctx.horse_index:
            continue
        if _is_behind(my_pos, my_vel, pos, 15.0):
            return True
    return False


def cond_pack_pressure(ctx: ModifierContext) -> bool:
    return _horses_within(ctx, 20.0) >= 2


def cond_pack_anxiety(ctx: ModifierContext) -> bool:
    return _horses_within(ctx, 10.0) >= 3


def cond_front_runner(ctx: ModifierContext) -> bool:
    my_progress = ctx.track_progress[ctx.horse_index]
    for i in range(len(ctx.track_progress)):
        if i == ctx.horse_index:
            continue
        other = ctx.track_progress[i]
        if other > my_progress:
            return False
        # Tied: lower index wins (matches TS ranking tie-break)
        if other == my_progress and i < ctx.horse_index:
            return False
    return True


def cond_closer(ctx: ModifierContext) -> bool:
    my_progress = ctx.track_progress[ctx.horse_index]
    if my_progress < 0.75:
        return False
    return not cond_front_runner(ctx)


def cond_mudder(ctx: ModifierContext) -> bool:
    return ctx.track_surface in ("wet", "heavy")


def cond_gate_speed(ctx: ModifierContext) -> bool:
    return ctx.track_progress[ctx.horse_index] < 0.10


def cond_endurance(ctx: ModifierContext) -> bool:
    if ctx.max_stamina < 1e-6:
        return False
    return (ctx.current_stamina / ctx.max_stamina) < 0.40


# ---------------------------------------------------------------------------
# Registry of the 8 built-in modifiers
# ---------------------------------------------------------------------------

# Canonical ordered list of modifier IDs (used for observation vector)
MODIFIER_IDS: list[str] = [
    "drafting", "pack_pressure", "pack_anxiety", "front_runner",
    "closer", "mudder", "gate_speed", "endurance",
]

MODIFIER_REGISTRY: dict[str, ModifierDefinition] = {
    "drafting": ModifierDefinition(
        id="drafting",
        description="Aerodynamic benefit when behind another horse",
        condition=cond_drafting,
        effects=[ModifierEffect(target="cruise_speed", pct=0.08)],  # 2-8% scaled by strength
    ),
    "pack_pressure": ModifierDefinition(
        id="pack_pressure",
        description="Competitive instinct in a pack",
        condition=cond_pack_pressure,
        effects=[ModifierEffect(target="forward_accel", pct=0.10)],  # 3-10%
    ),
    "pack_anxiety": ModifierDefinition(
        id="pack_anxiety",
        description="Nervous in tight spaces",
        condition=cond_pack_anxiety,
        effects=[ModifierEffect(target="turn_accel", pct=-0.15)],  # -5 to -15%
    ),
    "front_runner": ModifierDefinition(
        id="front_runner",
        description="Performs better when leading",
        condition=cond_front_runner,
        effects=[ModifierEffect(target="cruise_speed", flat=0.7)],  # 0.2-0.7
    ),
    "closer": ModifierDefinition(
        id="closer",
        description="Closing kick in final stretch",
        condition=cond_closer,
        effects=[ModifierEffect(target="max_speed", pct=0.08)],  # 3-8%
    ),
    "mudder": ModifierDefinition(
        id="mudder",
        description="Handles wet conditions well",
        condition=cond_mudder,
        effects=[ModifierEffect(target="cornering_grip", pct=0.15)],  # 5-15%
    ),
    "gate_speed": ModifierDefinition(
        id="gate_speed",
        description="Explosive starts",
        condition=cond_gate_speed,
        effects=[ModifierEffect(target="forward_accel", pct=0.25)],  # 10-25%
    ),
    "endurance": ModifierDefinition(
        id="endurance",
        description="Recovers better when tired",
        condition=cond_endurance,
        effects=[ModifierEffect(target="stamina_recovery", pct=0.30)],  # 10-30%
    ),
}
