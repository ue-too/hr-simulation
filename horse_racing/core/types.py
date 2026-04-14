"""Simulation types and physics constants — mirrors TS types.ts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .attributes import CoreAttributes


# --- Physics constants (literal copies from TS) ---

K_CRUISE = 2.0
"""Cruise controller proportional gain (1/s)."""

C_DRAG = 0.1
"""Linear drag coefficient (1/s)."""

TRACK_HALF_WIDTH = 10.325
"""Half-width of the track in meters."""

PHYS_HZ = 240
"""Physics substep frequency in Hz."""

PHYS_SUBSTEPS = 8
"""Number of physics substeps per game tick."""

FIXED_DT = 1 / PHYS_HZ
"""Fixed physics timestep in seconds (one substep)."""

NORMAL_DAMP = 0.5
"""Lateral velocity damping coefficient."""

HORSE_HALF_LENGTH = 1.0
"""Half-length of horse collision body in meters."""

HORSE_HALF_WIDTH = 0.325
"""Half-width of horse collision body in meters."""

MAX_HORSES = 24
"""Maximum number of horses supported per race."""


@dataclass
class InputState:
    tangential: float = 0.0
    normal: float = 0.0


@dataclass
class Horse:
    id: int
    color: int
    pos: np.ndarray  # shape (2,), [x, y]
    tangential_vel: float = 0.0
    normal_vel: float = 0.0
    track_progress: float = 0.0
    navigator: object = None  # TrackNavigator, set after import
    finished: bool = False
    finish_order: int | None = None
    base_attributes: CoreAttributes = field(
        default_factory=lambda: CoreAttributes(13, 20, 1.0, 1.0, 1.0, 100, 1.0, 500)
    )
    current_stamina: float = 100.0
    effective_attributes: CoreAttributes = field(
        default_factory=lambda: CoreAttributes(13, 20, 1.0, 1.0, 1.0, 100, 1.0, 500)
    )
    last_drain: float = 0.0
