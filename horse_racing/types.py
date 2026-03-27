"""Core data types and physics constants for the horse racing simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Physics constants (global, not per-horse)
# ---------------------------------------------------------------------------

HORSE_COUNT: int = 4
HORSE_RADIUS: float = 1.3  # ~2.6 m diameter circle wrapping a 2.5 m horse top-down
HORSE_SPACING: float = 3.0  # 3 m lane spacing (> 2 × radius, no overlap)

PHYS_HZ: int = 240
PHYS_SUBSTEPS: int = 8

NORMAL_DAMP: float = 0.5

TRACK_HALF_WIDTH: float = HORSE_SPACING * HORSE_COUNT + HORSE_RADIUS  # 13.3

RAIL_THICKNESS: float = 0.5  # 0.5 m rail (scaled down from 3.0 to match)

# Stamina constants
STAMINA_DRAIN_RATE: float = 0.1
OVERDRIVE_DRAIN_RATE: float = 0.05
CORNERING_DRAIN_RATE: float = 0.02
GRIP_FORCE_BASELINE: float = 150.0

# ---------------------------------------------------------------------------
# Track segment types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StraightSegment:
    tracktype: Literal["STRAIGHT"]
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    slope: float = 0.0  # grade (rise/run), positive = uphill


@dataclass(frozen=True)
class CurveSegment:
    tracktype: Literal["CURVE"]
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    center: tuple[float, float]
    radius: float
    angle_span: float  # radians, positive = CCW, negative = CW
    slope: float = 0.0  # grade (rise/run), positive = uphill


TrackSegment = StraightSegment | CurveSegment

# ---------------------------------------------------------------------------
# Track frame — local reference frame at a horse's position
# ---------------------------------------------------------------------------


@dataclass
class TrackFrame:
    tangential: np.ndarray  # unit vector (2,)
    normal: np.ndarray  # unit vector (2,), points outward
    turn_radius: float  # distance from curve center (inf on straights)
    target_radius: float  # lane-keeping target radius (inf on straights)
    segment_index: int  # which segment the horse is on
    slope: float = 0.0  # grade (rise/run), 0 = flat


# ---------------------------------------------------------------------------
# Horse action (from the RL agent)
# ---------------------------------------------------------------------------


@dataclass
class HorseAction:
    extra_tangential: float = 0.0
    extra_normal: float = 0.0


# ---------------------------------------------------------------------------
# Per-horse runtime body state
# ---------------------------------------------------------------------------


@dataclass
class HorseBody:
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    orientation: float = 0.0
    force: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def clear_force(self) -> None:
        self.force[:] = 0.0

    def apply_force(self, f: np.ndarray) -> None:
        self.force += f
