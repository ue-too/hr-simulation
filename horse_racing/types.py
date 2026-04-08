"""Core data types and physics constants for the horse racing simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np

# ---------------------------------------------------------------------------
# Physics constants (global, not per-horse)
# ---------------------------------------------------------------------------

HORSE_COUNT: int = 4
MAX_HORSE_COUNT: int = 20
MAX_REL_HORSES: int = MAX_HORSE_COUNT - 1  # 19 observable opponents
REL_FEATURES_PER_HORSE: int = 4  # tang_offset, norm_offset, rel_tang_vel, rel_norm_vel
NUM_SKILLS: int = 6
SKILL_IDS: list[str] = [
    "stamina_management",
    "sprint_timing",
    "overtake",
    "drafting_exploit",
    "cornering_line",
    "pace_pressure",
]
OBS_SIZE: int = 8 + MAX_REL_HORSES * REL_FEATURES_PER_HORSE + 13 + 8 + NUM_SKILLS  # 111

# Rectangle horse dimensions (realistic: ~2.5 m nose-to-tail, ~0.65 m shoulder width)
HORSE_LENGTH: float = 2.5
HORSE_WIDTH: float = 0.65
HORSE_HALF_LENGTH: float = HORSE_LENGTH / 2   # 1.25 m
HORSE_HALF_WIDTH: float = HORSE_WIDTH / 2     # 0.325 m

HORSE_SPACING: float = 1.0  # 1 m lane spacing (0.65 m horse + 0.35 m gap)

PHYS_HZ: int = 240
PHYS_SUBSTEPS: int = 8

NORMAL_DAMP: float = 0.5
CRUISE_BLEND_THRESHOLD: float = 1.0  # jockey input (m/s²) at which auto-cruise fully fades out

# Track wide enough for MAX_HORSE_COUNT horses centered + clearance
TRACK_HALF_WIDTH: float = HORSE_SPACING * MAX_HORSE_COUNT / 2 + HORSE_HALF_WIDTH  # ~10.325 m

RAIL_THICKNESS: float = 0.5  # 0.5 m rail (scaled down from 3.0 to match)

# Stamina constants (no recovery — fixed pool, drain only)
STAMINA_DRAIN_RATE: float = 0.03         # tangential push drain (3x from 0.01)
OVERDRIVE_DRAIN_RATE: float = 0.015      # exceeding cruise speed (3x from 0.005)
CORNERING_DRAIN_RATE: float = 0.002      # cornering beyond grip (was 0.02)
SPEED_DRAIN_RATE: float = 0.0014         # distance tax (was 0.014)
GRIP_FORCE_BASELINE: float = 150.0       # unchanged
LATERAL_STEERING_DRAIN_RATE: float = 0.006   # lateral steering input
LATERAL_VELOCITY_DRAIN_RATE: float = 0.0008  # sustained lateral drift

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


TrackSegment = Union[StraightSegment, CurveSegment]


@dataclass(frozen=True)
class TrackData:
    """Track definition with centerline and optional explicit rail paths."""
    segments: list[TrackSegment]
    inner_rails: list[TrackSegment]
    outer_rails: list[TrackSegment]

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
