"""Core data types and constants for the v2 horse racing simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------

MAX_HORSE_COUNT: int = 20
MAX_REL_HORSES: int = 8  # observe top 8 nearby horses

# Horse body dimensions (meters)
HORSE_LENGTH: float = 2.5
HORSE_WIDTH: float = 0.65
HORSE_HALF_LENGTH: float = HORSE_LENGTH / 2
HORSE_HALF_WIDTH: float = HORSE_WIDTH / 2

HORSE_SPACING: float = 1.0  # lane spacing

PHYS_HZ: int = 240
PHYS_SUBSTEPS: int = 8

NORMAL_DAMP: float = 0.5

# Track width accommodates MAX_HORSE_COUNT horses
TRACK_HALF_WIDTH: float = HORSE_SPACING * MAX_HORSE_COUNT / 2 + HORSE_HALF_WIDTH

RAIL_THICKNESS: float = 0.5
WALL_RESTITUTION: float = 0.4

# Drafting
DRAFT_DISTANCE: float = 15.0  # meters behind to receive draft
DRAFT_SPEED_BONUS: float = 0.08  # 8% speed boost when drafting

# Movement smoothing
RESPONSE_TAU: float = 0.3  # seconds for action-to-force EMA
STRIDE_AMPLITUDE: float = 0.2  # m/s forward speed oscillation
STRIDE_FREQUENCY: float = 2.5  # Hz gallop rhythm
FATIGUE_WOBBLE_SCALE: float = 0.1  # m/s lateral drift when exhausted

# ---------------------------------------------------------------------------
# Track segment types (reused from v1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StraightSegment:
    tracktype: Literal["STRAIGHT"]
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    slope: float = 0.0


@dataclass(frozen=True)
class CurveSegment:
    tracktype: Literal["CURVE"]
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    center: tuple[float, float]
    radius: float
    angle_span: float  # positive = CCW, negative = CW
    slope: float = 0.0


TrackSegment = Union[StraightSegment, CurveSegment]


@dataclass(frozen=True)
class TrackData:
    segments: list[TrackSegment]
    inner_rails: list[TrackSegment]
    outer_rails: list[TrackSegment]


# ---------------------------------------------------------------------------
# Track frame — local reference at a horse's position
# ---------------------------------------------------------------------------


@dataclass
class TrackFrame:
    tangential: np.ndarray  # unit vector (2,), forward direction
    normal: np.ndarray  # unit vector (2,), points outward
    turn_radius: float  # distance from curve center (inf on straights)
    target_radius: float  # lane-keeping target (inf on straights)
    segment_index: int
    slope: float = 0.0
    turn_direction: float = 0.0  # -1 = left turn, +1 = right turn, 0 = straight


# ---------------------------------------------------------------------------
# Jockey action (effort + lane)
# ---------------------------------------------------------------------------


@dataclass
class JockeyAction:
    effort: float = 0.0  # [-1, 1]: -1=ease, 0=cruise, 1=max push
    lane: float = 0.0  # [-1, 1]: -1=inside, 0=hold, 1=outside


# ---------------------------------------------------------------------------
# Per-horse runtime body state
# ---------------------------------------------------------------------------


@dataclass
class HorseBody:
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    orientation: float = 0.0
    force: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # Smoothed force for response lag
    smoothed_forward_force: float = 0.0
    smoothed_lateral_force: float = 0.0

    def clear_force(self) -> None:
        self.force[:] = 0.0

    def apply_force(self, f: np.ndarray) -> None:
        self.force += f


# ---------------------------------------------------------------------------
# Jockey style parameters
# ---------------------------------------------------------------------------


@dataclass
class JockeyStyle:
    risk_tolerance: float = 0.5  # [0, 1]: conservative to aggressive
    tactical_bias: float = 0.0  # [-1, 1]: front-runner to closer
    skill_level: float = 1.0  # [0, 1]: novice to elite
