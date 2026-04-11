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

# Stamina constants — physics redesign (validated in /tmp/hr-tests/probe_redesign.py)
#
# The aerobic pool is the main fuel; the burst pool is a small "kick reserve."
# Mechanics layered together:
#   A. Lead penalty: frontmost horse pays extra aerobic drain proportional to
#      (speed - cruise), scaled non-linearly by stamina. Stayers get cubic relief.
#   B. Draft recovery: horses within DRAFT_DISTANCE behind another horse AND
#      at/near cruise recover aerobic at a rate scaled by drain_rate_mult.
#   C. Burst pool: separate reserve sized as
#      BURST_K × (max_speed - cruise_speed) × stamina/100. Drains on excess²,
#      refills when speed < cruise - 0.5. Max_speed clamps to cruise + 0.5
#      when empty (no kick available).
#   D. Distance tax raised so cruising actually depletes without draft recovery.
#   E. Cliff collapse: at ≤5% aerobic, cruise drops to 40% and max_speed drops
#      to cruise + 0.5 in a single tick (no gradual lerp).
#
# The +8% drafting cruise modifier in modifiers.py is intentionally empty —
# drafting now provides aerobic recovery, not raw speed.

STAMINA_DRAIN_RATE: float = 0.01         # tangential push drain
SPEED_DRAIN_RATE: float = 0.0042         # distance tax (3× the legacy 0.0014)
CORNERING_DRAIN_RATE: float = 0.002      # cornering beyond grip
GRIP_FORCE_BASELINE: float = 150.0
LATERAL_STEERING_DRAIN_RATE: float = 0.006
LATERAL_VELOCITY_DRAIN_RATE: float = 0.0008

# A. Lead penalty
LEAD_K: float = 0.025
LEAD_STAMINA_REF: float = 100.0
LEAD_STAMINA_EXP: float = 2.5

# B. Draft recovery
DRAFT_DISTANCE: float = 15.0
DRAFT_RECOVERY_K: float = 0.04
DRAFT_RECOVERY_CRUISE_BUFFER: float = 0.5

# C. Burst pool
BURST_K: float = 2.5
BURST_DRAIN_K: float = 0.10
BURST_RECOVERY_K: float = 0.015
BURST_EMPTY_CLAMP: float = 0.5

# E. Cliff collapse
CLIFF_THRESHOLD: float = 0.05
CLIFF_CRUISE_MULT: float = 0.40
CLIFF_ACCEL_MULT: float = 0.20

# Legacy constants no longer used by the redesign (kept for now in case other
# modules import them; remove if no callers remain).
OVERDRIVE_DRAIN_RATE: float = 0.0  # replaced by burst pool dynamics

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
