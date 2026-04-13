"""v2 simulation core — mirrors TS ground truth."""

from .attributes import CoreAttributes, TRAIT_RANGES, F_T_MAX, F_N_MAX, create_default_attributes
from .collision import CollisionWorld
from .exhaustion import apply_exhaustion
from .observation import build_observations, OBS_SIZE
from .physics import compute_accelerations, project_velocity, step_physics
from .race import Race, RaceState, spawn_horses
from .stamina import drain_stamina
from .track import load_track_json, StraightSegment, CurveSegment, TrackSegment
from .track_navigator import TrackNavigator, TrackFrame
from .types import (
    Horse, InputState, K_CRUISE, C_DRAG, TRACK_HALF_WIDTH,
    PHYS_HZ, PHYS_SUBSTEPS, FIXED_DT, NORMAL_DAMP,
    HORSE_HALF_LENGTH, HORSE_HALF_WIDTH, MAX_HORSES,
)
