"""Observation vector construction for the v2 racing environment.

Layout (~63 features):
  Ego state (7): speed, lateral_vel, displacement, progress, curvature,
                 curvature_direction, stamina_ratio
  Horse profile (6): top_speed, acceleration, stamina_efficiency,
                     cornering_grip_left, cornering_grip_right, climbing_power
  Jockey style (3): risk_tolerance, tactical_bias, skill_level
  Race context (3): placement_norm, num_horses_norm, race_progress_elapsed
  Track lookahead (12): [curvature, turn_direction, length, slope] x 3 segments
  Relative horses (32): [tang_offset, norm_offset, rel_speed, stamina_est] x 8
  Total: 63
"""

from __future__ import annotations

import math

import numpy as np

from horse_racing.core.horse import HorseProfile
from horse_racing.core.stamina import StaminaState
from horse_racing.core.track import compute_segment_length
from horse_racing.core.track_navigator import TrackNavigator
from horse_racing.core.types import (
    MAX_REL_HORSES,
    CurveSegment,
    HorseBody,
    JockeyStyle,
    TrackFrame,
    TrackSegment,
)

OBS_SIZE: int = 63

# Normalization constants
MAX_SPEED: float = 25.0  # m/s (above any horse's top_speed for headroom)
MAX_DISPLACEMENT: float = 12.0  # meters from centerline
MAX_CURVATURE: float = 0.1  # 1/radius, capped
MAX_SEGMENT_LENGTH: float = 500.0  # meters
MAX_TANG_OFFSET: float = 100.0  # meters ahead/behind
MAX_NORM_OFFSET: float = 25.0  # meters lateral


def build_observation(
    horse_idx: int,
    bodies: list[HorseBody],
    profiles: list[HorseProfile],
    staminas: list[StaminaState],
    frames: list[TrackFrame],
    navigators: list[TrackNavigator],
    jockey_style: JockeyStyle,
    num_horses: int,
    placement: int,
    sim_time: float,
    total_race_time_est: float,
) -> np.ndarray:
    """Build the observation vector for a single horse."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    idx = 0

    body = bodies[horse_idx]
    profile = profiles[horse_idx]
    stamina = staminas[horse_idx]
    frame = frames[horse_idx]
    nav = navigators[horse_idx]

    # --- Ego state (7) ---
    tang_speed = float(np.dot(body.velocity, frame.tangential))
    norm_speed = float(np.dot(body.velocity, frame.normal))
    progress = nav.compute_progress(body.position)

    # Displacement from centerline (signed)
    displacement = _compute_displacement(body, frame)

    curvature = 1.0 / frame.turn_radius if frame.turn_radius < 1e6 else 0.0

    obs[idx] = tang_speed / MAX_SPEED; idx += 1
    obs[idx] = norm_speed / MAX_SPEED; idx += 1
    obs[idx] = displacement / MAX_DISPLACEMENT; idx += 1
    obs[idx] = progress; idx += 1
    obs[idx] = min(curvature, MAX_CURVATURE) / MAX_CURVATURE; idx += 1
    obs[idx] = frame.turn_direction; idx += 1  # -1, 0, or 1
    obs[idx] = stamina.ratio; idx += 1

    # --- Horse profile (6) ---
    obs[idx] = (profile.top_speed - 16.0) / 4.0; idx += 1  # normalize to [0, 1]
    obs[idx] = (profile.acceleration - 0.5) / 1.0; idx += 1
    obs[idx] = (profile.stamina_efficiency - 0.7) / 0.6; idx += 1
    obs[idx] = (profile.cornering_grip_left - 0.5) / 1.0; idx += 1
    obs[idx] = (profile.cornering_grip_right - 0.5) / 1.0; idx += 1
    obs[idx] = (profile.climbing_power - 0.5) / 1.0; idx += 1

    # --- Jockey style (3) ---
    obs[idx] = jockey_style.risk_tolerance; idx += 1
    obs[idx] = (jockey_style.tactical_bias + 1.0) / 2.0; idx += 1  # [-1,1] -> [0,1]
    obs[idx] = jockey_style.skill_level; idx += 1

    # --- Race context (3) ---
    obs[idx] = placement / max(1, num_horses - 1); idx += 1  # 0 = first
    obs[idx] = num_horses / 20.0; idx += 1
    elapsed_ratio = sim_time / total_race_time_est if total_race_time_est > 0 else 0.0
    obs[idx] = min(1.0, elapsed_ratio); idx += 1

    # --- Track lookahead (12) ---
    lookahead = nav.lookahead_segments(3)
    for seg in lookahead:
        seg_curvature, seg_direction, seg_length, seg_slope = _segment_features(seg)
        obs[idx] = min(seg_curvature, MAX_CURVATURE) / MAX_CURVATURE; idx += 1
        obs[idx] = seg_direction; idx += 1
        obs[idx] = min(seg_length, MAX_SEGMENT_LENGTH) / MAX_SEGMENT_LENGTH; idx += 1
        obs[idx] = seg_slope; idx += 1  # raw slope value, usually small

    # --- Relative horses (32) ---
    my_progress = progress
    my_tang_speed = tang_speed

    # Collect (distance, j) pairs for sorting
    others = []
    for j in range(num_horses):
        if j == horse_idx:
            continue
        other_progress = navigators[j].compute_progress(bodies[j].position)
        dx = float(bodies[j].position[0] - body.position[0])
        dy = float(bodies[j].position[1] - body.position[1])
        dist = math.sqrt(dx * dx + dy * dy)
        others.append((dist, j, other_progress))

    # Sort by proximity (closest first)
    others.sort(key=lambda x: x[0])

    for k in range(MAX_REL_HORSES):
        if k < len(others):
            _, j, other_progress = others[k]
            other_body = bodies[j]
            other_frame = frames[j]

            # Tangential offset (progress difference in track-length units)
            tang_offset = (other_progress - my_progress) * nav.total_length

            # Normal offset (lateral separation)
            diff = other_body.position - body.position
            norm_offset = float(np.dot(diff, frame.normal))

            # Relative speed
            other_tang_speed = float(np.dot(other_body.velocity, other_frame.tangential))
            rel_speed = other_tang_speed - my_tang_speed

            # Stamina estimate (noisy — 80% accurate)
            stamina_est = staminas[j].ratio * 0.8 + 0.1  # add noise band

            obs[idx] = tang_offset / MAX_TANG_OFFSET; idx += 1
            obs[idx] = norm_offset / MAX_NORM_OFFSET; idx += 1
            obs[idx] = rel_speed / MAX_SPEED; idx += 1
            obs[idx] = stamina_est; idx += 1
        else:
            # Pad with zeros
            idx += 4

    return obs


def _compute_displacement(body: HorseBody, frame: TrackFrame) -> float:
    """Signed displacement from track centerline. Negative = inside, positive = outside."""
    # For curves, displacement = distance_from_center - curve_radius
    if frame.turn_radius < 1e6:
        return frame.turn_radius - frame.target_radius
    # For straights, project position onto normal
    return 0.0  # on straights, displacement is relative to entry position


def _segment_features(seg: TrackSegment) -> tuple[float, float, float, float]:
    """Extract (curvature, turn_direction, length, slope) from a segment."""
    length = compute_segment_length(seg)
    slope = seg.slope

    if isinstance(seg, CurveSegment):
        curvature = 1.0 / seg.radius if seg.radius > 0 else 0.0
        # Direction from angle_span sign: positive = CCW = left turn convention
        direction = 1.0 if seg.angle_span >= 0 else -1.0
        return curvature, direction, length, slope
    else:
        return 0.0, 0.0, length, slope
