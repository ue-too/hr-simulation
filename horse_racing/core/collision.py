"""SAT-based collision solver — mirrors @ue-too/dynamics behavior."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .track import CurveSegment, StraightSegment, TrackSegment
from .track_navigator import TrackFrame
from .types import HORSE_HALF_LENGTH, HORSE_HALF_WIDTH


@dataclass
class OBB:
    """Oriented bounding box."""
    center: np.ndarray  # shape (2,)
    half_length: float
    half_width: float
    angle: float  # radians

    def vertices(self) -> list[np.ndarray]:
        c, s = math.cos(self.angle), math.sin(self.angle)
        ax = np.array([c, s])
        ay = np.array([-s, c])
        hl, hw = self.half_length, self.half_width
        return [
            self.center + ax * hl + ay * hw,
            self.center + ax * hl - ay * hw,
            self.center - ax * hl - ay * hw,
            self.center - ax * hl + ay * hw,
        ]

    def axes(self) -> list[np.ndarray]:
        c, s = math.cos(self.angle), math.sin(self.angle)
        return [np.array([c, s]), np.array([-s, c])]


def _project_verts(verts: list[np.ndarray], axis: np.ndarray) -> tuple[float, float]:
    dots = [float(np.dot(v, axis)) for v in verts]
    return min(dots), max(dots)


def sat_overlap(a: OBB, b: OBB) -> tuple[float, np.ndarray] | None:
    """Test two OBBs for overlap using SAT.
    Returns None if separated, or (depth, normal) pointing from a to b.
    """
    verts_a = a.vertices()
    verts_b = b.vertices()
    axes = a.axes() + b.axes()

    min_depth = float("inf")
    min_normal = axes[0]

    for axis in axes:
        lo_a, hi_a = _project_verts(verts_a, axis)
        lo_b, hi_b = _project_verts(verts_b, axis)

        if hi_a <= lo_b or hi_b <= lo_a:
            return None

        depth = min(hi_a - lo_b, hi_b - lo_a)
        if depth < min_depth:
            min_depth = depth
            center_diff = b.center - a.center
            if np.dot(center_diff, axis) < 0:
                min_normal = -axis
            else:
                min_normal = axis.copy()

    return min_depth, min_normal


def _segment_vs_obb(
    rail_obb: OBB, horse_obb: OBB
) -> tuple[float, np.ndarray] | None:
    """Test a precomputed rail OBB against a horse OBB."""
    return sat_overlap(rail_obb, horse_obb)


def _aabb_overlap(
    cx_a: float, cy_a: float, r_a: float,
    cx_b: float, cy_b: float, r_b: float,
) -> bool:
    """Fast circle-based broad-phase check."""
    dx = cx_a - cx_b
    dy = cy_a - cy_b
    total_r = r_a + r_b
    return dx * dx + dy * dy <= total_r * total_r


@dataclass
class RailBody:
    """Precomputed rail segment OBB + bounding circle for broad-phase."""
    obb: OBB
    cx: float  # center x
    cy: float  # center y
    radius: float  # bounding circle radius
    track_seg_index: int  # which track segment this rail belongs to


@dataclass
class HorseBody:
    id: int
    obb: OBB
    velocity: np.ndarray
    mass: float


# Bounding circle radius for a horse OBB
_HORSE_BOUND_RADIUS = math.sqrt(HORSE_HALF_LENGTH**2 + HORSE_HALF_WIDTH**2)

# How many track segments ahead/behind to check for rail collisions
_RAIL_SEARCH_WINDOW = 2


class CollisionWorld:
    """Manages horse bodies and rail segments for collision detection."""

    def __init__(self, segments: list[TrackSegment], half_track_width: float):
        self._horses: dict[int, HorseBody] = {}
        self._rail_bodies: list[RailBody] = []
        self._rails_by_seg: dict[int, list[int]] = {}  # seg_index -> [rail_body indices]
        self._num_track_segments = len(segments)
        self._build_rails(segments, half_track_width)

    def _build_rails(
        self, segments: list[TrackSegment], half_track_width: float
    ) -> None:
        for seg_idx, seg in enumerate(segments):
            seg_rails: list[int] = []

            if isinstance(seg, StraightSegment):
                d = seg.end_point - seg.start_point
                length = np.linalg.norm(d)
                if length < 1e-6:
                    continue
                fwd = d / length
                n = np.array([-fwd[1], fwd[0]])
                angle = math.atan2(fwd[1], fwd[0])

                for sign in [1.0, -1.0]:
                    p1 = seg.start_point + n * (sign * half_track_width)
                    p2 = seg.end_point + n * (sign * half_track_width)
                    mid = (p1 + p2) / 2
                    obb = OBB(mid, length / 2, 0.01, angle)
                    r = math.sqrt((length / 2) ** 2 + 0.01**2)
                    idx = len(self._rail_bodies)
                    self._rail_bodies.append(RailBody(
                        obb=obb, cx=float(mid[0]), cy=float(mid[1]),
                        radius=r, track_seg_index=seg_idx,
                    ))
                    seg_rails.append(idx)
            else:
                arc_steps = max(4, int(abs(seg.angle_span) / (5 * math.pi / 180)))
                to_start = seg.start_point - seg.center
                start_angle = math.atan2(to_start[1], to_start[0])

                for r_offset in [-half_track_width, half_track_width]:
                    r = seg.radius + r_offset
                    if r < 0.1:
                        continue
                    for i in range(arc_steps):
                        t0 = i / arc_steps
                        t1 = (i + 1) / arc_steps
                        a0 = start_angle + seg.angle_span * t0
                        a1 = start_angle + seg.angle_span * t1
                        p0 = seg.center + np.array([math.cos(a0), math.sin(a0)]) * r
                        p1 = seg.center + np.array([math.cos(a1), math.sin(a1)]) * r
                        d = p1 - p0
                        seg_len = float(np.linalg.norm(d))
                        if seg_len < 1e-9:
                            continue
                        mid = (p0 + p1) / 2
                        angle = math.atan2(d[1], d[0])
                        obb = OBB(mid, seg_len / 2, 0.01, angle)
                        bound_r = math.sqrt((seg_len / 2) ** 2 + 0.01**2)
                        idx = len(self._rail_bodies)
                        self._rail_bodies.append(RailBody(
                            obb=obb, cx=float(mid[0]), cy=float(mid[1]),
                            radius=bound_r, track_seg_index=seg_idx,
                        ))
                        seg_rails.append(idx)

            self._rails_by_seg[seg_idx] = seg_rails

    def _nearby_rail_indices(self, seg_index: int) -> list[int]:
        """Get rail body indices near a given track segment index."""
        indices: list[int] = []
        n = self._num_track_segments
        for offset in range(-_RAIL_SEARCH_WINDOW, _RAIL_SEARCH_WINDOW + 1):
            # Wrap around for closed tracks
            idx = (seg_index + offset) % n
            if idx in self._rails_by_seg:
                indices.extend(self._rails_by_seg[idx])
        return indices

    def add_horse(
        self, horse_id: int, pos: np.ndarray, frame: TrackFrame, mass: float,
    ) -> None:
        angle = math.atan2(frame.tangential[1], frame.tangential[0])
        obb = OBB(pos.copy(), HORSE_HALF_LENGTH, HORSE_HALF_WIDTH, angle)
        self._horses[horse_id] = HorseBody(
            id=horse_id, obb=obb, velocity=np.zeros(2), mass=mass
        )

    def remove_horse(self, horse_id: int) -> None:
        self._horses.pop(horse_id, None)

    def set_horse_state(
        self, horse_id: int, pos: np.ndarray, vel: np.ndarray, frame: TrackFrame,
        seg_index: int = 0,
    ) -> None:
        body = self._horses[horse_id]
        body.obb.center = pos.copy()
        body.obb.angle = math.atan2(frame.tangential[1], frame.tangential[0])
        body.velocity = vel.copy()

    def get_horse_state(self, horse_id: int) -> tuple[np.ndarray, np.ndarray]:
        body = self._horses[horse_id]
        return body.obb.center.copy(), body.velocity.copy()

    def step(self, dt: float, horse_seg_indices: dict[int, int] | None = None) -> None:
        horse_ids = list(self._horses.keys())

        # Horse-horse collisions
        for i in range(len(horse_ids)):
            for j in range(i + 1, len(horse_ids)):
                a = self._horses[horse_ids[i]]
                b = self._horses[horse_ids[j]]
                # Quick distance check
                if not _aabb_overlap(
                    float(a.obb.center[0]), float(a.obb.center[1]), _HORSE_BOUND_RADIUS,
                    float(b.obb.center[0]), float(b.obb.center[1]), _HORSE_BOUND_RADIUS,
                ):
                    continue
                result = sat_overlap(a.obb, b.obb)
                if result is not None:
                    depth, normal = result
                    self._resolve_dynamic(a, b, depth, normal)

        # Horse-rail collisions — only check nearby segments
        for hid in horse_ids:
            body = self._horses[hid]
            hx = float(body.obb.center[0])
            hy = float(body.obb.center[1])

            if horse_seg_indices is not None and hid in horse_seg_indices:
                rail_indices = self._nearby_rail_indices(horse_seg_indices[hid])
            else:
                rail_indices = range(len(self._rail_bodies))

            for ri in rail_indices:
                rail = self._rail_bodies[ri]
                # Broad-phase: bounding circle check
                if not _aabb_overlap(hx, hy, _HORSE_BOUND_RADIUS,
                                     rail.cx, rail.cy, rail.radius):
                    continue
                result = sat_overlap(rail.obb, body.obb)
                if result is not None:
                    depth, normal = result
                    self._resolve_static(body, depth, normal)

    def _resolve_dynamic(
        self, a: HorseBody, b: HorseBody, depth: float, normal: np.ndarray
    ) -> None:
        a.obb.center -= normal * (depth / 2)
        b.obb.center += normal * (depth / 2)
        rel_vel = float(np.dot(b.velocity - a.velocity, normal))
        if rel_vel >= 0:
            return
        total_mass = a.mass + b.mass
        impulse = -rel_vel / total_mass
        a.velocity -= normal * (impulse * b.mass)
        b.velocity += normal * (impulse * a.mass)

    def _resolve_static(
        self, body: HorseBody, depth: float, normal: np.ndarray
    ) -> None:
        body.obb.center += normal * depth
        vel_along_normal = float(np.dot(body.velocity, normal))
        if vel_along_normal < 0:
            body.velocity -= normal * vel_along_normal
