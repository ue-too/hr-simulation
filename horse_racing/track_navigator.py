"""TrackNavigator — tracks which segment a horse is on and computes the local frame."""

from __future__ import annotations

import math

import numpy as np

from horse_racing.track import compute_segment_length, compute_total_length
from horse_racing.types import (
    TRACK_HALF_WIDTH,
    CurveSegment,
    StraightSegment,
    TrackFrame,
    TrackSegment,
)


def _vec2(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=np.float64)


def _normalize(v: np.ndarray) -> np.ndarray:
    vx, vy = float(v[0]), float(v[1])
    n = math.sqrt(vx * vx + vy * vy)
    if n < 1e-12:
        return _vec2(1.0, 0.0)
    inv_n = 1.0 / n
    return _vec2(vx * inv_n, vy * inv_n)


def _rotate_90_cw(v: np.ndarray) -> np.ndarray:
    """Rotate vector -90 degrees (clockwise)."""
    return _vec2(v[1], -v[0])


def _rotate_90_ccw(v: np.ndarray) -> np.ndarray:
    """Rotate vector +90 degrees (counter-clockwise)."""
    return _vec2(-v[1], v[0])


class TrackNavigator:
    """Per-horse navigator that tracks segment index and computes local frame."""

    def __init__(self, segments: list[TrackSegment]) -> None:
        self.segments = segments
        self.segment_index: int = 0
        self.entry_radius: float = float("inf")  # for curves: radius when entering
        self.completed_lap: bool = False
        self._segment_lengths = [compute_segment_length(s) for s in segments]
        self._total_length = compute_total_length(segments)
        self._cumulative_lengths: list[float] = []
        acc = 0.0
        for l in self._segment_lengths:
            self._cumulative_lengths.append(acc)
            acc += l
        self._outward_normals = self._compute_outward_normals()

    @property
    def total_length(self) -> float:
        return self._total_length

    def reset(self, start_segment: int = 0) -> None:
        self.segment_index = start_segment
        self.entry_radius = float("inf")
        self.completed_lap = False

    def position_at_distance(self, distance: float) -> np.ndarray:
        """Return the centerline position at a given distance along the track."""
        remaining = distance
        for i, seg in enumerate(self.segments):
            seg_len = self._segment_lengths[i]
            if remaining <= seg_len or i == len(self.segments) - 1:
                t = min(remaining / seg_len, 1.0) if seg_len > 1e-6 else 0.0
                if isinstance(seg, StraightSegment):
                    s = _vec2(*seg.start_point)
                    e = _vec2(*seg.end_point)
                    return s + t * (e - s)
                else:
                    # Curve: interpolate angle
                    to_start = _vec2(*seg.start_point) - _vec2(*seg.center)
                    start_angle = math.atan2(to_start[1], to_start[0])
                    angle = start_angle + t * seg.angle_span
                    return _vec2(*seg.center) + seg.radius * _vec2(
                        math.cos(angle), math.sin(angle)
                    )
            remaining -= seg_len
        # Fallback: end of last segment
        last = self.segments[-1]
        return _vec2(*last.end_point)

    def compute_frame(self, position: np.ndarray) -> TrackFrame:
        """Compute the local track frame at the given position."""
        seg = self.segments[self.segment_index]

        if isinstance(seg, StraightSegment):
            return self._straight_frame(seg, position)
        else:
            return self._curve_frame(seg, position)

    def update(self, position: np.ndarray) -> TrackFrame:
        """Check for segment transitions, update state, and return current frame."""
        self._check_transition(position)
        return self.compute_frame(position)

    def compute_progress(self, position: np.ndarray) -> float:
        """Return fraction [0, 1] along total track length."""
        if self._total_length < 1e-6:
            return 0.0

        seg = self.segments[self.segment_index]
        base = self._cumulative_lengths[self.segment_index]

        if isinstance(seg, StraightSegment):
            fwd = _normalize(
                _vec2(
                    seg.end_point[0] - seg.start_point[0],
                    seg.end_point[1] - seg.start_point[1],
                )
            )
            offset = _vec2(
                position[0] - seg.start_point[0],
                position[1] - seg.start_point[1],
            )
            along = float(np.dot(offset, fwd))
            along = max(0.0, min(self._segment_lengths[self.segment_index], along))
        else:
            to_pos = _vec2(
                position[0] - seg.center[0],
                position[1] - seg.center[1],
            )
            angle_pos = math.atan2(to_pos[1], to_pos[0])
            to_start = _vec2(
                seg.start_point[0] - seg.center[0],
                seg.start_point[1] - seg.center[1],
            )
            angle_start = math.atan2(to_start[1], to_start[0])

            delta = angle_pos - angle_start
            if seg.angle_span >= 0:
                while delta < 0:
                    delta += 2 * math.pi
                while delta > 2 * math.pi:
                    delta -= 2 * math.pi
            else:
                while delta > 0:
                    delta -= 2 * math.pi
                while delta < -2 * math.pi:
                    delta += 2 * math.pi

            along = abs(delta) * seg.radius
            along = max(0.0, min(self._segment_lengths[self.segment_index], along))

        return (base + along) / self._total_length

    def lookahead_segments(self, count: int = 3) -> list[TrackSegment]:
        """Return the next `count` segments starting from current."""
        result = []
        for i in range(count):
            idx = self.segment_index + i
            if idx < len(self.segments):
                result.append(self.segments[idx])
            else:
                result.append(self.segments[-1])
        return result

    # ------------------------------------------------------------------
    # Frame computation
    # ------------------------------------------------------------------

    def _straight_frame(self, seg: StraightSegment, position: np.ndarray) -> TrackFrame:
        tangential = _normalize(
            _vec2(
                seg.end_point[0] - seg.start_point[0],
                seg.end_point[1] - seg.start_point[1],
            )
        )
        normal = self._outward_normals[self.segment_index]  # points outward (away from track interior)
        return TrackFrame(
            tangential=tangential,
            normal=normal,
            turn_radius=float("inf"),
            target_radius=float("inf"),
            segment_index=self.segment_index,
            slope=seg.slope,
        )

    def _curve_frame(self, seg: CurveSegment, position: np.ndarray) -> TrackFrame:
        tpx = float(position[0]) - seg.center[0]
        tpy = float(position[1]) - seg.center[1]
        dist = math.sqrt(tpx * tpx + tpy * tpy)
        if dist < 1e-6:
            dist = seg.radius

        inv_dist = 1.0 / dist
        nx, ny = tpx * inv_dist, tpy * inv_dist  # normal: outward from center

        if seg.angle_span >= 0:  # CCW
            tangential = _vec2(-ny, nx)
        else:  # CW
            tangential = _vec2(ny, -nx)

        normal = _vec2(nx, ny)
        turn_radius = dist

        # Capture entry radius on first visit to this curve
        if self.entry_radius == float("inf"):
            self.entry_radius = max(dist, seg.radius - TRACK_HALF_WIDTH)

        target_radius = self.entry_radius

        return TrackFrame(
            tangential=tangential,
            normal=normal,
            turn_radius=turn_radius,
            target_radius=target_radius,
            segment_index=self.segment_index,
            slope=seg.slope,
        )

    # ------------------------------------------------------------------
    # Segment transitions
    # ------------------------------------------------------------------

    def _check_transition(self, position: np.ndarray) -> None:
        seg = self.segments[self.segment_index]

        if isinstance(seg, StraightSegment):
            self._check_straight_exit(seg, position)
        else:
            self._check_curve_exit(seg, position)

    def _check_straight_exit(self, seg: StraightSegment, position: np.ndarray) -> None:
        fwd = _normalize(
            _vec2(
                seg.end_point[0] - seg.start_point[0],
                seg.end_point[1] - seg.start_point[1],
            )
        )
        to_end = _vec2(
            position[0] - seg.end_point[0],
            position[1] - seg.end_point[1],
        )
        if float(np.dot(to_end, fwd)) > 0:
            self._advance_segment(position)

    def _check_curve_exit(self, seg: CurveSegment, position: np.ndarray) -> None:
        to_pos = _vec2(
            position[0] - seg.center[0],
            position[1] - seg.center[1],
        )
        angle_pos = math.atan2(to_pos[1], to_pos[0])

        to_end = _vec2(
            seg.end_point[0] - seg.center[0],
            seg.end_point[1] - seg.center[1],
        )
        angle_end = math.atan2(to_end[1], to_end[0])

        # Signed angle from end direction to horse direction
        delta = angle_pos - angle_end
        # Normalize to [-pi, pi]
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi

        # Exit condition: delta crosses zero in the direction of travel
        if seg.angle_span >= 0 and delta > 0:
            self._advance_segment(position)
        elif seg.angle_span < 0 and delta < 0:
            self._advance_segment(position)

    def _advance_segment(self, position: np.ndarray) -> None:
        next_idx = self.segment_index + 1
        if next_idx >= len(self.segments):
            self.completed_lap = True
            return  # stay on last segment (race is over)

        prev_seg = self.segments[self.segment_index]
        next_seg = self.segments[next_idx]

        self.segment_index = next_idx

        # Handle entry radius for curves
        if isinstance(next_seg, CurveSegment):
            if isinstance(prev_seg, CurveSegment):
                # Carry lane offset from previous curve
                lane_offset = self.entry_radius - prev_seg.radius
                self.entry_radius = max(
                    next_seg.radius + lane_offset,
                    next_seg.radius - TRACK_HALF_WIDTH,
                )
            else:
                # Coming from straight — capture actual distance
                dx = float(position[0]) - next_seg.center[0]
                dy = float(position[1]) - next_seg.center[1]
                dist = math.sqrt(dx * dx + dy * dy)
                self.entry_radius = max(dist, next_seg.radius - TRACK_HALF_WIDTH)
        else:
            self.entry_radius = float("inf")

    # ------------------------------------------------------------------
    # Outward normals for straights
    # ------------------------------------------------------------------

    def _compute_outward_normals(self) -> list[np.ndarray | None]:
        """Precompute outward normal for each segment.

        For curves: None (computed dynamically by _curve_frame).
        For straights: determined by nearest curve center — outward = away from it.
        """
        normals: list[np.ndarray | None] = []
        for i, seg in enumerate(self.segments):
            if isinstance(seg, CurveSegment):
                normals.append(None)
                continue
            start = _vec2(*seg.start_point)
            end = _vec2(*seg.end_point)
            fwd = _normalize(end - start)
            normal = _rotate_90_cw(fwd)

            curve = self._find_nearest_curve(i)
            if curve is not None:
                mid = (start + end) / 2
                to_center = _vec2(*curve.center) - mid
                if float(np.dot(to_center, normal)) > 0:
                    normal = -normal  # flip so it points outward
            normals.append(normal)
        return normals

    def _find_nearest_curve(self, idx: int) -> CurveSegment | None:
        """Find the nearest curve segment to the given index."""
        for j in range(idx + 1, len(self.segments)):
            if isinstance(self.segments[j], CurveSegment):
                return self.segments[j]
        for j in range(idx - 1, -1, -1):
            if isinstance(self.segments[j], CurveSegment):
                return self.segments[j]
        return None
