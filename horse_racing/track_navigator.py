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
    n = np.linalg.norm(v)
    if n < 1e-12:
        return _vec2(1.0, 0.0)
    return v / n


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
        self._segment_lengths = [compute_segment_length(s) for s in segments]
        self._total_length = compute_total_length(segments)
        self._cumulative_lengths: list[float] = []
        acc = 0.0
        for l in self._segment_lengths:
            self._cumulative_lengths.append(acc)
            acc += l

    @property
    def total_length(self) -> float:
        return self._total_length

    def reset(self, start_segment: int = 0) -> None:
        self.segment_index = start_segment
        self.entry_radius = float("inf")

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
        normal = _rotate_90_cw(tangential)  # points outward/right
        return TrackFrame(
            tangential=tangential,
            normal=normal,
            turn_radius=float("inf"),
            target_radius=float("inf"),
            segment_index=self.segment_index,
        )

    def _curve_frame(self, seg: CurveSegment, position: np.ndarray) -> TrackFrame:
        to_pos = _vec2(
            position[0] - seg.center[0],
            position[1] - seg.center[1],
        )
        dist = float(np.linalg.norm(to_pos))
        if dist < 1e-6:
            dist = seg.radius

        normal = to_pos / dist  # points outward from center

        if seg.angle_span >= 0:  # CCW
            tangential = _rotate_90_ccw(normal)
        else:  # CW
            tangential = _rotate_90_cw(normal)

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
                to_pos = _vec2(
                    position[0] - next_seg.center[0],
                    position[1] - next_seg.center[1],
                )
                dist = float(np.linalg.norm(to_pos))
                self.entry_radius = max(dist, next_seg.radius - TRACK_HALF_WIDTH)
        else:
            self.entry_radius = float("inf")
