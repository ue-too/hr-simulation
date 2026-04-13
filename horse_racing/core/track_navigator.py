"""Track navigation — mirrors TS track-navigator.ts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .track import CurveSegment, StraightSegment, TrackSegment


@dataclass
class TrackFrame:
    tangential: np.ndarray  # shape (2,), unit vector
    normal: np.ndarray      # shape (2,), unit vector
    turn_radius: float
    nominal_radius: float
    target_radius: float
    slope: float


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0])
    return v / n


def _rotate(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _angle_a2b(a: np.ndarray, b: np.ndarray) -> float:
    """Signed angle from unit vector a to unit vector b."""
    cross = a[0] * b[1] - a[1] * b[0]
    dot = a[0] * b[0] + a[1] * b[1]
    return math.atan2(cross, dot)


class TrackNavigator:
    def __init__(
        self,
        segments: list[TrackSegment],
        start_index: int = 0,
        half_track_width: float = 15.0,
    ):
        self._segments = segments
        self._current_index = start_index
        self._half_track_width = half_track_width
        self._curve_entry_radius = float("nan")
        self._completed_lap = False

        self._outward_normals: list[np.ndarray | None] = self._compute_outward_normals()

        self._segment_lengths: list[float] = []
        for seg in segments:
            if isinstance(seg, StraightSegment):
                self._segment_lengths.append(float(np.linalg.norm(seg.end_point - seg.start_point)))
            else:
                self._segment_lengths.append(abs(seg.angle_span) * seg.radius)

        self._cumulative_lengths: list[float] = []
        acc = 0.0
        for length in self._segment_lengths:
            self._cumulative_lengths.append(acc)
            acc += length
        self._total_length = acc

    @property
    def segment_index(self) -> int:
        return self._current_index

    @property
    def completed_lap(self) -> bool:
        return self._completed_lap

    @property
    def segment(self) -> TrackSegment:
        return self._segments[self._current_index]

    @property
    def total_length(self) -> float:
        return self._total_length

    @property
    def target_radius(self) -> float:
        seg = self.segment
        if isinstance(seg, CurveSegment) and not math.isnan(self._curve_entry_radius):
            return self._curve_entry_radius
        return math.inf

    def compute_progress(self, position: np.ndarray) -> float:
        if self._total_length < 1e-6:
            return 0.0

        seg = self._segments[self._current_index]
        base = self._cumulative_lengths[self._current_index]
        seg_len = self._segment_lengths[self._current_index]

        if isinstance(seg, StraightSegment):
            d = seg.end_point - seg.start_point
            length = np.linalg.norm(d)
            if length < 1e-6:
                along = 0.0
            else:
                fwd = d / length
                off = position - seg.start_point
                along = float(np.dot(off, fwd))
        else:
            to_pos = position - seg.center
            angle_pos = math.atan2(to_pos[1], to_pos[0])
            to_start = seg.start_point - seg.center
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

        along = max(0.0, min(seg_len, along))
        return (base + along) / self._total_length

    def get_track_frame(self, position: np.ndarray) -> TrackFrame:
        seg = self.segment
        if isinstance(seg, CurveSegment):
            return self._curve_frame(seg, position)
        return self._straight_frame(seg)

    def update_segment(self, position: np.ndarray) -> None:
        if len(self._segments) <= 1:
            return

        seg = self.segment
        if isinstance(seg, StraightSegment):
            exited = self._exited_straight(seg, position)
        else:
            exited = self._exited_curve(seg, position)

        if exited:
            if self._current_index == len(self._segments) - 1:
                self._completed_lap = True
                return
            prev_seg = seg
            self._current_index += 1
            new_seg = self.segment

            if isinstance(new_seg, CurveSegment):
                inner_rail = new_seg.radius - self._half_track_width
                if isinstance(prev_seg, CurveSegment) and not math.isnan(self._curve_entry_radius):
                    lane_offset = self._curve_entry_radius - prev_seg.radius
                    self._curve_entry_radius = max(new_seg.radius + lane_offset, inner_rail)
                else:
                    raw_radius = float(np.linalg.norm(position - new_seg.center))
                    self._curve_entry_radius = max(raw_radius, inner_rail)
            else:
                self._curve_entry_radius = float("nan")

    def sample_track_ahead(self, position: np.ndarray, distance: float) -> TrackFrame:
        seg = self._segments[self._current_index]
        along = self._distance_along_segment(seg, position)

        if distance <= 0:
            clamped = max(0.0, min(self._segment_lengths[self._current_index], along))
            return self._frame_at_segment_offset(self._current_index, clamped)

        remaining = distance + along
        idx = self._current_index

        while idx < len(self._segments):
            seg_len = self._segment_lengths[idx]
            if remaining <= seg_len:
                return self._frame_at_segment_offset(idx, remaining)
            remaining -= seg_len
            idx += 1

        last_idx = len(self._segments) - 1
        return self._frame_at_segment_offset(last_idx, self._segment_lengths[last_idx])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_outward_normals(self) -> list[np.ndarray | None]:
        result: list[np.ndarray | None] = []
        for i, seg in enumerate(self._segments):
            if isinstance(seg, CurveSegment):
                result.append(None)
                continue
            d = seg.end_point - seg.start_point
            length = np.linalg.norm(d)
            if length < 1e-6:
                result.append(np.array([0.0, -1.0]))
                continue
            nx = d[1] / length
            ny = -d[0] / length
            curve = self._find_nearest_curve(i)
            if curve is not None:
                mx = (seg.start_point[0] + seg.end_point[0]) / 2
                my = (seg.start_point[1] + seg.end_point[1]) / 2
                to_center_dot = (curve.center[0] - mx) * nx + (curve.center[1] - my) * ny
                if to_center_dot > 0:
                    nx = -nx
                    ny = -ny
            result.append(np.array([nx, ny]))
        return result

    def _find_nearest_curve(self, idx: int) -> CurveSegment | None:
        for j in range(idx + 1, len(self._segments)):
            if isinstance(self._segments[j], CurveSegment):
                return self._segments[j]
        for j in range(idx - 1, -1, -1):
            if isinstance(self._segments[j], CurveSegment):
                return self._segments[j]
        return None

    def _straight_frame(self, seg: StraightSegment) -> TrackFrame:
        tangential = _unit(seg.end_point - seg.start_point)
        curve = self._find_nearest_curve(self._current_index)
        rotation = math.pi / 2 if (curve and curve.angle_span < 0) else -math.pi / 2
        normal = _unit(_rotate(tangential, rotation))
        return TrackFrame(
            tangential=tangential,
            normal=normal,
            turn_radius=math.inf,
            nominal_radius=math.inf,
            target_radius=math.inf,
            slope=seg.slope,
        )

    def _curve_frame(self, seg: CurveSegment, position: np.ndarray) -> TrackFrame:
        radial = position - seg.center
        turn_radius = float(np.linalg.norm(radial))
        normal = _unit(radial) if turn_radius > 1e-6 else np.array([1.0, 0.0])

        rot = math.pi / 2 if seg.angle_span >= 0 else -math.pi / 2
        tangential = _unit(_rotate(normal, rot))

        if math.isnan(self._curve_entry_radius):
            self._curve_entry_radius = max(
                turn_radius, seg.radius - self._half_track_width
            )

        return TrackFrame(
            tangential=tangential,
            normal=normal,
            turn_radius=turn_radius,
            nominal_radius=seg.radius,
            target_radius=seg.radius,
            slope=seg.slope,
        )

    def _distance_along_segment(self, seg: TrackSegment, position: np.ndarray) -> float:
        if isinstance(seg, StraightSegment):
            d = seg.end_point - seg.start_point
            length = np.linalg.norm(d)
            if length < 1e-6:
                return 0.0
            fwd = d / length
            off = position - seg.start_point
            along = float(np.dot(off, fwd))
            return max(0.0, min(float(length), along))

        to_pos = position - seg.center
        angle_pos = math.atan2(to_pos[1], to_pos[0])
        to_start = seg.start_point - seg.center
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
        seg_len = abs(seg.angle_span) * seg.radius
        return max(0.0, min(seg_len, abs(delta) * seg.radius))

    def _frame_at_segment_offset(self, seg_idx: int, offset: float) -> TrackFrame:
        seg = self._segments[seg_idx]

        if isinstance(seg, StraightSegment):
            tangential = _unit(seg.end_point - seg.start_point)
            curve = self._find_nearest_curve(seg_idx)
            rotation = math.pi / 2 if (curve and curve.angle_span < 0) else -math.pi / 2
            normal = _unit(_rotate(tangential, rotation))
            return TrackFrame(
                tangential=tangential,
                normal=normal,
                turn_radius=math.inf,
                nominal_radius=math.inf,
                target_radius=math.inf,
                slope=seg.slope,
            )

        to_start = seg.start_point - seg.center
        angle_start = math.atan2(to_start[1], to_start[0])
        angle_at_offset = angle_start + (offset / seg.radius) * (1.0 if seg.angle_span >= 0 else -1.0)

        normal = np.array([math.cos(angle_at_offset), math.sin(angle_at_offset)])
        rot = math.pi / 2 if seg.angle_span >= 0 else -math.pi / 2
        tangential = _unit(_rotate(normal, rot))

        return TrackFrame(
            tangential=tangential,
            normal=normal,
            turn_radius=seg.radius,
            nominal_radius=seg.radius,
            target_radius=seg.radius,
            slope=seg.slope,
        )

    def _exited_straight(self, seg: StraightSegment, position: np.ndarray) -> bool:
        forward = _unit(seg.end_point - seg.start_point)
        to_end = position - seg.end_point
        return float(np.dot(to_end, forward)) > 0

    def _exited_curve(self, seg: CurveSegment, position: np.ndarray) -> bool:
        end_dir = _unit(seg.end_point - seg.center)
        horse_dir = _unit(position - seg.center)
        angle = _angle_a2b(end_dir, horse_dir)
        if seg.angle_span >= 0:
            return angle > 0
        return angle < 0
