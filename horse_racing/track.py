"""Track JSON parsing — loads camelCase track files into typed segment dataclasses."""

from __future__ import annotations

import json
import math
from pathlib import Path

from horse_racing.types import CurveSegment, StraightSegment, TrackSegment


def load_track(path: str | Path) -> list[TrackSegment]:
    """Load a track JSON file and return a list of TrackSegment objects."""
    with open(path) as f:
        data = json.load(f)

    segments: list[TrackSegment] = []
    for raw in data:
        seg = parse_segment(raw)
        segments.append(seg)
    return segments


def parse_segment(raw: dict) -> TrackSegment:
    """Parse a single segment dict (camelCase keys) into a typed dataclass."""
    tracktype = raw["tracktype"]
    start_point = _parse_point(raw["startPoint"])
    end_point = _parse_point(raw["endPoint"])

    slope = float(raw.get("slope", 0.0))

    if tracktype == "STRAIGHT":
        return StraightSegment(
            tracktype="STRAIGHT",
            start_point=start_point,
            end_point=end_point,
            slope=slope,
        )
    elif tracktype == "CURVE":
        center = _parse_point(raw["center"])
        radius = float(raw["radius"])
        angle_span = float(raw["angleSpan"])
        return CurveSegment(
            tracktype="CURVE",
            start_point=start_point,
            end_point=end_point,
            center=center,
            radius=radius,
            angle_span=angle_span,
            slope=slope,
        )
    else:
        raise ValueError(f"Unknown track segment type: {tracktype}")


def _parse_point(p: dict | list) -> tuple[float, float]:
    if isinstance(p, dict):
        return (float(p["x"]), float(p["y"]))
    return (float(p[0]), float(p[1]))


def compute_segment_length(seg: TrackSegment) -> float:
    """Compute arc/line length of a segment."""
    if isinstance(seg, StraightSegment):
        dx = seg.end_point[0] - seg.start_point[0]
        dy = seg.end_point[1] - seg.start_point[1]
        return math.hypot(dx, dy)
    else:
        return abs(seg.angle_span) * seg.radius


def compute_total_length(segments: list[TrackSegment]) -> float:
    return sum(compute_segment_length(s) for s in segments)
