"""Track JSON parsing — loads camelCase track files into typed segment dataclasses."""

from __future__ import annotations

import json
import math
from pathlib import Path

from horse_racing.types import (
    TRACK_HALF_WIDTH,
    CurveSegment,
    StraightSegment,
    TrackData,
    TrackSegment,
)


def load_track(path: str | Path) -> TrackData:
    """Load a track JSON file and return a TrackData object.

    Supports two formats:
    - Legacy: bare JSON array of segments (rails auto-generated from TRACK_HALF_WIDTH)
    - New: JSON object with "segments", optional "inner_rails" and "outer_rails"
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        segments = _parse_segment_list(data)
        inner_rails, outer_rails = _generate_rails_from_centerline(segments)
        return TrackData(
            segments=segments,
            inner_rails=inner_rails,
            outer_rails=outer_rails,
        )

    if isinstance(data, dict) and "segments" in data:
        segments = _parse_segment_list(data["segments"])
        if "inner_rails" in data and "outer_rails" in data:
            inner_rails = _parse_segment_list(data["inner_rails"])
            outer_rails = _parse_segment_list(data["outer_rails"])
        else:
            inner_rails, outer_rails = _generate_rails_from_centerline(segments)
        return TrackData(
            segments=segments,
            inner_rails=inner_rails,
            outer_rails=outer_rails,
        )

    raise ValueError("Track JSON must be an array or an object with a 'segments' key")


def _parse_segment_list(raw_list: list) -> list[TrackSegment]:
    segments: list[TrackSegment] = []
    for raw in raw_list:
        segments.append(parse_segment(raw))
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


# ---------------------------------------------------------------------------
# Rail auto-generation from centerline
# ---------------------------------------------------------------------------


def _generate_rails_from_centerline(
    segments: list[TrackSegment],
    half_width: float = TRACK_HALF_WIDTH,
) -> tuple[list[TrackSegment], list[TrackSegment]]:
    """Generate inner and outer rail paths by offsetting the centerline."""
    inner_rails: list[TrackSegment] = []
    outer_rails: list[TrackSegment] = []

    for seg in segments:
        if isinstance(seg, StraightSegment):
            inner, outer = _offset_straight(seg, half_width)
        else:
            inner, outer = _offset_curve(seg, half_width)
        inner_rails.append(inner)
        outer_rails.append(outer)

    return inner_rails, outer_rails


def _offset_straight(
    seg: StraightSegment, half_width: float
) -> tuple[TrackSegment, TrackSegment]:
    """Offset a straight segment laterally to create inner and outer rail segments."""
    sx, sy = seg.start_point
    ex, ey = seg.end_point
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return seg, seg

    # Outward direction (forward rotated -90 degrees)
    nx, ny = dy / length, -dx / length

    inner = StraightSegment(
        tracktype="STRAIGHT",
        start_point=(sx - nx * half_width, sy - ny * half_width),
        end_point=(ex - nx * half_width, ey - ny * half_width),
        slope=seg.slope,
    )
    outer = StraightSegment(
        tracktype="STRAIGHT",
        start_point=(sx + nx * half_width, sy + ny * half_width),
        end_point=(ex + nx * half_width, ey + ny * half_width),
        slope=seg.slope,
    )
    return inner, outer


def _offset_curve(
    seg: CurveSegment, half_width: float
) -> tuple[TrackSegment, TrackSegment]:
    """Create parallel arcs at radius ± half_width for inner and outer rails."""
    cx, cy = seg.center
    inner_radius = max(seg.radius - half_width, 0.1)
    outer_radius = seg.radius + half_width

    # Recompute start/end points on the offset circles
    # Start angle from center to start_point
    start_angle = math.atan2(
        seg.start_point[1] - cy, seg.start_point[0] - cx
    )
    end_angle = start_angle + seg.angle_span

    inner = CurveSegment(
        tracktype="CURVE",
        start_point=(
            cx + inner_radius * math.cos(start_angle),
            cy + inner_radius * math.sin(start_angle),
        ),
        end_point=(
            cx + inner_radius * math.cos(end_angle),
            cy + inner_radius * math.sin(end_angle),
        ),
        center=seg.center,
        radius=inner_radius,
        angle_span=seg.angle_span,
        slope=seg.slope,
    )
    outer = CurveSegment(
        tracktype="CURVE",
        start_point=(
            cx + outer_radius * math.cos(start_angle),
            cy + outer_radius * math.sin(start_angle),
        ),
        end_point=(
            cx + outer_radius * math.cos(end_angle),
            cy + outer_radius * math.sin(end_angle),
        ),
        center=seg.center,
        radius=outer_radius,
        angle_span=seg.angle_span,
        slope=seg.slope,
    )
    return inner, outer


# ---------------------------------------------------------------------------
# Bounding box utilities for rail collision spatial culling
# ---------------------------------------------------------------------------


def compute_rail_bboxes(
    rail_segments: list[TrackSegment],
) -> list[tuple[float, float, float, float]]:
    """Compute axis-aligned bounding boxes for each rail segment.

    Returns list of (min_x, min_y, max_x, max_y) tuples.
    """
    bboxes = []
    for seg in rail_segments:
        if isinstance(seg, StraightSegment):
            x0, y0 = seg.start_point
            x1, y1 = seg.end_point
            bboxes.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
        else:
            # For curves, use center ± radius as conservative bbox
            cx, cy = seg.center
            r = seg.radius
            bboxes.append((cx - r, cy - r, cx + r, cy + r))
    return bboxes


# ---------------------------------------------------------------------------
# Segment length computation
# ---------------------------------------------------------------------------


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
