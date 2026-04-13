"""Track segment types and JSON loading — mirrors TS track-types.ts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np


@dataclass(frozen=True)
class StraightSegment:
    start_point: np.ndarray  # shape (2,)
    end_point: np.ndarray    # shape (2,)
    slope: float = 0.0


@dataclass(frozen=True)
class CurveSegment:
    start_point: np.ndarray  # shape (2,)
    end_point: np.ndarray    # shape (2,)
    center: np.ndarray       # shape (2,)
    radius: float = 0.0
    angle_span: float = 0.0
    slope: float = 0.0


TrackSegment = Union[StraightSegment, CurveSegment]


def _point(raw: dict) -> np.ndarray:
    return np.array([raw["x"], raw["y"]], dtype=np.float64)


def load_track_json(path: str | Path) -> list[TrackSegment]:
    """Load track segments from a JSON file."""
    with open(path) as f:
        raw_list = json.load(f)

    segments: list[TrackSegment] = []
    for raw in raw_list:
        slope = float(raw.get("slope", 0.0))
        if raw["tracktype"] == "STRAIGHT":
            segments.append(StraightSegment(
                start_point=_point(raw["startPoint"]),
                end_point=_point(raw["endPoint"]),
                slope=slope,
            ))
        else:
            segments.append(CurveSegment(
                start_point=_point(raw["startPoint"]),
                end_point=_point(raw["endPoint"]),
                center=_point(raw["center"]),
                radius=float(raw["radius"]),
                angle_span=float(raw["angleSpan"]),
                slope=slope,
            ))
    return segments
