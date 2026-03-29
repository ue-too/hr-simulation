"""Tests for track parsing."""

import math

from horse_racing.track import compute_segment_length, compute_total_length, load_track
from horse_racing.types import CurveSegment, StraightSegment


SIMPLE_OVAL = "tracks/test_oval.json"


def test_load_track_segment_count():
    segments = load_track(SIMPLE_OVAL).segments
    assert len(segments) == 4


def test_load_track_types():
    segments = load_track(SIMPLE_OVAL).segments
    assert isinstance(segments[0], StraightSegment)
    assert isinstance(segments[1], CurveSegment)
    assert isinstance(segments[2], StraightSegment)
    assert isinstance(segments[3], CurveSegment)


def test_straight_segment_length():
    segments = load_track(SIMPLE_OVAL).segments
    length = compute_segment_length(segments[0])
    assert abs(length - 300.0) < 0.01


def test_curve_segment_length():
    segments = load_track(SIMPLE_OVAL).segments
    # semicircle with radius 100 → pi * 100 ≈ 314.16
    length = compute_segment_length(segments[1])
    assert abs(length - math.pi * 100) < 0.1


def test_total_length():
    segments = load_track(SIMPLE_OVAL).segments
    total = compute_total_length(segments)
    expected = 300 * 2 + math.pi * 100 * 2  # two straights + two semicircles
    assert abs(total - expected) < 0.1


def test_curve_segment_attributes():
    segments = load_track(SIMPLE_OVAL).segments
    curve = segments[1]
    assert isinstance(curve, CurveSegment)
    assert curve.radius == 100
    assert abs(curve.angle_span - (-math.pi)) < 0.001
    assert curve.center == (300, 100)
