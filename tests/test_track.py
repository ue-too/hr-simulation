"""Tests for track parsing."""

import math

from horse_racing.track import compute_segment_length, compute_total_length, load_track
from horse_racing.types import CurveSegment, StraightSegment


TEST_OVAL = "tracks/test_oval.json"


def test_load_track_segment_count():
    segments = load_track(TEST_OVAL).segments
    # test_oval: 2 straights + 12 curves per turn × 2 turns = 26
    assert len(segments) == 26


def test_load_track_types():
    segments = load_track(TEST_OVAL).segments
    # First segment is a straight (homestretch)
    assert isinstance(segments[0], StraightSegment)
    # Segments 1-12 are curves (Turn 1)
    for i in range(1, 13):
        assert isinstance(segments[i], CurveSegment)
    # Segment 13 is a straight (backstretch)
    assert isinstance(segments[13], StraightSegment)
    # Segments 14-25 are curves (Turn 2)
    for i in range(14, 26):
        assert isinstance(segments[i], CurveSegment)


def test_straight_segment_length():
    segments = load_track(TEST_OVAL).segments
    length = compute_segment_length(segments[0])
    # Homestretch is 250m
    assert abs(length - 250.0) < 0.01


def test_curve_segment_has_valid_radius():
    segments = load_track(TEST_OVAL).segments
    for seg in segments:
        if isinstance(seg, CurveSegment):
            assert seg.radius >= 25.0, f"Radius {seg.radius} below minimum"


def test_total_length():
    segments = load_track(TEST_OVAL).segments
    total = compute_total_length(segments)
    # test_oval is approximately 905m
    assert abs(total - 905.0) < 1.0


def test_curve_segments_wind_consistently():
    """All curves in the track should wind in the same direction."""
    segments = load_track(TEST_OVAL).segments
    curves = [s for s in segments if isinstance(s, CurveSegment)]
    signs = [s.angle_span > 0 for s in curves]
    assert all(s == signs[0] for s in signs), "Inconsistent winding direction"
