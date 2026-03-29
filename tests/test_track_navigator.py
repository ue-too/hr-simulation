"""Tests for TrackNavigator."""

import math

import numpy as np

from horse_racing.track import load_track
from horse_racing.track_navigator import TrackNavigator


SIMPLE_OVAL = "tracks/test_oval.json"


def test_initial_frame_on_straight():
    segments = load_track(SIMPLE_OVAL).segments
    nav = TrackNavigator(segments)
    pos = np.array([10.0, 0.0])
    frame = nav.compute_frame(pos)

    # On first straight going right (+x direction)
    assert abs(frame.tangential[0] - 1.0) < 0.01
    assert abs(frame.tangential[1]) < 0.01
    assert frame.turn_radius == float("inf")
    assert frame.segment_index == 0


def test_progress_at_start():
    segments = load_track(SIMPLE_OVAL).segments
    nav = TrackNavigator(segments)
    pos = np.array([0.0, 0.0])
    progress = nav.compute_progress(pos)
    assert progress < 0.01


def test_progress_at_midpoint_of_first_straight():
    segments = load_track(SIMPLE_OVAL).segments
    nav = TrackNavigator(segments)
    pos = np.array([150.0, 0.0])
    progress = nav.compute_progress(pos)

    total = nav.total_length
    expected = 150.0 / total
    assert abs(progress - expected) < 0.01


def test_segment_transition():
    segments = load_track(SIMPLE_OVAL).segments
    nav = TrackNavigator(segments)

    # Position past end of first straight
    pos = np.array([310.0, 0.0])
    frame = nav.update(pos)

    # Should have transitioned to segment 1 (curve)
    assert nav.segment_index == 1


def test_curve_frame_has_finite_radius():
    segments = load_track(SIMPLE_OVAL).segments
    nav = TrackNavigator(segments)

    # Move to curve
    nav.segment_index = 1
    nav.entry_radius = float("inf")
    pos = np.array([400.0, 50.0])  # near the curve
    frame = nav.compute_frame(pos)

    assert frame.turn_radius < 1e6
    assert frame.turn_radius > 0
