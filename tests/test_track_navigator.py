import math
from pathlib import Path

import numpy as np
import pytest

from horse_racing.core.track import load_track_json
from horse_racing.core.track_navigator import TrackNavigator
from horse_racing.core.types import TRACK_HALF_WIDTH

TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"


def load_oval():
    return load_track_json(TRACKS_DIR / "test_oval.json")


class TestGetTrackFrame:
    def test_straight_segment_tangential_is_forward(self):
        segments = load_oval()
        nav = TrackNavigator(segments, half_track_width=TRACK_HALF_WIDTH)
        frame = nav.get_track_frame(np.array([100.0, 0.0]))
        assert frame.tangential[0] == pytest.approx(1.0, abs=1e-6)
        assert frame.tangential[1] == pytest.approx(0.0, abs=1e-6)

    def test_straight_segment_turn_radius_is_inf(self):
        segments = load_oval()
        nav = TrackNavigator(segments, half_track_width=TRACK_HALF_WIDTH)
        frame = nav.get_track_frame(np.array([100.0, 0.0]))
        assert frame.turn_radius == math.inf

    def test_curve_segment_has_finite_turn_radius(self):
        segments = load_oval()
        nav = TrackNavigator(segments, half_track_width=TRACK_HALF_WIDTH)
        curves = [i for i, s in enumerate(segments) if hasattr(s, "radius")]
        assert len(curves) > 0
        nav._current_index = curves[0]
        seg = segments[curves[0]]
        pos = np.array([seg.start_point[0], seg.start_point[1]])
        frame = nav.get_track_frame(pos)
        assert frame.turn_radius < 1e6
        assert frame.turn_radius > 0


class TestComputeProgress:
    def test_start_is_zero(self):
        segments = load_oval()
        nav = TrackNavigator(segments, half_track_width=TRACK_HALF_WIDTH)
        progress = nav.compute_progress(np.array([0.0, 0.0]))
        assert progress == pytest.approx(0.0, abs=1e-3)

    def test_midway_through_first_straight(self):
        segments = load_oval()
        nav = TrackNavigator(segments, half_track_width=TRACK_HALF_WIDTH)
        progress = nav.compute_progress(np.array([125.0, 0.0]))
        assert 0.0 < progress < 0.5

    def test_progress_increases_along_track(self):
        segments = load_oval()
        nav = TrackNavigator(segments, half_track_width=TRACK_HALF_WIDTH)
        p1 = nav.compute_progress(np.array([50.0, 0.0]))
        p2 = nav.compute_progress(np.array([200.0, 0.0]))
        assert p2 > p1


class TestUpdateSegment:
    def test_advances_past_first_straight(self):
        segments = load_oval()
        nav = TrackNavigator(segments, half_track_width=TRACK_HALF_WIDTH)
        assert nav.segment_index == 0
        nav.update_segment(np.array([260.0, 0.0]))
        assert nav.segment_index == 1

    def test_stays_on_segment_when_inside(self):
        segments = load_oval()
        nav = TrackNavigator(segments, half_track_width=TRACK_HALF_WIDTH)
        nav.update_segment(np.array([100.0, 0.0]))
        assert nav.segment_index == 0


class TestSampleTrackAhead:
    def test_zero_distance_returns_current_frame(self):
        segments = load_oval()
        nav = TrackNavigator(segments, half_track_width=TRACK_HALF_WIDTH)
        pos = np.array([100.0, 0.0])
        frame_here = nav.get_track_frame(pos)
        frame_ahead = nav.sample_track_ahead(pos, 0)
        assert frame_ahead.slope == pytest.approx(frame_here.slope, abs=1e-6)

    def test_lookahead_past_end_returns_last_segment_frame(self):
        segments = load_oval()
        nav = TrackNavigator(segments, half_track_width=TRACK_HALF_WIDTH)
        pos = np.array([100.0, 0.0])
        frame = nav.sample_track_ahead(pos, 100000)
        assert frame.tangential is not None
