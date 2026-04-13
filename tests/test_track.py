import json
from pathlib import Path

from horse_racing.core.track import (
    CurveSegment,
    StraightSegment,
    TrackSegment,
    load_track_json,
)

TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"


def test_load_test_oval():
    segments = load_track_json(TRACKS_DIR / "test_oval.json")
    assert len(segments) > 0
    first = segments[0]
    assert isinstance(first, StraightSegment)
    assert first.start_point[0] == 0.0
    assert first.start_point[1] == 0.0


def test_straight_segment_fields():
    segments = load_track_json(TRACKS_DIR / "test_oval.json")
    straights = [s for s in segments if isinstance(s, StraightSegment)]
    assert len(straights) > 0
    s = straights[0]
    assert len(s.start_point) == 2
    assert len(s.end_point) == 2
    assert isinstance(s.slope, float)


def test_curve_segment_fields():
    segments = load_track_json(TRACKS_DIR / "test_oval.json")
    curves = [s for s in segments if isinstance(s, CurveSegment)]
    assert len(curves) > 0
    c = curves[0]
    assert len(c.start_point) == 2
    assert len(c.end_point) == 2
    assert len(c.center) == 2
    assert c.radius > 0
    assert c.angle_span != 0
    assert isinstance(c.slope, float)


def test_load_preserves_order():
    path = TRACKS_DIR / "test_oval.json"
    with open(path) as f:
        raw = json.load(f)
    segments = load_track_json(path)
    assert len(segments) == len(raw)
    for seg, r in zip(segments, raw):
        if r["tracktype"] == "STRAIGHT":
            assert isinstance(seg, StraightSegment)
        else:
            assert isinstance(seg, CurveSegment)
