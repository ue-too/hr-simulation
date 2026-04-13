import math
from pathlib import Path

import numpy as np
import pytest

from horse_racing.core.collision import (
    CollisionWorld,
    OBB,
    sat_overlap,
)
from horse_racing.core.track import load_track_json
from horse_racing.core.track_navigator import TrackFrame
from horse_racing.core.types import HORSE_HALF_LENGTH, HORSE_HALF_WIDTH, TRACK_HALF_WIDTH

TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"


def straight_frame() -> TrackFrame:
    return TrackFrame(
        tangential=np.array([1.0, 0.0]),
        normal=np.array([0.0, -1.0]),
        turn_radius=math.inf,
        nominal_radius=math.inf,
        target_radius=math.inf,
        slope=0.0,
    )


class TestOBB:
    def test_vertices_axis_aligned(self):
        obb = OBB(center=np.array([0.0, 0.0]), half_length=1.0, half_width=0.5, angle=0.0)
        verts = obb.vertices()
        assert len(verts) == 4
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        assert max(xs) == pytest.approx(1.0)
        assert min(xs) == pytest.approx(-1.0)
        assert max(ys) == pytest.approx(0.5)
        assert min(ys) == pytest.approx(-0.5)

    def test_vertices_rotated_90(self):
        obb = OBB(center=np.array([0.0, 0.0]), half_length=1.0, half_width=0.5, angle=math.pi / 2)
        verts = obb.vertices()
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        assert max(ys) == pytest.approx(1.0, abs=1e-6)
        assert max(xs) == pytest.approx(0.5, abs=1e-6)


class TestSATOverlap:
    def test_overlapping_boxes(self):
        a = OBB(np.array([0.0, 0.0]), 1.0, 0.5, 0.0)
        b = OBB(np.array([1.5, 0.0]), 1.0, 0.5, 0.0)
        result = sat_overlap(a, b)
        assert result is not None
        depth, normal = result
        assert depth > 0

    def test_separated_boxes(self):
        a = OBB(np.array([0.0, 0.0]), 1.0, 0.5, 0.0)
        b = OBB(np.array([5.0, 0.0]), 1.0, 0.5, 0.0)
        result = sat_overlap(a, b)
        assert result is None

    def test_touching_boxes(self):
        a = OBB(np.array([0.0, 0.0]), 1.0, 0.5, 0.0)
        b = OBB(np.array([2.0, 0.0]), 1.0, 0.5, 0.0)
        result = sat_overlap(a, b)
        assert result is None or result[0] == pytest.approx(0.0, abs=1e-6)


class TestCollisionWorld:
    def test_create_from_track(self):
        segments = load_track_json(TRACKS_DIR / "test_oval.json")
        world = CollisionWorld(segments, TRACK_HALF_WIDTH)
        assert world is not None

    def test_add_and_get_horse(self):
        segments = load_track_json(TRACKS_DIR / "test_oval.json")
        world = CollisionWorld(segments, TRACK_HALF_WIDTH)
        pos = np.array([10.0, 0.0])
        vel = np.array([5.0, 0.0])
        frame = straight_frame()
        world.add_horse(0, pos, frame, mass=500.0)
        world.set_horse_state(0, pos, vel, frame)
        world.step(1 / 240)
        got_pos, got_vel = world.get_horse_state(0)
        assert got_pos is not None
        assert got_vel is not None

    def test_two_horses_collide_and_separate(self):
        segments = load_track_json(TRACKS_DIR / "test_oval.json")
        world = CollisionWorld(segments, TRACK_HALF_WIDTH)
        frame = straight_frame()
        world.add_horse(0, np.array([10.0, 0.0]), frame, mass=500.0)
        world.add_horse(1, np.array([11.5, 0.0]), frame, mass=500.0)
        world.set_horse_state(0, np.array([10.0, 0.0]), np.array([5.0, 0.0]), frame)
        world.set_horse_state(1, np.array([11.5, 0.0]), np.array([0.0, 0.0]), frame)
        world.step(1 / 240)
        pos0, _ = world.get_horse_state(0)
        pos1, _ = world.get_horse_state(1)
        gap = np.linalg.norm(pos1 - pos0)
        assert gap >= HORSE_HALF_LENGTH * 2 - 0.1
