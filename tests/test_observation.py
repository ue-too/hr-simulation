import math
from pathlib import Path

import numpy as np
import pytest

from horse_racing.core.observation import (
    OBS_SIZE,
    OPPONENT_SLOTS,
    OPPONENT_SLOT_SIZE,
    SELF_STATE_SIZE,
    TRACK_CONTEXT_SIZE,
    build_observations,
    curvature,
    normalize_trait,
)
from horse_racing.core.race import Race
from horse_racing.core.track import load_track_json

TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"


def test_obs_size_is_139():
    assert OBS_SIZE == 139

def test_self_state_size():
    assert SELF_STATE_SIZE == 14

def test_track_context_size():
    assert TRACK_CONTEXT_SIZE == 10

def test_opponent_slots():
    assert OPPONENT_SLOTS == 23
    assert OPPONENT_SLOT_SIZE == 5

def test_curvature_straight():
    assert curvature(math.inf) == 0.0
    assert curvature(1e7) == 0.0

def test_curvature_curve():
    assert curvature(100) == pytest.approx(0.01)

def test_normalize_trait():
    assert normalize_trait(13, "cruise_speed") == pytest.approx(0.5)
    assert normalize_trait(8, "cruise_speed") == pytest.approx(0.0)
    assert normalize_trait(18, "cruise_speed") == pytest.approx(1.0)

def test_build_observations_shape():
    segments = load_track_json(TRACKS_DIR / "test_oval.json")
    race = Race(segments, horse_count=4)
    race.start(None)
    obs = build_observations(race)
    assert len(obs) == 4
    for o in obs:
        assert o.shape == (OBS_SIZE,)
        assert o.dtype == np.float64

def test_build_observations_progress_near_zero():
    segments = load_track_json(TRACKS_DIR / "test_oval.json")
    race = Race(segments, horse_count=2)
    race.start(None)
    obs = build_observations(race)
    for o in obs:
        assert abs(o[0]) < 0.01

def test_build_observations_stamina_full():
    segments = load_track_json(TRACKS_DIR / "test_oval.json")
    race = Race(segments, horse_count=2)
    race.start(None)
    obs = build_observations(race)
    for o in obs:
        assert o[3] == pytest.approx(1.0, abs=0.01)

def test_opponent_slots_zero_padded():
    segments = load_track_json(TRACKS_DIR / "test_oval.json")
    race = Race(segments, horse_count=2)
    race.start(None)
    obs = build_observations(race)
    o = obs[0]
    base = SELF_STATE_SIZE + TRACK_CONTEXT_SIZE
    assert o[base + 0] == 1.0  # first opponent active
    for slot in range(1, OPPONENT_SLOTS):
        offset = base + slot * OPPONENT_SLOT_SIZE
        assert o[offset + 0] == 0.0  # inactive
