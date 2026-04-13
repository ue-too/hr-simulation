from pathlib import Path

import numpy as np
import pytest

from horse_racing.core.race import Race, spawn_horses
from horse_racing.core.track import load_track_json
from horse_racing.core.types import InputState, MAX_HORSES, TRACK_HALF_WIDTH

TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"


def load_oval():
    return load_track_json(TRACKS_DIR / "test_oval.json")


class TestSpawnHorses:
    def test_default_count(self):
        horses = spawn_horses(load_oval())
        assert len(horses) == 4

    def test_custom_count(self):
        horses = spawn_horses(load_oval(), horse_count=8)
        assert len(horses) == 8

    def test_clamps_to_max(self):
        horses = spawn_horses(load_oval(), horse_count=50)
        assert len(horses) == MAX_HORSES

    def test_clamps_to_min(self):
        horses = spawn_horses(load_oval(), horse_count=0)
        assert len(horses) == 1

    def test_unique_ids(self):
        horses = spawn_horses(load_oval(), horse_count=6)
        assert len(set(h.id for h in horses)) == 6

    def test_start_at_track_start(self):
        horses = spawn_horses(load_oval(), horse_count=4)
        for h in horses:
            assert h.pos[0] == pytest.approx(0.0, abs=20)

    def test_full_stamina(self):
        horses = spawn_horses(load_oval())
        for h in horses:
            assert h.current_stamina == h.base_attributes.max_stamina


class TestRace:
    def test_initial_phase(self):
        race = Race(load_oval(), horse_count=4)
        assert race.state.phase == "gate"

    def test_start(self):
        race = Race(load_oval(), horse_count=4)
        race.start(None)
        assert race.state.phase == "running"

    def test_tick_no_op_in_gate(self):
        race = Race(load_oval(), horse_count=4)
        race.tick({})
        assert race.state.tick == 0

    def test_tick_increments(self):
        race = Race(load_oval(), horse_count=4)
        race.start(None)
        race.tick({})
        assert race.state.tick == 1

    def test_horses_move(self):
        race = Race(load_oval(), horse_count=2)
        race.start(None)
        initial = [h.track_progress for h in race.state.horses]
        for _ in range(60):
            race.tick({})
        for i, h in enumerate(race.state.horses):
            assert h.track_progress > initial[i]

    def test_reset(self):
        race = Race(load_oval(), horse_count=4)
        race.start(None)
        for _ in range(10):
            race.tick({})
        race.reset()
        assert race.state.phase == "gate"
        assert race.state.tick == 0

    def test_finish(self):
        race = Race(load_oval(), horse_count=2)
        race.start(None)
        push = {0: InputState(1.0, 0.0), 1: InputState(1.0, 0.0)}
        for _ in range(5000):
            if race.state.phase == "finished":
                break
            race.tick(push)
        assert race.state.phase == "finished"
        assert len(race.state.finish_order) > 0
