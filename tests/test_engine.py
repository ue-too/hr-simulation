"""Tests for the horse racing engine."""

import numpy as np

from horse_racing.engine import EngineConfig, HorseRacingEngine
from horse_racing.types import HORSE_COUNT, HorseAction


SIMPLE_OVAL = "tracks/simple_oval.json"


def test_engine_creates_horses():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    assert len(engine.horses) == HORSE_COUNT


def test_engine_initial_positions_differ():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    positions = [hs.body.position.copy() for hs in engine.horses]
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            assert dist > 1.0, "Horses should start at different positions"


def test_engine_step_zero_actions():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    actions = [HorseAction() for _ in range(HORSE_COUNT)]

    initial_positions = [hs.body.position.copy() for hs in engine.horses]
    engine.step(actions)
    final_positions = [hs.body.position.copy() for hs in engine.horses]

    # Horses should have moved (auto-cruise kicks in)
    for i in range(HORSE_COUNT):
        dist = float(np.linalg.norm(final_positions[i] - initial_positions[i]))
        assert dist > 0.0, f"Horse {i} did not move"


def test_engine_step_forward_action():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    actions = [HorseAction(extra_tangential=5.0) for _ in range(HORSE_COUNT)]
    engine.step(actions)

    # Horses should have positive forward velocity
    for hs in engine.horses:
        if hs.frame is not None:
            tang_vel = float(np.dot(hs.body.velocity, hs.frame.tangential))
            assert tang_vel > 0, "Horse should be moving forward"


def test_engine_observations():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    actions = [HorseAction() for _ in range(HORSE_COUNT)]
    engine.step(actions)

    obs = engine.get_observations()
    assert len(obs) == HORSE_COUNT

    for o in obs:
        assert "tangential_vel" in o
        assert "track_progress" in o
        assert "stamina_ratio" in o


def test_engine_obs_to_array():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    actions = [HorseAction() for _ in range(HORSE_COUNT)]
    engine.step(actions)

    obs = engine.get_observations()
    arr = engine.obs_to_array(obs[0])
    assert arr.shape == (18,)
    assert arr.dtype == np.float32


def test_engine_reset():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    actions = [HorseAction(extra_tangential=5.0) for _ in range(HORSE_COUNT)]

    # Run a few steps
    for _ in range(10):
        engine.step(actions)

    # Reset
    engine.reset()
    assert engine.tick == 0
    for hs in engine.horses:
        assert hs.track_progress == 0.0
        assert not hs.finished


def test_engine_progress_increases():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    actions = [HorseAction(extra_tangential=3.0) for _ in range(HORSE_COUNT)]

    for _ in range(50):
        engine.step(actions)

    for hs in engine.horses:
        assert hs.track_progress > 0.0, "Progress should increase after running"


def test_engine_custom_horse_count():
    config = EngineConfig(horse_count=2)
    engine = HorseRacingEngine(SIMPLE_OVAL, config=config)
    assert len(engine.horses) == 2
