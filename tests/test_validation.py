"""Tests for validation module (offline, no JS server required)."""

from horse_racing.types import HorseAction
from horse_racing.validation import run_python_engine


SIMPLE_OVAL = "tracks/simple_oval.json"


def test_run_python_engine_returns_trajectories():
    actions = [[HorseAction() for _ in range(4)] for _ in range(10)]
    trajectories = run_python_engine(SIMPLE_OVAL, actions)
    # 10 steps = 10 entries (one per step, post-step state)
    assert len(trajectories) == 10
    assert len(trajectories[0]) == 4  # 4 horses


def test_trajectory_positions_change():
    actions = [[HorseAction() for _ in range(4)] for _ in range(10)]
    trajectories = run_python_engine(SIMPLE_OVAL, actions)

    # Positions should change from step 0 to step 10
    for horse in range(4):
        x0 = trajectories[0][horse]["x"]
        y0 = trajectories[0][horse]["y"]
        x1 = trajectories[-1][horse]["x"]
        y1 = trajectories[-1][horse]["y"]
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        assert dist > 0, f"Horse {horse} did not move"


def test_trajectory_has_velocity():
    actions = [[HorseAction() for _ in range(4)] for _ in range(5)]
    trajectories = run_python_engine(SIMPLE_OVAL, actions)

    # After some steps, horses should have velocity
    last = trajectories[-1]
    for horse in range(4):
        vx = last[horse]["vx"]
        vy = last[horse]["vy"]
        speed = (vx ** 2 + vy ** 2) ** 0.5
        assert speed > 0, f"Horse {horse} has zero velocity"
