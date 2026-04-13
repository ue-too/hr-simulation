"""Tests for SoloTimeTrialEnv."""

from __future__ import annotations

import pytest

from horse_racing.env.solo_env import (
    COMPLETION_BONUS,
    PROGRESS_WEIGHT,
    SOLO_EXHAUSTION_PENALTY,
    SOLO_STEP_PENALTY,
    SoloTimeTrialEnv,
)

TRACK = "tracks/test_oval.json"


@pytest.fixture
def env():
    e = SoloTimeTrialEnv(track_path=TRACK, max_steps=200)
    yield e
    e.close()


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert "progress" in info
    assert "stamina" in info
    assert info["progress"] == pytest.approx(0.0, abs=0.01)


def test_step_returns_five_tuple(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(12)  # action 12 = (0, 0)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_cruise_action_makes_progress(env):
    env.reset()
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(12)
    assert info["progress"] > 0.0


def test_pushing_makes_more_progress_than_cruise(env):
    """Pushing (+1.0) should yield more progress than cruising (0.0) over same ticks."""
    env.reset(seed=42)
    for _ in range(50):
        env.step(22)  # tangential +1.0
    push_progress = env._race.state.horses[0].track_progress

    env.reset(seed=42)
    for _ in range(50):
        env.step(12)  # tangential 0.0
    cruise_progress = env._race.state.horses[0].track_progress

    assert push_progress > cruise_progress


def test_step_penalty_every_tick(env):
    """Even zero-progress ticks should include step penalty."""
    env.reset()
    # First tick from standstill — progress is near zero
    _, reward, _, _, _ = env.step(12)
    # Reward should be close to SOLO_STEP_PENALTY (progress is tiny)
    assert reward < 0.01  # bounded above by small progress
    assert reward > SOLO_STEP_PENALTY - 0.01  # bounded below


def test_truncates_at_max_steps(env):
    """Should truncate after max_steps."""
    env.reset()
    truncated = False
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step(12)
        if terminated or truncated:
            break
    assert truncated


def test_no_opponents_in_obs(env):
    """Solo env has 1 horse — opponent slots should be zeros."""
    obs, _ = env.reset()
    # Opponent slots start at index 24 (14 self + 10 track)
    opponent_data = obs[24:]
    assert all(v == 0.0 for v in opponent_data)
