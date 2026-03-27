"""Tests for Gymnasium and PettingZoo environments."""

import numpy as np

from horse_racing.env import HorseRacingSingleEnv
from horse_racing.multi_agent_env import HorseRacingEnv


SIMPLE_OVAL = "tracks/simple_oval.json"


def test_single_env_reset():
    env = HorseRacingSingleEnv(track_path=SIMPLE_OVAL)
    obs, info = env.reset()
    assert obs.shape == (15,)
    assert isinstance(info, dict)


def test_single_env_step():
    env = HorseRacingSingleEnv(track_path=SIMPLE_OVAL)
    env.reset()
    action = np.array([2.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (15,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_single_env_action_space():
    env = HorseRacingSingleEnv(track_path=SIMPLE_OVAL)
    assert env.action_space.shape == (2,)
    assert env.action_space.low[0] == -10.0
    assert env.action_space.high[0] == 10.0


def test_multi_agent_env_reset():
    env = HorseRacingEnv(track_path=SIMPLE_OVAL)
    observations, infos = env.reset()
    assert len(observations) == 4
    for agent in env.possible_agents:
        assert observations[agent].shape == (15,)


def test_multi_agent_env_step():
    env = HorseRacingEnv(track_path=SIMPLE_OVAL)
    env.reset()

    actions = {
        agent: np.array([1.0, 0.0], dtype=np.float32)
        for agent in env.agents
    }
    observations, rewards, terminated, truncated, infos = env.step(actions)
    assert len(observations) > 0
    for agent in observations:
        assert observations[agent].shape == (15,)


def test_multi_agent_env_agents():
    env = HorseRacingEnv(track_path=SIMPLE_OVAL)
    env.reset()
    assert env.possible_agents == ["horse_0", "horse_1", "horse_2", "horse_3"]
    assert env.agents == env.possible_agents
