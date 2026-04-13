from pathlib import Path

import numpy as np
import pytest

from horse_racing.action import NUM_ACTIONS
from horse_racing.core.observation import OBS_SIZE
from horse_racing.env.single_env import HorseRacingSingleEnv

TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"
OVAL_PATH = str(TRACKS_DIR / "test_oval.json")


class TestHorseRacingSingleEnv:
    def test_observation_space(self):
        env = HorseRacingSingleEnv(track_path=OVAL_PATH, horse_count=4)
        assert env.observation_space.shape == (OBS_SIZE,)

    def test_action_space(self):
        env = HorseRacingSingleEnv(track_path=OVAL_PATH, horse_count=4)
        assert env.action_space.n == NUM_ACTIONS

    def test_reset_returns_obs_and_info(self):
        env = HorseRacingSingleEnv(track_path=OVAL_PATH, horse_count=4)
        obs, info = env.reset()
        assert obs.shape == (OBS_SIZE,)
        assert obs.dtype == np.float32
        assert "progress" in info
        assert "stamina" in info

    def test_step_returns_correct_tuple(self):
        env = HorseRacingSingleEnv(track_path=OVAL_PATH, horse_count=4)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(12)  # cruise
        assert obs.shape == (OBS_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    @pytest.mark.slow
    def test_terminates_when_agent_finishes(self):
        env = HorseRacingSingleEnv(track_path=OVAL_PATH, horse_count=2)
        env.reset()
        terminated = False
        for _ in range(5000):
            _, _, terminated, truncated, _ = env.step(32)  # push hard
            if terminated or truncated:
                break
        assert terminated or truncated

    def test_truncates_at_max_steps(self):
        env = HorseRacingSingleEnv(
            track_path=OVAL_PATH, horse_count=4, max_steps=10
        )
        env.reset()
        truncated = False
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(12)
            if terminated or truncated:
                break
        assert truncated

    def test_info_has_progress_and_stamina(self):
        env = HorseRacingSingleEnv(track_path=OVAL_PATH, horse_count=2)
        env.reset()
        _, _, _, _, info = env.step(12)
        assert 0.0 <= info["progress"] <= 1.0
        assert 0.0 <= info["stamina"]

    def test_gymnasium_check_env(self):
        env = HorseRacingSingleEnv(track_path=OVAL_PATH, horse_count=4)
        # Basic Gymnasium API compliance
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        obs2, _, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs2)
