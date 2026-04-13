"""Solo time-trial environment for learning pacing before racing."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..action import NUM_ACTIONS, decode_action
from ..core.observation import OBS_SIZE, build_observations
from ..core.race import Race
from ..core.track import load_track_json
from ..core.types import InputState

# Reward weights — tuned so pushing fast beats cruising, exhaustion is punished
PROGRESS_WEIGHT = 20.0
COMPLETION_BONUS = 10.0
SOLO_STEP_PENALTY = -0.005
SOLO_EXHAUSTION_PENALTY = -0.01


class SoloTimeTrialEnv(gym.Env):
    """Solo time-trial: one horse, no opponents, learn to pace.

    Reward heavily penalises each tick (incentivise speed) and exhaustion
    (incentivise stamina management).  Scaled delta-progress gives a
    per-tick gradient toward faster movement.

    Expected episode rewards (default track):
      Fast finish (~1500 ticks):  ~+22.5
      Cruise (~2100 ticks):       ~+19.5
      Push-hard-exhaust (~3500):  ~-12.5
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        track_path: str,
        max_steps: int = 5000,
    ):
        super().__init__()
        self._track_path = track_path
        self._segments = load_track_json(track_path)
        self._max_steps = max_steps

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._race: Race | None = None
        self._step_count = 0
        self._prev_progress = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._race = Race(self._segments, horse_count=1)
        self._race.start(player_horse_id=0)
        self._step_count = 0
        self._prev_progress = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert self._race is not None

        tang, norm = decode_action(action)
        inputs = {0: InputState(tang, norm)}

        self._race.tick(inputs)
        self._step_count += 1

        horse = self._race.state.horses[0]
        curr_progress = horse.track_progress

        # Reward: scaled progress + step penalty + exhaustion penalty + completion bonus
        delta = curr_progress - self._prev_progress
        reward = delta * PROGRESS_WEIGHT
        reward += SOLO_STEP_PENALTY
        if horse.current_stamina <= 0:
            reward += SOLO_EXHAUSTION_PENALTY
        if horse.finished:
            reward += COMPLETION_BONUS
        self._prev_progress = curr_progress

        terminated = horse.finished
        truncated = self._step_count >= self._max_steps

        obs = self._get_obs()
        info = self._get_info()
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        assert self._race is not None
        all_obs = build_observations(self._race)
        return all_obs[0].astype(np.float32)

    def _get_info(self) -> dict:
        assert self._race is not None
        horse = self._race.state.horses[0]
        return {
            "progress": horse.track_progress,
            "stamina": horse.current_stamina,
            "step": self._step_count,
        }
