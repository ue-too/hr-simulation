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

# Reward weights — tuned so smart pacing beats cruise by ~+5
PROGRESS_WEIGHT = 10.0
PAR_TICKS = 2200          # ~cruise finish time; finishing faster earns a bonus
TIME_BONUS_MAX = 50.0     # bonus at tick 0; linearly decays to 0 at PAR_TICKS
SOLO_EXHAUSTION_PENALTY = -0.05


class SoloTimeTrialEnv(gym.Env):
    """Solo time-trial: one horse, no opponents, learn to pace.

    Reward = scaled delta-progress (dense signal) + time bonus at finish
    (faster = bigger bonus) + exhaustion penalty (stamina=0 is punished).

    No per-tick step penalty — only the time bonus at finish rewards speed.

    Expected episode rewards (default track):
      Smart pacing (~1977 ticks):  ~+15.1  (progress 10.0 + bonus 5.1)
      Cruise (~2219 ticks):        ~+10.0  (progress 10.0 + bonus 0.0)
      Push-hard-exhaust (~2068):   ~ -9.1  (progress 10.0 + bonus 3.0 - exhaust)
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

        # Reward: scaled progress + exhaustion penalty + time bonus at finish
        delta = curr_progress - self._prev_progress
        reward = delta * PROGRESS_WEIGHT
        if horse.current_stamina <= 0:
            reward += SOLO_EXHAUSTION_PENALTY
        if horse.finished:
            reward += TIME_BONUS_MAX * max(0.0, 1.0 - self._step_count / PAR_TICKS)
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
