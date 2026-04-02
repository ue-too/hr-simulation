"""Single-agent Gymnasium environment wrapper."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from horse_racing.engine import EngineConfig, HorseRacingEngine
from horse_racing.reward import compute_reward
from horse_racing.types import HorseAction, OBS_SIZE, SKILL_IDS


class HorseRacingSingleEnv(gym.Env):
    """Single-agent env: controls horse 0, other horses use zero actions."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        track_path: str | Path = "tracks/tokyo.json",
        max_steps: int = 5000,
        config: EngineConfig | None = None,
        render_mode: str | None = None,
        active_skills: set[str] | None = None,
        random_skills: bool = False,
        min_skills: int = 1,
        max_skills: int = 3,
    ) -> None:
        super().__init__()
        self.track_path = track_path
        self.max_steps = max_steps
        self.config = config or EngineConfig()
        self.render_mode = render_mode

        self.engine = HorseRacingEngine(track_path, self.config)

        self.action_space = spaces.Box(
            low=np.array([-10.0, -5.0], dtype=np.float32),
            high=np.array([10.0, 5.0], dtype=np.float32),
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

        self._active_skills = active_skills or set()
        self._random_skills = random_skills
        self._min_skills = min_skills
        self._max_skills = max_skills

        self._step_count = 0
        self._prev_obs: dict | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.engine.reset()
        self._step_count = 0
        self._prev_placement: int = 1

        # Sample skills if random mode
        if self._random_skills:
            import random as _rng
            k = _rng.randint(self._min_skills, self._max_skills)
            self._active_skills = set(_rng.sample(SKILL_IDS, k))

        all_obs = self.engine.get_observations()
        self._prev_obs = all_obs[0]
        obs_array = self.engine.obs_to_array(all_obs[0], active_skills=self._active_skills)
        return obs_array, {"active_skills": self._active_skills}

    def step(self, action: np.ndarray):
        self._step_count += 1

        actions = [HorseAction(float(action[0]), float(action[1]))]
        # Other horses get zero actions
        for _ in range(1, self.engine.horse_count):
            actions.append(HorseAction())

        self.engine.step(actions)

        all_obs = self.engine.get_observations()
        obs_curr = all_obs[0]
        obs_array = self.engine.obs_to_array(obs_curr, active_skills=self._active_skills)

        placements = self.engine.get_placements()
        finish_order = placements[0] if obs_curr["finished"] else None
        reward = compute_reward(
            self._prev_obs, obs_curr, obs_curr["collision"],
            placement=placements[0],
            num_horses=self.engine.horse_count,
            finish_order=finish_order,
            prev_placement=self._prev_placement,
            active_skills=self._active_skills if self._active_skills else None,
        )
        self._prev_placement = placements[0]

        terminated = obs_curr["finished"]
        truncated = self._step_count >= self.max_steps

        self._prev_obs = obs_curr

        return obs_array, reward, terminated, truncated, {}
