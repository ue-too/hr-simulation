"""PettingZoo ParallelEnv for multi-agent horse racing."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
from gymnasium import spaces

from horse_racing.engine import EngineConfig, HorseRacingEngine
from horse_racing.reward import compute_reward
from horse_racing.types import HORSE_COUNT, HorseAction

try:
    from pettingzoo import ParallelEnv
except ImportError as e:
    raise ImportError("pettingzoo is required for multi-agent env: pip install pettingzoo") from e


class HorseRacingEnv(ParallelEnv):
    """Multi-agent parallel environment for horse racing."""

    metadata = {"name": "horse_racing_v0", "render_modes": ["human"]}

    def __init__(
        self,
        track_path: str | Path = "tracks/exp_track_8.json",
        max_steps: int = 5000,
        config: EngineConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.track_path = track_path
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._config = config or EngineConfig()

        self.possible_agents = [f"horse_{i}" for i in range(self._config.horse_count)]
        self.agents = list(self.possible_agents)

        self.engine = HorseRacingEngine(track_path, self._config)

        self._step_count = 0
        self._prev_obs: dict[str, dict] | None = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Box(
            low=np.array([-10.0, -5.0], dtype=np.float32),
            high=np.array([10.0, 5.0], dtype=np.float32),
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.agents = list(self.possible_agents)
        self.engine.reset()
        self._step_count = 0

        all_obs = self.engine.get_observations()
        observations = {}
        self._prev_obs = {}
        self._prev_placements: dict[str, int] = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self.engine.obs_to_array(all_obs[i])
            self._prev_obs[agent] = all_obs[i]
            self._prev_placements[agent] = i + 1

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, np.ndarray]):
        self._step_count += 1

        action_list = []
        for i in range(len(self.possible_agents)):
            agent = f"horse_{i}"
            if agent in actions:
                a = actions[agent]
                action_list.append(HorseAction(float(a[0]), float(a[1])))
            else:
                action_list.append(HorseAction())

        self.engine.step(action_list)

        all_obs = self.engine.get_observations()
        observations = {}
        rewards = {}
        terminated = {}
        truncated = {}
        infos = {}

        placements = self.engine.get_placements()
        for i, agent in enumerate(self.agents):
            obs_curr = all_obs[i]
            observations[agent] = self.engine.obs_to_array(obs_curr)

            prev = self._prev_obs.get(agent, obs_curr)
            finish_order = placements[i] if obs_curr["finished"] else None
            prev_place = self._prev_placements.get(agent)
            rewards[agent] = compute_reward(
                prev, obs_curr, obs_curr["collision"],
                placement=placements[i],
                num_horses=self._config.horse_count,
                finish_order=finish_order,
                prev_placement=prev_place,
            )
            self._prev_placements[agent] = placements[i]
            terminated[agent] = obs_curr["finished"]
            truncated[agent] = self._step_count >= self.max_steps
            infos[agent] = {}
            self._prev_obs[agent] = obs_curr

        # Remove finished agents
        self.agents = [a for a in self.agents if not terminated[a] and not truncated[a]]

        return observations, rewards, terminated, truncated, infos
