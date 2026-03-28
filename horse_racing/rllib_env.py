"""RLlib MultiAgentEnv for horse racing."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from horse_racing.engine import EngineConfig, HorseRacingEngine
from horse_racing.reward import compute_reward
from horse_racing.types import HorseAction

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError as e:
    raise ImportError("ray[rllib] is required: uv sync --extra ray") from e


class HorseRacingRLlibEnv(MultiAgentEnv):
    """RLlib-native multi-agent environment for horse racing.

    Each horse is an independent agent. Supports shared policy (all horses
    use the same policy) or per-agent policies.
    """

    metadata = {"name": "horse_racing_v0"}

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}

        self.track_path = config.get("track_path", "tracks/exp_track_8.json")
        self.max_steps = config.get("max_steps", 5000)

        engine_config = EngineConfig(
            horse_count=config.get("horse_count", 4),
            track_surface=config.get("track_surface", "dry"),
        )
        self.engine = HorseRacingEngine(self.track_path, engine_config)
        self.horse_count = engine_config.horse_count

        self._agent_ids = {f"horse_{i}" for i in range(self.horse_count)}
        self._step_count = 0
        self._prev_obs: dict[str, dict] = {}

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-10.0, -5.0], dtype=np.float32),
            high=np.array([10.0, 5.0], dtype=np.float32),
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.engine.reset()
        self._step_count = 0

        all_obs = self.engine.get_observations()
        observations = {}
        self._prev_obs = {}
        for i in range(self.horse_count):
            agent_id = f"horse_{i}"
            observations[agent_id] = self.engine.obs_to_array(all_obs[i])
            self._prev_obs[agent_id] = all_obs[i]

        infos = {agent_id: {} for agent_id in observations}
        return observations, infos

    def step(self, action_dict: dict[str, np.ndarray]):
        self._step_count += 1

        action_list = []
        for i in range(self.horse_count):
            agent_id = f"horse_{i}"
            if agent_id in action_dict:
                a = action_dict[agent_id]
                action_list.append(HorseAction(float(a[0]), float(a[1])))
            else:
                action_list.append(HorseAction())

        self.engine.step(action_list)

        all_obs = self.engine.get_observations()
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        any_terminated = False
        any_truncated = self._step_count >= self.max_steps

        for i in range(self.horse_count):
            agent_id = f"horse_{i}"
            obs_curr = all_obs[i]
            observations[agent_id] = self.engine.obs_to_array(obs_curr)

            prev = self._prev_obs.get(agent_id, obs_curr)
            rewards[agent_id] = compute_reward(prev, obs_curr, obs_curr["collision"])
            terminateds[agent_id] = obs_curr["finished"]
            truncateds[agent_id] = any_truncated
            infos[agent_id] = {}
            self._prev_obs[agent_id] = obs_curr

            if obs_curr["finished"]:
                any_terminated = True

        # __all__ key signals episode-level done to RLlib
        terminateds["__all__"] = all(
            terminateds.get(f"horse_{i}", False) for i in range(self.horse_count)
        )
        truncateds["__all__"] = any_truncated

        return observations, rewards, terminateds, truncateds, infos
