"""RLlib MultiAgentEnv for horse racing."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from horse_racing.engine import EngineConfig, HorseRacingEngine
from horse_racing.reward import compute_reward
from horse_racing.types import HorseAction, OBS_SIZE

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

        self.track_path = config.get("track_path", "tracks/tokyo.json")
        self.max_steps = config.get("max_steps", 5000)
        # Per-agent archetype: {"horse_0": "front_runner", "horse_1": "stalker", ...}
        self.archetypes: dict[str, str | None] = config.get("archetypes", {})

        engine_config = EngineConfig(
            horse_count=config.get("horse_count", 4),
            track_surface=config.get("track_surface", "dry"),
        )
        self.engine = HorseRacingEngine(self.track_path, engine_config)
        self.horse_count = engine_config.horse_count

        self._agent_ids = {f"horse_{i}" for i in range(self.horse_count)}
        # RLlib new API stack requires these PettingZoo-style attributes
        self.possible_agents = [f"horse_{i}" for i in range(self.horse_count)]
        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._prev_obs: dict[str, dict] = {}

        # Per-agent space (singular — what each agent sees/does)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-10.0, -5.0], dtype=np.float32),
            high=np.array([10.0, 5.0], dtype=np.float32),
        )

        # Per-agent space dicts (plural — required by RLlib new API stack)
        self.observation_spaces = {
            f"horse_{i}": self.observation_space for i in range(self.horse_count)
        }
        self.action_spaces = {
            f"horse_{i}": self.action_space for i in range(self.horse_count)
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.engine.reset()
        self._step_count = 0
        self.agents = list(self.possible_agents)
        self._done_agents: set[str] = set()

        all_obs = self.engine.get_observations()
        observations = {}
        self._prev_obs = {}
        self._prev_placements: dict[str, int] = {}
        for i in range(self.horse_count):
            agent_id = f"horse_{i}"
            observations[agent_id] = self.engine.obs_to_array(all_obs[i])
            self._prev_obs[agent_id] = all_obs[i]
            self._prev_placements[agent_id] = i + 1  # initial placement by lane

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

        any_truncated = self._step_count >= self.max_steps
        placements = self.engine.get_placements()

        for i in range(self.horse_count):
            agent_id = f"horse_{i}"
            # Skip agents that are already done
            if agent_id in self._done_agents:
                continue

            obs_curr = all_obs[i]
            observations[agent_id] = self.engine.obs_to_array(obs_curr)

            prev = self._prev_obs.get(agent_id, obs_curr)
            finish_order = placements[i] if obs_curr["finished"] else None
            archetype = self.archetypes.get(agent_id)
            prev_place = self._prev_placements.get(agent_id)
            rewards[agent_id] = compute_reward(
                prev, obs_curr, obs_curr["collision"],
                placement=placements[i],
                num_horses=self.horse_count,
                finish_order=finish_order,
                archetype=archetype,
                prev_placement=prev_place,
            )
            self._prev_placements[agent_id] = placements[i]
            terminateds[agent_id] = obs_curr["finished"]
            truncateds[agent_id] = any_truncated
            infos[agent_id] = {}
            self._prev_obs[agent_id] = obs_curr

            if obs_curr["finished"] or any_truncated:
                self._done_agents.add(agent_id)

        # Update active agents list
        self.agents = [a for a in self.possible_agents if a not in self._done_agents]

        # __all__ key signals episode-level done to RLlib
        terminateds["__all__"] = len(self.agents) == 0
        truncateds["__all__"] = any_truncated

        return observations, rewards, terminateds, truncateds, infos
