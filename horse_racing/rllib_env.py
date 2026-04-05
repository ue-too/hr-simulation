"""RLlib MultiAgentEnv for horse racing."""

from __future__ import annotations

import random
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from horse_racing.bt_jockey import BTJockey, make_bt_jockey
from horse_racing.engine import EngineConfig, HorseRacingEngine
from horse_racing.genome import random_genome
from horse_racing.reward import ARCHETYPES, compute_reward
from horse_racing.types import MAX_HORSE_COUNT, OBS_SIZE, HorseAction

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError as e:
    raise ImportError("ray[rllib] is required: uv sync --extra ray") from e


class HorseRacingRLlibEnv(MultiAgentEnv):
    """RLlib-native multi-agent environment for horse racing.

    Each horse is an independent agent. All agents share a single policy
    (parameter sharing). Per-episode randomization of tracks, horse count,
    genomes, and archetypes creates diverse training scenarios.
    """

    metadata = {"name": "horse_racing_v2"}

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}

        # Track config: single path or list of paths for randomization
        track_paths = config.get("track_paths", None)
        if track_paths:
            self.track_paths = list(track_paths)
        else:
            self.track_paths = [config.get("track_path", "tracks/tokyo.json")]

        self.max_steps = config.get("max_steps", 5000)

        # Horse count range for per-episode randomization
        self.min_horse_count = config.get("min_horse_count", config.get("horse_count", 4))
        self.max_horse_count = config.get("max_horse_count", config.get("horse_count", 4))
        self.max_horse_count = min(self.max_horse_count, MAX_HORSE_COUNT)

        # Randomization flags
        self.randomize_archetypes = config.get("randomize_archetypes", True)
        self.randomize_genomes = config.get("randomize_genomes", True)

        # Static archetypes (used when randomize_archetypes=False)
        self.archetypes: dict[str, str | None] = config.get("archetypes", {})

        self.track_surface = config.get("track_surface", "dry")

        # BT opponent ratio: fraction of horses controlled by behavior tree
        self.bt_opponent_ratio = config.get("bt_opponent_ratio", 0.0)

        # possible_agents must cover the max to satisfy RLlib
        self.possible_agents = [f"horse_{i}" for i in range(self.max_horse_count)]
        self._agent_ids = set(self.possible_agents)
        self.agents = list(self.possible_agents)

        # Per-agent spaces (same for all agents)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-10.0, -5.0], dtype=np.float32),
            high=np.array([10.0, 5.0], dtype=np.float32),
        )
        self.observation_spaces = {
            agent: self.observation_space for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.action_space for agent in self.possible_agents
        }

        # State (initialized in reset)
        self._rng = random.Random()
        self._horse_count = self.min_horse_count
        self._archetypes_this_ep: dict[str, str | None] = {}
        self._step_count = 0
        self._prev_obs: dict[str, dict] = {}
        self._prev_placements: dict[str, int] = {}
        self._done_agents: set[str] = set()
        self.engine: HorseRacingEngine | None = None
        self._bt_agents: dict[str, BTJockey] = {}  # agent_id -> BTJockey

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._rng = random.Random(seed)
        self._step_count = 0
        self._done_agents = set()

        # Randomize horse count
        self._horse_count = self._rng.randint(self.min_horse_count, self.max_horse_count)

        # Randomize track
        track_path = self._rng.choice(self.track_paths)

        # Randomize genomes
        genomes = None
        if self.randomize_genomes:
            genomes = [random_genome() for _ in range(self._horse_count)]

        # Create engine
        engine_config = EngineConfig(
            horse_count=self._horse_count,
            track_surface=self.track_surface,
        )
        self.engine = HorseRacingEngine(track_path, engine_config, genomes=genomes)

        # Randomize archetypes
        self._archetypes_this_ep = {}
        if self.randomize_archetypes:
            choices = ARCHETYPES + [None]  # type: ignore[list-item]
            for i in range(self._horse_count):
                self._archetypes_this_ep[f"horse_{i}"] = self._rng.choice(choices)
        else:
            self._archetypes_this_ep = dict(self.archetypes)

        # Assign BT opponents — pick a random subset of horses
        self._bt_agents = {}
        if self.bt_opponent_ratio > 0 and self._horse_count > 1:
            bt_count = max(1, int(self._horse_count * self.bt_opponent_ratio))
            # Never make all horses BT — at least one must be RL
            bt_count = min(bt_count, self._horse_count - 1)
            bt_indices = self._rng.sample(range(self._horse_count), bt_count)
            archetype_choices = ARCHETYPES + [None]  # type: ignore[list-item]
            for idx in bt_indices:
                archetype = self._archetypes_this_ep.get(f"horse_{idx}")
                self._bt_agents[f"horse_{idx}"] = make_bt_jockey(archetype)

        # Set active RL agents for this episode (exclude BT-controlled horses)
        self.agents = [
            f"horse_{i}" for i in range(self._horse_count)
            if f"horse_{i}" not in self._bt_agents
        ]

        # Get initial observations
        all_obs = self.engine.get_observations()
        observations = {}
        self._prev_obs = {}
        self._prev_placements = {}
        for i in range(self._horse_count):
            agent_id = f"horse_{i}"
            self._prev_obs[agent_id] = all_obs[i]
            self._prev_placements[agent_id] = i + 1
            # Only return obs for RL agents
            if agent_id not in self._bt_agents:
                observations[agent_id] = self.engine.obs_to_array(all_obs[i])

        infos = {agent_id: {} for agent_id in observations}
        return observations, infos

    def step(self, action_dict: dict[str, np.ndarray]):
        self._step_count += 1

        # Build action list: RL actions from action_dict, BT actions computed here
        action_list = []
        bt_obs = self.engine.get_observations() if self._bt_agents else None
        for i in range(self._horse_count):
            agent_id = f"horse_{i}"
            if agent_id in self._bt_agents:
                action_list.append(self._bt_agents[agent_id].compute_action(bt_obs[i]))
            elif agent_id in action_dict:
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

        for i in range(self._horse_count):
            agent_id = f"horse_{i}"
            if agent_id in self._done_agents or agent_id in self._bt_agents:
                continue

            obs_curr = all_obs[i]
            observations[agent_id] = self.engine.obs_to_array(obs_curr)

            prev = self._prev_obs.get(agent_id, obs_curr)
            finish_order = placements[i] if obs_curr["finished"] else None
            archetype = self._archetypes_this_ep.get(agent_id)
            prev_place = self._prev_placements.get(agent_id)
            rewards[agent_id] = compute_reward(
                prev, obs_curr, obs_curr["collision"],
                placement=placements[i],
                num_horses=self._horse_count,
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

        # Track done state for BT agents too (needed for __all__ check)
        for i in range(self._horse_count):
            agent_id = f"horse_{i}"
            if agent_id in self._bt_agents:
                if all_obs[i]["finished"] or any_truncated:
                    self._done_agents.add(agent_id)

        # Active RL agents only
        self.agents = [
            a for a in self.possible_agents[:self._horse_count]
            if a not in self._done_agents and a not in self._bt_agents
        ]

        terminateds["__all__"] = all(
            f"horse_{i}" in self._done_agents
            for i in range(self._horse_count)
        )
        truncateds["__all__"] = any_truncated

        return observations, rewards, terminateds, truncateds, infos
