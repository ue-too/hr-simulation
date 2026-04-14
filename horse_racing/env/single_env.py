"""Gymnasium environment for single-agent horse racing training."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..action import NUM_ACTIONS, decode_action
from ..core.attributes import create_default_attributes, create_randomized_attributes
from ..core.observation import OBS_SIZE, build_observations
from ..core.race import Race
from ..core.track import load_track_json
from ..core.types import InputState
from ..opponents.scripted import Strategy, random_strategy
from ..reward import compute_reward


class HorseRacingSingleEnv(gym.Env):
    """Single-agent horse racing environment.

    The agent controls one horse. Opponents use scripted strategies.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        track_path: str,
        horse_count: int = 4,
        agent_horse_id: int = 0,
        max_steps: int = 5000,
    ):
        super().__init__()
        self._track_path = track_path
        self._segments = load_track_json(track_path)
        self._horse_count = horse_count
        self._agent_id = agent_horse_id
        self._max_steps = max_steps

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._race: Race | None = None
        self._opponent_strategies: dict[int, Strategy] = {}
        self._step_count = 0
        self._prev_progress = 0.0
        self._prev_rank = 1

    def _build_attr_factories(self) -> dict:
        """Agent gets default attributes, opponents get randomized (±10%)."""
        factories = {self._agent_id: create_default_attributes}
        for i in range(self._horse_count):
            if i != self._agent_id:
                factories[i] = create_randomized_attributes
        return factories

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._race = Race(
            self._segments, self._horse_count,
            attr_factories=self._build_attr_factories(),
        )
        self._race.start(self._agent_id)
        self._step_count = 0
        self._prev_progress = 0.0
        self._prev_rank = self._horse_count  # start at the back

        # Assign scripted strategies to opponents
        self._opponent_strategies = {}
        for h in self._race.state.horses:
            if h.id != self._agent_id:
                self._opponent_strategies[h.id] = random_strategy()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert self._race is not None

        # Decode agent action
        tang, norm = decode_action(action)
        inputs: dict[int, InputState] = {
            self._agent_id: InputState(tang, norm)
        }

        # Compute opponent actions
        for h in self._race.state.horses:
            if h.id != self._agent_id and not h.finished:
                strategy = self._opponent_strategies[h.id]
                continuous = strategy.act_continuous(h)
                if continuous is not None:
                    inputs[h.id] = continuous
                else:
                    opp_action = strategy.act(h.track_progress)
                    opp_tang, opp_norm = decode_action(opp_action)
                    inputs[h.id] = InputState(opp_tang, opp_norm)

        self._race.tick(inputs)
        self._step_count += 1

        agent_horse = self._race.state.horses[self._agent_id]
        curr_progress = agent_horse.track_progress

        # Compute current rank (1 = leading)
        curr_rank = 1
        for h in self._race.state.horses:
            if h.id != self._agent_id and h.track_progress > agent_horse.track_progress:
                curr_rank += 1
        overtakes = max(0, self._prev_rank - curr_rank)

        # Compute reward
        reward = compute_reward(
            self._prev_progress, curr_progress, agent_horse.finish_order,
            agent_horse.current_stamina,
            agent_horse.base_attributes.max_stamina,
            overtakes=overtakes,
        )
        self._prev_progress = curr_progress
        self._prev_rank = curr_rank

        terminated = agent_horse.finished
        truncated = self._step_count >= self._max_steps

        obs = self._get_obs()
        info = self._get_info()
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        assert self._race is not None
        all_obs = build_observations(self._race)
        return all_obs[self._agent_id].astype(np.float32)

    def _get_info(self) -> dict:
        assert self._race is not None
        agent = self._race.state.horses[self._agent_id]
        return {
            "progress": agent.track_progress,
            "stamina": agent.current_stamina,
            "finish_order": agent.finish_order,
        }
