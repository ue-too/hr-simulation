"""Self-play Gymnasium environment for Phase 3 multi-agent training.

Controls one horse (the trainee) while opponents are driven by frozen ONNX
models. Supports variable field sizes (2-20 horses) with random opponent
sampling each episode.
"""

from __future__ import annotations

import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import onnxruntime as ort
from gymnasium import spaces

from horse_racing.engine import EngineConfig, HorseRacingEngine
from horse_racing.reward import compute_reward
from horse_racing.types import HorseAction, OBS_SIZE, SKILL_IDS


class SelfPlayEnv(gym.Env):
    """Single-agent env where opponents are driven by frozen ONNX models.

    Each reset() samples a random track and random number of opponents.
    The trainee is always horse index 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        tracks: list[str] | str,
        max_steps: int = 5000,
        opponent_onnx_paths: list[str] | None = None,
        trainee_archetype: str | None = None,
        min_opponents: int = 3,
        max_opponents: int = 19,
        config: EngineConfig | None = None,
        render_mode: str | None = None,
        stagger_range: tuple[float, float] = (0.0, 0.0),
        active_skills: set[str] | None = None,
        random_skills: bool = False,
        min_skills: int = 1,
        max_skills: int = 3,
        skill_reward_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.tracks = [tracks] if isinstance(tracks, str) else list(tracks)
        self.max_steps = max_steps
        self.trainee_archetype = trainee_archetype
        self.min_opponents = min_opponents
        self.max_opponents = max_opponents
        self._base_config = config or EngineConfig()
        self.render_mode = render_mode
        self.stagger_range = stagger_range

        # Load opponent ONNX sessions
        self._opponent_sessions: list[ort.InferenceSession] = []
        if opponent_onnx_paths:
            for path in opponent_onnx_paths:
                self._opponent_sessions.append(
                    ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                )

        self.action_space = spaces.Box(
            low=np.array([-10.0, -5.0], dtype=np.float32),
            high=np.array([10.0, 5.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32,
        )

        self._active_skills = active_skills or set()
        self._random_skills = random_skills
        self._min_skills = min_skills
        self._max_skills = max_skills
        self._skill_reward_scale = skill_reward_scale

        self._step_count = 0
        self._prev_obs: dict | None = None
        self._prev_placement: int = 1
        self._all_prev_obs: list[dict] = []
        self._num_opponents = 0
        self._opponent_indices: list[int] = []  # which ONNX session each opponent uses
        self.engine: HorseRacingEngine | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_placement = 1

        # Sample number of opponents
        max_opp = min(self.max_opponents, len(self._opponent_sessions)) if self._opponent_sessions else 0
        if max_opp < self.min_opponents:
            # Not enough ONNX models — use what we have, pad with zero-action horses
            self._num_opponents = max(self.min_opponents, max_opp)
        else:
            self._num_opponents = random.randint(self.min_opponents, max_opp)

        # Sample which ONNX models to use (with replacement if more opponents than models)
        if self._opponent_sessions:
            self._opponent_indices = [
                random.randrange(len(self._opponent_sessions))
                for _ in range(self._num_opponents)
            ]
        else:
            self._opponent_indices = []

        # Pick random track and create engine
        track = random.choice(self.tracks)
        horse_count = 1 + self._num_opponents
        config = EngineConfig(
            horse_count=horse_count,
            track_surface=self._base_config.track_surface,
        )
        self.engine = HorseRacingEngine(track, config)

        # Stagger opponents ahead of trainee for overtake training
        if self.stagger_range[1] > 0:
            offsets = [0.0]  # trainee stays at start
            for _ in range(self._num_opponents):
                offsets.append(random.uniform(self.stagger_range[0], self.stagger_range[1]))
            self.engine.stagger_horses(offsets)

        # Sample skills if random mode
        if self._random_skills:
            # 20% no-skill episodes for contrastive learning signal
            if random.random() < 0.2:
                self._active_skills = set()
            else:
                k = random.randint(self._min_skills, self._max_skills)
                self._active_skills = set(random.sample(SKILL_IDS, k))

        all_obs = self.engine.get_observations()
        self._all_prev_obs = list(all_obs)
        self._prev_obs = all_obs[0]

        obs_array = self.engine.obs_to_array(all_obs[0], active_skills=self._active_skills)
        return obs_array, {"active_skills": self._active_skills}

    def step(self, action: np.ndarray):
        self._step_count += 1

        # Build action list: trainee + opponents
        actions: list[HorseAction] = [
            HorseAction(float(action[0]), float(action[1]))
        ]

        for opp_i in range(self._num_opponents):
            horse_idx = opp_i + 1  # opponent horse indices start at 1
            if opp_i < len(self._opponent_indices) and self._opponent_indices[opp_i] < len(self._opponent_sessions):
                # Use cached observation from previous step
                session = self._opponent_sessions[self._opponent_indices[opp_i]]
                obs_array = self.engine.obs_to_array(
                    self._all_prev_obs[horse_idx]
                ).reshape(1, -1)
                onnx_action = session.run(["action"], {"obs": obs_array})[0][0]
                actions.append(HorseAction(float(onnx_action[0]), float(onnx_action[1])))
            else:
                actions.append(HorseAction())  # zero action fallback

        self.engine.step(actions)

        all_obs = self.engine.get_observations()
        self._all_prev_obs = list(all_obs)

        obs_curr = all_obs[0]
        obs_array = self.engine.obs_to_array(obs_curr, active_skills=self._active_skills)

        placements = self.engine.get_placements()
        num_horses = 1 + self._num_opponents
        finish_order = placements[0] if obs_curr["finished"] else None

        reward = compute_reward(
            self._prev_obs, obs_curr, obs_curr["collision"],
            placement=placements[0],
            num_horses=num_horses,
            finish_order=finish_order,
            archetype=self.trainee_archetype,
            prev_placement=self._prev_placement,
            active_skills=self._active_skills if self._active_skills else None,
            skill_reward_scale=self._skill_reward_scale,
        )

        # Track overtakes for monitoring
        overtakes = max(0, self._prev_placement - placements[0])

        self._prev_placement = placements[0]
        self._prev_obs = obs_curr

        terminated = obs_curr["finished"]
        truncated = self._step_count >= self.max_steps

        info = {
            "placement": placements[0],
            "overtakes": overtakes,
            "num_opponents": self._num_opponents,
            "finished": obs_curr["finished"],
        }

        return obs_array, reward, terminated, truncated, info
