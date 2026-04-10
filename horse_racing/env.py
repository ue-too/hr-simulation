"""Single-agent Gymnasium environment wrapper."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from horse_racing.bt_jockey import PERSONALITIES, make_bt_jockey
from horse_racing.engine import EngineConfig, HorseRacingEngine
from horse_racing.genome import random_genome, skill_biased_genome
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
        skill_reward_scale: float = 10.0,
        skill_physics: bool = True,
        bt_opponents: bool = True,
        reward_phase: int = 3,
    ) -> None:
        super().__init__()
        self.track_path = track_path
        self.max_steps = max_steps
        self.config = config or EngineConfig()
        self.render_mode = render_mode

        self.engine = HorseRacingEngine(track_path, self.config)

        # Action space: network outputs in [-1, 1], remapped in step() to
        # physics range. Tangential is asymmetric [-3, +10] — brake side
        # trimmed because the auto-cruise spring already decelerates toward
        # cruise at action=0; deep braking is rarely needed. Positive side
        # keeps the full +10 burst capability. Normal stays symmetric [-5, 5].
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
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
        self._skill_reward_scale = skill_reward_scale
        self._skill_physics = skill_physics
        self._bt_opponents = bt_opponents
        self._reward_phase = reward_phase

        self._step_count = 0
        self._prev_obs: dict | None = None
        self._bt_jockeys: list = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_placement: int = 1

        # Sample skills if random mode
        if self._random_skills:
            import random as _rng
            # 20% no-skill episodes for contrastive learning signal
            if _rng.random() < 0.2:
                self._active_skills = set()
            else:
                k = _rng.randint(self._min_skills, self._max_skills)
                self._active_skills = set(_rng.sample(SKILL_IDS, k))

        # Generate genomes: all horses share the same random genome so no
        # jockey gets a stat advantage — differences come from strategy only.
        if self._active_skills:
            shared = skill_biased_genome(self._active_skills)
        else:
            shared = random_genome()
        genomes = [shared] * self.engine.horse_count
        self.engine.reset(genomes=genomes)
        if self._skill_physics:
            self.engine.active_skills = self._active_skills

        # Scale stamina for long tracks so all horses can finish with pacing.
        # Reference: 900m track needs no scaling. Longer tracks get proportionally
        # more stamina so the agent can learn strategy instead of being doomed.
        track_length = self.engine.horses[0].navigator._total_length
        REFERENCE_LENGTH = 900.0
        if track_length > REFERENCE_LENGTH:
            stamina_scale = track_length / REFERENCE_LENGTH
            for hs in self.engine.horses:
                hs.base_attrs.stamina *= stamina_scale
                hs.effective_attrs.stamina *= stamina_scale
                hs.runtime.current_stamina *= stamina_scale

        # Randomize starting lane: swap horse 0's position with a random horse
        if self.engine.horse_count > 1:
            swap_idx = self.np_random.integers(0, self.engine.horse_count)
            if swap_idx != 0:
                h0 = self.engine.horses[0]
                h1 = self.engine.horses[swap_idx]
                h0.body.position, h1.body.position = (
                    h1.body.position.copy(), h0.body.position.copy(),
                )

        # Create BT opponents for horses 1..N
        self._bt_jockeys = []
        if self._bt_opponents:
            personality_choices = list(PERSONALITIES.keys())
            for _ in range(1, self.engine.horse_count):
                arch = self.np_random.choice(personality_choices)
                self._bt_jockeys.append(make_bt_jockey(arch))

        all_obs = self.engine.get_observations()
        self._prev_obs = all_obs[0]
        obs_array = self.engine.obs_to_array(all_obs[0], active_skills=self._active_skills)
        return obs_array, {"active_skills": self._active_skills}

    @staticmethod
    def remap_action(action: np.ndarray) -> tuple[float, float]:
        """Map network output [-1, 1] to physics action range.

        Tangential: [-1, 1] → [-3, +10]  (asymmetric — brake side trimmed,
                                          full burst preserved. Neutral
                                          action 0 → physics +3.5, biasing
                                          initial policy toward pushing forward.)
        Normal:     [-1, 1] → [-5, 5]    (matches BT jockey / engine range)
        """
        tang = float(action[0]) * 6.5 + 3.5
        norm = float(action[1]) * 5.0
        return tang, norm

    def step(self, action: np.ndarray):
        self._step_count += 1

        tang, norm = self.remap_action(action)
        actions = [HorseAction(tang, norm)]
        # Other horses: BT opponents or zero actions
        if self._bt_jockeys:
            bt_obs = self.engine.get_observations()
            for j, bt in enumerate(self._bt_jockeys):
                actions.append(bt.compute_action(bt_obs[j + 1]))
        else:
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
            skill_reward_scale=self._skill_reward_scale,
            reward_phase=self._reward_phase,
            raw_tang_action=tang,
        )
        self._prev_placement = placements[0]

        terminated = obs_curr["finished"]
        truncated = self._step_count >= self.max_steps

        self._prev_obs = obs_curr

        return obs_array, reward, terminated, truncated, {}
