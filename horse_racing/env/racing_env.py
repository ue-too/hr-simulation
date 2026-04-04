"""Gymnasium environment for the v2 horse racing simulation.

Single-agent: controls one horse (the trainee). Other horses use zero actions
(or frozen ONNX policies in self-play mode, handled externally).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from horse_racing.core.horse import HorseProfile, random_horse
from horse_racing.core.physics import (
    compute_drafting_bonus,
    resolve_horse_collisions,
    resolve_wall_collisions,
    step_horse,
    update_orientation,
)
from horse_racing.core.stamina import StaminaState, compute_drain, create_stamina
from horse_racing.core.track import load_track, compute_total_length
from horse_racing.core.track_navigator import TrackNavigator
from horse_racing.core.types import (
    HORSE_SPACING,
    PHYS_HZ,
    PHYS_SUBSTEPS,
    HorseBody,
    JockeyAction,
    JockeyStyle,
    TrackFrame,
)
from horse_racing.env.observation import OBS_SIZE, build_observation
from horse_racing.env.reward import compute_reward


class HorseRacingEnv(gym.Env):
    """Single-agent racing environment with effort+lane action space."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        track_path: str | Path = "tracks/tokyo.json",
        num_horses: int = 4,
        max_steps: int = 5000,
        render_mode: str | None = None,
        randomize_horses: bool = True,
        randomize_jockey_style: bool = False,
    ) -> None:
        super().__init__()
        self.track_path = str(track_path)
        self.num_horses = num_horses
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.randomize_horses = randomize_horses
        self.randomize_jockey_style = randomize_jockey_style

        # Action: [effort, lane] both in [-1, 1]
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

        # Load track
        self._track_data = load_track(track_path)
        self._track_length = compute_total_length(self._track_data.segments)
        self._est_race_time = self._track_length / 16.0  # rough estimate

        # Per-tick timing
        # Each env step = PHYS_SUBSTEPS physics ticks at PHYS_HZ
        self._substep_dt = 1.0 / PHYS_HZ              # physics tick: 1/240s
        self._dt = self._substep_dt * PHYS_SUBSTEPS    # env step:    1/30s

        # State (initialized in reset)
        self._profiles: list[HorseProfile] = []
        self._bodies: list[HorseBody] = []
        self._staminas: list[StaminaState] = []
        self._navigators: list[TrackNavigator] = []
        self._frames: list[TrackFrame] = []
        self._jockey_style = JockeyStyle()
        self._step_count = 0
        self._sim_time = 0.0
        self._finished: list[bool] = []
        self._finish_order: list[int] = []
        self._prev_progress = 0.0
        self._prev_placement = 0
        self._placement_at_75: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = random.Random(seed)

        self._step_count = 0
        self._sim_time = 0.0
        self._finished = [False] * self.num_horses
        self._finish_order = []
        self._prev_progress = 0.0
        self._prev_placement = 0
        self._placement_at_75 = None

        # Generate horse profiles
        self._profiles = []
        for _ in range(self.num_horses):
            if self.randomize_horses:
                self._profiles.append(random_horse(rng))
            else:
                from horse_racing.core.horse import default_horse
                self._profiles.append(default_horse())

        # Initialize bodies at start line
        self._bodies = []
        self._navigators = []
        start_pos = self._track_data.segments[0].start_point

        # Get initial tangential direction for orientation
        seg0 = self._track_data.segments[0]
        if hasattr(seg0, 'end_point'):
            dx = seg0.end_point[0] - seg0.start_point[0]
            dy = seg0.end_point[1] - seg0.start_point[1]
            start_orientation = math.atan2(dy, dx)
        else:
            start_orientation = 0.0

        for i in range(self.num_horses):
            # Stagger horses laterally
            lateral_offset = (i - (self.num_horses - 1) / 2.0) * HORSE_SPACING
            body = HorseBody()
            body.position = np.array([
                start_pos[0],
                start_pos[1] + lateral_offset,
            ], dtype=np.float64)
            body.orientation = start_orientation
            self._bodies.append(body)

            nav = TrackNavigator(self._track_data.segments)
            self._navigators.append(nav)

        # Initialize stamina
        self._staminas = [create_stamina(p) for p in self._profiles]

        # Compute initial frames
        self._frames = [
            self._navigators[i].update(self._bodies[i].position)
            for i in range(self.num_horses)
        ]

        # Randomize jockey style if enabled
        if self.randomize_jockey_style:
            self._jockey_style = JockeyStyle(
                risk_tolerance=rng.random(),
                tactical_bias=rng.uniform(-1, 1),
                skill_level=rng.random(),
            )
        else:
            self._jockey_style = JockeyStyle()

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1

        # Parse action
        trainee_action = JockeyAction(
            effort=float(np.clip(action[0], -1.0, 1.0)),
            lane=float(np.clip(action[1], -1.0, 1.0)),
        )

        # All other horses get zero action (idle)
        actions = [JockeyAction()] * self.num_horses
        actions[0] = trainee_action

        # Physics substeps
        collided_this_tick = [False] * self.num_horses
        for _ in range(PHYS_SUBSTEPS):
            # Compute drafting bonuses
            progresses = [
                self._navigators[i].compute_progress(self._bodies[i].position)
                for i in range(self.num_horses)
            ]
            draft_bonuses = [
                compute_drafting_bonus(
                    i, self._bodies, self._frames, progresses, self._track_length
                )
                for i in range(self.num_horses)
            ]

            # Step each horse
            for i in range(self.num_horses):
                if self._finished[i]:
                    continue

                cornering_excess, _ = step_horse(
                    self._bodies[i],
                    self._profiles[i],
                    self._staminas[i],
                    actions[i],
                    self._frames[i],
                    self._substep_dt,
                    self._sim_time,
                    draft_bonus=draft_bonuses[i],
                )

                # Drain stamina
                tang_speed = float(np.dot(
                    self._bodies[i].velocity,
                    self._frames[i].tangential,
                ))
                norm_speed = float(np.dot(
                    self._bodies[i].velocity,
                    self._frames[i].normal,
                ))
                drain = compute_drain(
                    self._profiles[i],
                    abs(tang_speed),
                    abs(norm_speed),
                    cornering_excess,
                    self._substep_dt,
                )
                self._staminas[i].drain(drain)

            # Resolve collisions
            masses = [p.weight for p in self._profiles]
            horse_collisions = resolve_horse_collisions(
                self._bodies, masses,
            )
            for i, c in enumerate(horse_collisions):
                if c:
                    collided_this_tick[i] = True

            resolve_wall_collisions(
                self._bodies,
                self._track_data.segments,
                [n.segment_index for n in self._navigators],
            )

        # Update frames and check for finish
        self._sim_time += self._dt
        for i in range(self.num_horses):
            self._frames[i] = self._navigators[i].update(self._bodies[i].position)
            if not self._finished[i] and self._navigators[i].completed_lap:
                self._finished[i] = True
                self._finish_order.append(i)

        # Compute placement
        progresses = [
            self._navigators[i].compute_progress(self._bodies[i].position)
            for i in range(self.num_horses)
        ]
        sorted_indices = sorted(range(self.num_horses), key=lambda i: -progresses[i])
        placements = [0] * self.num_horses
        for rank, idx in enumerate(sorted_indices):
            placements[idx] = rank

        # Track placement at 75% for closer reward
        if self._placement_at_75 is None and progresses[0] >= 0.75:
            self._placement_at_75 = placements[0]

        # Check leading at checkpoints
        leading_at_25 = progresses[0] >= 0.25 and placements[0] == 0
        leading_at_50 = progresses[0] >= 0.50 and placements[0] == 0

        # Positions gained in final stretch
        positions_gained = 0
        if self._placement_at_75 is not None:
            positions_gained = max(0, self._placement_at_75 - placements[0])

        # Compute reward for trainee (horse 0)
        reward = compute_reward(
            placement=placements[0],
            num_horses=self.num_horses,
            progress=progresses[0],
            prev_progress=self._prev_progress,
            collided=collided_this_tick[0],
            finished=self._finished[0],
            jockey_style=self._jockey_style,
            stamina=self._staminas[0],
            leading_at_25=leading_at_25,
            leading_at_50=leading_at_50,
            positions_gained_final_25=positions_gained,
        )

        self._prev_progress = progresses[0]
        self._prev_placement = placements[0]

        # Termination conditions
        terminated = self._finished[0]
        truncated = self._step_count >= self.max_steps

        obs = self._build_obs()
        info = self._build_info()

        return obs, reward, terminated, truncated, info

    def _build_obs(self) -> np.ndarray:
        return build_observation(
            horse_idx=0,
            bodies=self._bodies,
            profiles=self._profiles,
            staminas=self._staminas,
            frames=self._frames,
            navigators=self._navigators,
            jockey_style=self._jockey_style,
            num_horses=self.num_horses,
            placement=self._prev_placement,
            sim_time=self._sim_time,
            total_race_time_est=self._est_race_time,
        )

    def _build_info(self) -> dict:
        return {
            "step": self._step_count,
            "progress": self._navigators[0].compute_progress(self._bodies[0].position),
            "stamina_ratio": self._staminas[0].ratio,
            "placement": self._prev_placement,
            "finished": self._finished[0],
            "jockey_style": {
                "risk_tolerance": self._jockey_style.risk_tolerance,
                "tactical_bias": self._jockey_style.tactical_bias,
                "skill_level": self._jockey_style.skill_level,
            },
        }
