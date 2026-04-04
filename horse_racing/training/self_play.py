"""Self-play environment — trainee vs frozen ONNX opponents with GRU state.

Controls horse 0 (trainee). Opponents are driven by frozen ONNX models that
include GRU recurrent state. Each reset() samples a random track, random
field size, and random opponent models.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import onnxruntime as ort
from gymnasium import spaces

from horse_racing.core.horse import HorseProfile, random_horse
from horse_racing.core.physics import (
    compute_drafting_bonus,
    resolve_horse_collisions,
    resolve_wall_collisions,
    step_horse,
)
from horse_racing.core.stamina import StaminaState, compute_drain, create_stamina
from horse_racing.core.track import compute_total_length, load_track
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


class _OnnxOpponent:
    """Wrapper for a frozen ONNX opponent with GRU hidden state."""

    def __init__(self, session: ort.InferenceSession, gru_size: int = 128) -> None:
        self.session = session
        self.gru_size = gru_size

        # Detect input names to determine if model has GRU state
        self._input_names = [inp.name for inp in session.get_inputs()]
        self._output_names = [out.name for out in session.get_outputs()]
        self._has_gru = len(self._input_names) > 1

        self.hidden_state = np.zeros((1, gru_size), dtype=np.float32)

    def reset(self) -> None:
        self.hidden_state = np.zeros((1, self.gru_size), dtype=np.float32)

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Run inference. Returns action [effort, lane]."""
        obs_input = obs.reshape(1, -1).astype(np.float32)

        if self._has_gru:
            feeds = {
                self._input_names[0]: obs_input,
                self._input_names[1]: self.hidden_state,
            }
            outputs = self.session.run(self._output_names, feeds)
            action = outputs[0][0]
            self.hidden_state = outputs[1]
        else:
            feeds = {self._input_names[0]: obs_input}
            outputs = self.session.run(self._output_names, feeds)
            action = outputs[0][0]

        return np.clip(action, -1.0, 1.0)


class SelfPlayEnv(gym.Env):
    """Single-agent env where opponents are driven by frozen ONNX models."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        tracks: list[str] | str,
        max_steps: int = 6000,
        opponent_onnx_paths: list[str] | None = None,
        min_opponents: int = 3,
        max_opponents: int = 19,
        gru_size: int = 128,
        render_mode: str | None = None,
        stagger_range: tuple[float, float] = (0.0, 0.0),
        randomize_jockey_style: bool = True,
    ) -> None:
        super().__init__()
        self.tracks = [tracks] if isinstance(tracks, str) else list(tracks)
        self.max_steps = max_steps
        self.min_opponents = min_opponents
        self.max_opponents = max_opponents
        self.render_mode = render_mode
        self.stagger_range = stagger_range
        self.randomize_jockey_style = randomize_jockey_style

        # Load opponent ONNX sessions
        self._opponent_pool: list[_OnnxOpponent] = []
        if opponent_onnx_paths:
            for path in opponent_onnx_paths:
                session = ort.InferenceSession(
                    path, providers=["CPUExecutionProvider"]
                )
                self._opponent_pool.append(_OnnxOpponent(session, gru_size))

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32,
        )

        # State
        self._rng = random.Random()
        self._profiles: list[HorseProfile] = []
        self._bodies: list[HorseBody] = []
        self._staminas: list[StaminaState] = []
        self._navigators: list[TrackNavigator] = []
        self._frames: list[TrackFrame] = []
        self._jockey_styles: list[JockeyStyle] = []
        self._active_opponents: list[_OnnxOpponent | None] = []
        self._num_horses = 0
        self._step_count = 0
        self._sim_time = 0.0
        self._track_data = None
        self._track_length = 0.0
        self._est_race_time = 0.0
        self._finished: list[bool] = []
        self._finish_order: list[int] = []
        self._prev_progress = 0.0
        self._prev_placement = 0
        self._placement_at_75: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._rng = random.Random(seed)
        self._step_count = 0
        self._sim_time = 0.0
        self._finish_order = []
        self._prev_progress = 0.0
        self._prev_placement = 0
        self._placement_at_75 = None

        # Random track
        track_path = self._rng.choice(self.tracks)
        self._track_data = load_track(track_path)
        self._track_length = compute_total_length(self._track_data.segments)
        self._est_race_time = self._track_length / 16.0

        # Random number of opponents
        max_opp = min(self.max_opponents, len(self._opponent_pool)) if self._opponent_pool else self.max_opponents
        num_opponents = self._rng.randint(
            min(self.min_opponents, max_opp),
            max_opp,
        )
        self._num_horses = 1 + num_opponents
        self._finished = [False] * self._num_horses

        # Generate horse profiles
        self._profiles = [random_horse(self._rng) for _ in range(self._num_horses)]

        # Generate jockey styles
        self._jockey_styles = []
        for i in range(self._num_horses):
            if self.randomize_jockey_style:
                self._jockey_styles.append(JockeyStyle(
                    risk_tolerance=self._rng.random(),
                    tactical_bias=self._rng.uniform(-1, 1),
                    skill_level=self._rng.random(),
                ))
            else:
                self._jockey_styles.append(JockeyStyle())

        # Assign ONNX opponents (sample with replacement)
        self._active_opponents = [None]  # trainee has no ONNX model
        for _ in range(num_opponents):
            if self._opponent_pool:
                opp = self._rng.choice(self._opponent_pool)
                opp.reset()
                self._active_opponents.append(opp)
            else:
                self._active_opponents.append(None)  # zero-action fallback

        # Initialize bodies
        seg0 = self._track_data.segments[0]
        start_pos = seg0.start_point
        dx = seg0.end_point[0] - seg0.start_point[0]
        dy = seg0.end_point[1] - seg0.start_point[1]
        start_orientation = math.atan2(dy, dx)

        self._bodies = []
        self._navigators = []
        for i in range(self._num_horses):
            lateral_offset = (i - (self._num_horses - 1) / 2.0) * HORSE_SPACING

            # Stagger: opponents start ahead
            stagger = 0.0
            if i > 0 and self.stagger_range[1] > 0:
                stagger = self._rng.uniform(*self.stagger_range)

            body = HorseBody()
            fwd = np.array([math.cos(start_orientation), math.sin(start_orientation)])
            lateral = np.array([-math.sin(start_orientation), math.cos(start_orientation)])
            body.position = np.array([
                start_pos[0] + fwd[0] * stagger + lateral[0] * lateral_offset,
                start_pos[1] + fwd[1] * stagger + lateral[1] * lateral_offset,
            ], dtype=np.float64)
            body.orientation = start_orientation
            self._bodies.append(body)
            self._navigators.append(TrackNavigator(self._track_data.segments))

        self._staminas = [create_stamina(p) for p in self._profiles]
        self._frames = [
            self._navigators[i].update(self._bodies[i].position)
            for i in range(self._num_horses)
        ]

        obs = self._build_obs(0)
        info = self._build_info()
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1

        # Build actions: trainee + ONNX opponents
        actions = [JockeyAction()] * self._num_horses
        actions[0] = JockeyAction(
            effort=float(np.clip(action[0], -1.0, 1.0)),
            lane=float(np.clip(action[1], -1.0, 1.0)),
        )

        # Get opponent actions from ONNX models
        for i in range(1, self._num_horses):
            opp = self._active_opponents[i]
            if opp is not None and not self._finished[i]:
                opp_obs = self._build_obs(i)
                opp_action = opp.infer(opp_obs)
                actions[i] = JockeyAction(
                    effort=float(opp_action[0]),
                    lane=float(opp_action[1]),
                )

        # Physics substeps
        collided_this_tick = [False] * self._num_horses
        for _ in range(PHYS_SUBSTEPS):
            progresses = [
                self._navigators[i].compute_progress(self._bodies[i].position)
                for i in range(self._num_horses)
            ]
            draft_bonuses = [
                compute_drafting_bonus(
                    i, self._bodies, self._frames, progresses, self._track_length
                )
                for i in range(self._num_horses)
            ]

            for i in range(self._num_horses):
                if self._finished[i]:
                    continue
                substep_dt = 1.0 / (PHYS_HZ * PHYS_SUBSTEPS)
                cornering_excess, _ = step_horse(
                    self._bodies[i], self._profiles[i], self._staminas[i],
                    actions[i], self._frames[i], substep_dt, self._sim_time,
                    draft_bonus=draft_bonuses[i],
                )
                tang_speed = float(np.dot(
                    self._bodies[i].velocity, self._frames[i].tangential
                ))
                norm_speed = float(np.dot(
                    self._bodies[i].velocity, self._frames[i].normal
                ))
                drain = compute_drain(
                    self._profiles[i], abs(tang_speed), abs(norm_speed),
                    cornering_excess, substep_dt,
                )
                self._staminas[i].drain(drain)

            masses = [p.weight for p in self._profiles]
            horse_collisions = resolve_horse_collisions(self._bodies, masses)
            for i, c in enumerate(horse_collisions):
                if c:
                    collided_this_tick[i] = True

            resolve_wall_collisions(
                self._bodies, self._track_data.segments,
                [n.segment_index for n in self._navigators],
            )

        # Update frames and check finish
        self._sim_time += 1.0 / PHYS_HZ
        for i in range(self._num_horses):
            self._frames[i] = self._navigators[i].update(self._bodies[i].position)
            if not self._finished[i] and self._navigators[i].completed_lap:
                self._finished[i] = True
                self._finish_order.append(i)

        # Compute placements
        progresses = [
            self._navigators[i].compute_progress(self._bodies[i].position)
            for i in range(self._num_horses)
        ]
        sorted_indices = sorted(range(self._num_horses), key=lambda i: -progresses[i])
        placements = [0] * self._num_horses
        for rank, idx in enumerate(sorted_indices):
            placements[idx] = rank

        # Track checkpoints
        if self._placement_at_75 is None and progresses[0] >= 0.75:
            self._placement_at_75 = placements[0]

        leading_at_25 = progresses[0] >= 0.25 and placements[0] == 0
        leading_at_50 = progresses[0] >= 0.50 and placements[0] == 0
        positions_gained = 0
        if self._placement_at_75 is not None:
            positions_gained = max(0, self._placement_at_75 - placements[0])

        # Reward
        reward = compute_reward(
            placement=placements[0],
            num_horses=self._num_horses,
            progress=progresses[0],
            prev_progress=self._prev_progress,
            collided=collided_this_tick[0],
            finished=self._finished[0],
            jockey_style=self._jockey_styles[0],
            stamina=self._staminas[0],
            leading_at_25=leading_at_25,
            leading_at_50=leading_at_50,
            positions_gained_final_25=positions_gained,
        )

        self._prev_progress = progresses[0]
        self._prev_placement = placements[0]

        terminated = self._finished[0]
        truncated = self._step_count >= self.max_steps

        obs = self._build_obs(0)
        info = self._build_info()

        return obs, reward, terminated, truncated, info

    def _build_obs(self, horse_idx: int) -> np.ndarray:
        return build_observation(
            horse_idx=horse_idx,
            bodies=self._bodies,
            profiles=self._profiles,
            staminas=self._staminas,
            frames=self._frames,
            navigators=self._navigators,
            jockey_style=self._jockey_styles[horse_idx],
            num_horses=self._num_horses,
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
            "num_opponents": self._num_horses - 1,
            "jockey_style": {
                "risk_tolerance": self._jockey_styles[0].risk_tolerance,
                "tactical_bias": self._jockey_styles[0].tactical_bias,
                "skill_level": self._jockey_styles[0].skill_level,
            },
        }
