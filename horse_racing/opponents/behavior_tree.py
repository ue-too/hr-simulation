"""Behavior-tree opponent: reactive, tactical jockey with racing knowledge.

Uses the same observation vector the RL agent sees. Holds per-horse state
so maneuvers (passing, kicking) commit for multiple ticks instead of
recalculating from scratch every frame.

Observation layout (from core/observation.py):
- [0]     track_progress
- [1]     tangential_vel / max_speed
- [3]     stamina / max_stamina
- [15]    lateral_offset / TRACK_HALF_WIDTH   (negative = toward inside)
- [26+]   23× opponent slots of 5 values:
    [0] active, [1] progress_delta, [2] tvel_delta, [3] norm_offset, [4] nvel_delta
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..core.observation import (
    OPPONENT_SLOT_SIZE,
    OPPONENT_SLOTS,
    SELF_STATE_SIZE,
    TRACK_CONTEXT_SIZE,
    build_observations,
)
from .scripted import Strategy

if TYPE_CHECKING:
    from ..core.race import Race
    from ..core.types import Horse, InputState


_OPP_BASE = SELF_STATE_SIZE + TRACK_CONTEXT_SIZE  # 26


@dataclass
class BTConfig:
    """Tunable parameters for the BT opponent."""
    cruise_low: float = 0.325   # obs speed ratio (tvel/max_speed)
    cruise_high: float = 0.4
    kick_phase: float = 0.75
    block_progress_max: float = 0.03
    block_lateral_tol: float = 0.15
    conserve_threshold: float = 0.30
    # Passing commitment — minimum ticks to stay in pass maneuver
    pass_min_ticks: int = 40
    # Done passing once this far ahead in lateral offset (normalized)
    pass_clear_lateral: float = 0.25


class BehaviorTreeStrategy(Strategy):
    """Reactive BT jockey with committed maneuvers.

    States:
    - CRUISE: hold inside line in speed band
    - PASSING: swung wide to overtake a blocker — commits for pass_min_ticks
    - KICK: final sprint, pull inside unless blocked
    """

    STATE_CRUISE = 0
    STATE_PASSING = 1
    STATE_KICK = 2

    def __init__(
        self,
        race_ref: "Race",
        horse_id: int,
        config: BTConfig | None = None,
    ):
        self._race = race_ref
        self._horse_id = horse_id
        self._cfg = config or BTConfig()
        self._state = self.STATE_CRUISE
        self._state_ticks = 0  # ticks spent in current state

    def act(self, progress: float) -> int:
        return 0

    def act_continuous(self, horse: "Horse") -> "InputState | None":
        from ..core.types import InputState

        cfg = self._cfg
        all_obs = build_observations(self._race)
        obs = all_obs[self._horse_id]

        progress = float(obs[0])
        speed_ratio = float(obs[1])
        stamina_frac = float(obs[3])
        lateral_norm = float(obs[15])

        # Transition to KICK when past kick phase (absorbing)
        if progress >= cfg.kick_phase:
            if self._state != self.STATE_KICK:
                self._state = self.STATE_KICK
                self._state_ticks = 0

        # State machine
        if self._state == self.STATE_KICK:
            self._state_ticks += 1
            return self._do_kick(obs, lateral_norm)

        if self._state == self.STATE_PASSING:
            self._state_ticks += 1
            # Exit passing when: committed ticks elapsed AND clear of blocker
            if self._state_ticks >= cfg.pass_min_ticks and not self._still_blocked(obs):
                self._state = self.STATE_CRUISE
                self._state_ticks = 0
            else:
                return self._do_pass(stamina_frac)

        # CRUISE state — check for blocker to switch to passing
        if self._state == self.STATE_CRUISE:
            if self._is_blocked(obs):
                self._state = self.STATE_PASSING
                self._state_ticks = 0
                return self._do_pass(stamina_frac)
            self._state_ticks += 1
            return self._do_cruise(speed_ratio, stamina_frac, lateral_norm)

        # Fallback
        return InputState(0.25, -0.25)

    # -------- Actions per state --------

    def _do_cruise(self, speed_ratio: float, stamina_frac: float, lateral_norm: float) -> "InputState":
        from ..core.types import InputState
        cfg = self._cfg
        if speed_ratio < cfg.cruise_low:
            tang = 0.75
        elif speed_ratio > cfg.cruise_high:
            tang = 0.0
        else:
            tang = 0.25
        if stamina_frac < cfg.conserve_threshold:
            tang = min(tang, 0.25)
        # Pull to inside rail. lateral_norm near -0.95 = inside rail.
        if lateral_norm > -0.80:
            normal = -0.5
        else:
            normal = -0.25
        return InputState(tang, normal)

    def _do_pass(self, stamina_frac: float) -> "InputState":
        from ..core.types import InputState
        cfg = self._cfg
        tang = 0.75 if stamina_frac > cfg.conserve_threshold else 0.5
        # Commit wide — stay there until maneuver ends
        return InputState(tang, 0.5)

    def _do_kick(self, obs: "np.ndarray", lateral_norm: float) -> "InputState":
        from ..core.types import InputState
        if self._is_blocked(obs):
            return InputState(1.0, 0.5)
        # Pull to inside line for optimal path
        if lateral_norm > -0.80:
            normal = -0.5
        else:
            normal = -0.25
        return InputState(1.0, normal)

    # -------- Perception helpers --------

    def _is_blocked(self, obs: "np.ndarray") -> bool:
        """Opponent directly ahead in same lane, moving slower."""
        cfg = self._cfg
        for s in range(OPPONENT_SLOTS):
            base = _OPP_BASE + s * OPPONENT_SLOT_SIZE
            if obs[base + 0] < 0.5:
                continue
            progress_delta = obs[base + 1]
            tvel_delta = obs[base + 2]
            normal_offset = obs[base + 3]
            if not (0.0 < progress_delta < cfg.block_progress_max):
                continue
            if abs(normal_offset) > cfg.block_lateral_tol:
                continue
            if tvel_delta >= 0:
                continue
            return True
        return False

    def _still_blocked(self, obs: "np.ndarray") -> bool:
        """Looser check during passing — any opponent at similar progress?"""
        cfg = self._cfg
        for s in range(OPPONENT_SLOTS):
            base = _OPP_BASE + s * OPPONENT_SLOT_SIZE
            if obs[base + 0] < 0.5:
                continue
            progress_delta = obs[base + 1]
            normal_offset = obs[base + 3]
            # Still blocked if an opponent is level-to-just-ahead on the inside
            if -0.01 < progress_delta < cfg.block_progress_max:
                if normal_offset < -cfg.pass_clear_lateral:
                    return True
        return False
