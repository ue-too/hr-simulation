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
    # obs speed_ratio is tvel/max_speed (20). Natural cruise ~13 m/s → ratio ~0.65.
    # Band centered on natural cruise, wide enough to avoid constant push/coast flip.
    cruise_low: float = 0.55    # ~11 m/s
    cruise_high: float = 0.70   # ~14 m/s
    kick_phase: float = 0.75
    block_progress_max: float = 0.03
    block_lateral_tol: float = 0.15
    # Blocker must be meaningfully slower (not just "slower by anything")
    block_min_slowness: float = 0.03
    conserve_threshold: float = 0.30
    # Passing commitment — minimum ticks to stay in pass maneuver
    pass_min_ticks: int = 40
    # Done passing once this far ahead in lateral offset (normalized)
    pass_clear_lateral: float = 0.25
    # Cooldown ticks after finishing a pass before another can start
    pass_cooldown_ticks: int = 80


# ============================================================
# Archetypes — presets for different racing styles.
# Each returns a BTConfig tuned for a specific role.
# ============================================================

def archetype_stalker() -> BTConfig:
    """Classic stalker: sits 2nd-4th, kicks at the quarter pole. Balanced."""
    return BTConfig()


def archetype_front_runner() -> BTConfig:
    """Front-runner: pushes to the lead early, tries to hold. May fade."""
    return BTConfig(
        cruise_low=0.65,        # ~13 m/s — faster cruise
        cruise_high=0.80,       # ~16 m/s
        kick_phase=0.65,        # earlier kick
        block_min_slowness=0.01,  # more aggressive passing
        pass_cooldown_ticks=40,
    )


def archetype_closer() -> BTConfig:
    """Closer: hangs back, makes dramatic late run."""
    return BTConfig(
        cruise_low=0.45,        # ~9 m/s — very conservative
        cruise_high=0.60,       # ~12 m/s
        kick_phase=0.85,        # late kick
        conserve_threshold=0.50, # holds back longer
    )


def archetype_speedball() -> BTConfig:
    """Aggressive passer: constantly tries to move up through the field."""
    return BTConfig(
        cruise_low=0.60,
        cruise_high=0.75,
        kick_phase=0.70,
        block_min_slowness=0.005, # passes on any slower horse
        pass_min_ticks=30,
        pass_cooldown_ticks=30,
    )


def archetype_steady() -> BTConfig:
    """Steady cruiser: narrow band, mid-pack finisher. Rarely passes."""
    return BTConfig(
        cruise_low=0.58,
        cruise_high=0.68,
        kick_phase=0.80,
        block_min_slowness=0.08, # only passes much-slower horses
        pass_cooldown_ticks=150,
    )


ARCHETYPES = {
    "stalker": archetype_stalker,
    "front-runner": archetype_front_runner,
    "closer": archetype_closer,
    "speedball": archetype_speedball,
    "steady": archetype_steady,
}


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
        self._cooldown_ticks = 0  # remaining cooldown ticks after a pass

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
                self._cooldown_ticks = cfg.pass_cooldown_ticks
            else:
                return self._do_pass(stamina_frac)

        # CRUISE state — check for blocker to switch to passing
        if self._state == self.STATE_CRUISE:
            if self._cooldown_ticks > 0:
                self._cooldown_ticks -= 1
            elif self._is_blocked(obs):
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
        # Default cruise effort is 0.25 (physics settles near cruise speed).
        # Use a wider tolerance so minor overshoots don't trigger coast.
        if speed_ratio < cfg.cruise_low - 0.05:
            tang = 0.5   # gentle push
        elif speed_ratio > cfg.cruise_high + 0.05:
            tang = 0.0   # coast
        else:
            tang = 0.25  # maintain (most of the time)
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
            # Blocker must be meaningfully slower, not just slower by epsilon
            if tvel_delta >= -cfg.block_min_slowness:
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
