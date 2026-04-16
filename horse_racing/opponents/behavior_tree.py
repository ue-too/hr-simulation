"""Behavior-tree opponent: reactive, tactical jockey with racing knowledge.

Operates on the same observation vector that the RL agent sees (141 floats).
This ensures the BT only uses information the agent has access to — if the
BT can win from this obs, the agent should be able to learn the same tactics.

Observation layout (from core/observation.py):
- [0]     track_progress
- [1]     tangential_vel / max_speed
- [2]     normal_vel / max_speed
- [3]     stamina / max_stamina
- [4-7]   effective degradation ratios
- [8-13]  normalized trait values
- [14]    last_drain / max_stamina
- [15]    lateral_offset / TRACK_HALF_WIDTH   (negative = toward inside)
- [16-17] curvature, slope at current position
- [18-25] 4× (lookahead curvature, slope)
- [26-140] 23× opponent slots of 5 values each:
    [0] active (1.0 if filled)
    [1] progress_delta (opp.progress - self.progress)
    [2] (opp.tvel - self.tvel) / max_speed
    [3] normal_offset (opp.lateral - self.lateral, normalized by TRACK_HALF_WIDTH)
    [4] (opp.nvel - self.nvel) / max_speed
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
    # Note: obs[1] is tvel / max_speed (20). cruise_speed = 13, so cruise = 0.65 ratio.
    # Band: 65-80% of cruise speed → 8.45-10.4 m/s → 0.42-0.52 ratio.
    cruise_low: float = 0.42
    cruise_high: float = 0.52
    kick_phase: float = 0.75    # when to start sprinting
    # Blocker detection (using obs opponent slots)
    block_progress_max: float = 0.03    # blocker within 3% ahead
    block_lateral_tol: float = 0.15     # norm_offset within this means same lane
    # Stamina
    conserve_threshold: float = 0.30


class BehaviorTreeStrategy(Strategy):
    """Reactive BT-style opponent using only the agent's observation vector.

    Priority-ordered decisions each tick:
    1. Final kick — if past kick_phase, sprint toward inside rail
    2. Unbox — if a slower opponent is directly ahead, swing wide
    3. Rail discipline — hold inside line if clear
    4. Cruise pacing — maintain speed in cruise band
    """

    def __init__(
        self,
        race_ref: "Race",
        horse_id: int,
        config: BTConfig | None = None,
    ):
        self._race = race_ref
        self._horse_id = horse_id
        self._cfg = config or BTConfig()

    def act(self, progress: float) -> int:
        return 0

    def act_continuous(self, horse: "Horse") -> "InputState | None":
        from ..core.types import InputState

        # Build the observation for this horse (same as what agent sees)
        all_obs = build_observations(self._race)
        obs = all_obs[self._horse_id]

        cfg = self._cfg
        progress = float(obs[0])
        speed_ratio = float(obs[1])           # tvel / max_speed
        stamina_frac = float(obs[3])
        lateral_norm = float(obs[15])         # lateral / TRACK_HALF_WIDTH

        # --- Phase 1: Final kick ---
        if progress >= cfg.kick_phase:
            tang = 1.0
            normal = self._kick_lane(obs, lateral_norm)
            return InputState(tang, normal)

        # --- Phase 2: Unbox — blocked by slower opponent ---
        if self._is_blocked(obs):
            # Swing wide to pass
            tang = 0.75 if stamina_frac > cfg.conserve_threshold else 0.5
            normal = 0.5  # outward (away from inside rail)
            return InputState(tang, normal)

        # --- Phase 3: Cruise + hold inside ---
        if speed_ratio < cfg.cruise_low:
            tang = 0.75
        elif speed_ratio > cfg.cruise_high:
            tang = 0.0
        else:
            tang = 0.25

        if stamina_frac < cfg.conserve_threshold:
            tang = min(tang, 0.25)

        # lateral_norm is negative near inside rail. Push in if not close.
        # Inside rail at -0.95 after 0.95 × TRACK_HALF_WIDTH normalization
        if lateral_norm > -0.75:
            normal = -0.75
        else:
            normal = -0.25

        return InputState(tang, normal)

    # -------- Obs-based perception --------

    def _is_blocked(self, obs: "np.ndarray") -> bool:
        """Check if an opponent is directly ahead in the same lane, moving slower."""
        cfg = self._cfg
        for s in range(OPPONENT_SLOTS):
            base = _OPP_BASE + s * OPPONENT_SLOT_SIZE
            active = obs[base + 0]
            if active < 0.5:
                continue
            progress_delta = obs[base + 1]      # positive = opp ahead
            tvel_delta = obs[base + 2]          # negative = opp slower
            normal_offset = obs[base + 3]       # lateral distance, normalized
            if not (0.0 < progress_delta < cfg.block_progress_max):
                continue
            if abs(normal_offset) > cfg.block_lateral_tol:
                continue
            if tvel_delta >= 0:
                continue
            return True
        return False

    def _kick_lane(self, obs: "np.ndarray", lateral_norm: float) -> float:
        """During kick phase: go wide if blocked, else inside."""
        if self._is_blocked(obs):
            return 0.5
        if lateral_norm > -0.75:
            return -0.75
        return -0.25
