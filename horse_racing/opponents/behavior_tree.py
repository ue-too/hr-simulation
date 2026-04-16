"""Utility-scored opponent: reactive, tactical jockey with racing knowledge.

Uses the same observation vector the RL agent sees. Holds per-horse state
so maneuvers (passing, kicking) commit for multiple ticks instead of
recalculating from scratch every frame.

Observation layout (from core/observation.py):
- [0]     track_progress
- [1]     tangential_vel / max_speed
- [3]     stamina / max_stamina
- [15]    lateral_offset / TRACK_HALF_WIDTH   (negative = toward inside)
- [26+]   23x opponent slots of 5 values:
    [0] active, [1] progress_delta, [2] tvel_delta, [3] norm_offset, [4] nvel_delta

States: CRUISE / PASSING / KICK / SETTLING.
Each tick the utility selector scores CRUISE, PASS, and KICK; the highest
wins (subject to commitment windows and transition budgets). A defensive
overlay nudges outputs when an opponent threatens to pass, without adding
a dedicated state.

Archetypes are expressed as weight profiles over the scoring functions
plus a few direct constants (target lane, cruise band, kick timing).
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
    """Tunable parameters for the utility-scored opponent.

    Keep **field names and defaults** in sync with ``bt-jockey.ts`` (``BTConfig`` +
    ``DEFAULT_CONFIG``) and archetype tables in the ue-too horse-racing app.
    """
    cruise_low: float = 0.55
    cruise_high: float = 0.70
    target_lane: float = -0.80
    lateral_aggression: float = 0.6
    kick_phase: float = 0.75
    kick_early_margin: float = 0.10
    kick_late_cap: float = 0.92
    block_progress_max: float = 0.03
    block_lateral_tol: float = 0.15
    block_min_slowness: float = 0.03
    conserve_threshold: float = 0.30
    pass_min_ticks: int = 40
    pass_clear_lateral: float = 0.25
    pass_cooldown_ticks: int = 80
    settle_ticks: int = 40
    transition_min_ticks: int = 30
    defend_on_score: float = 0.6
    defend_off_score: float = 0.3
    defend_tang_min: float = 0.5
    defend_drift: float = 0.15
    w_pass: float = 1.0
    w_kick: float = 1.0
    w_draft: float = 1.0
    off_lane_penalty_start: float = 0.06
    off_lane_tang_penalty_scale: float = 0.5
    off_lane_tang_penalty_max: float = 0.18
    off_lane_decel_scale: float = 1.0
    off_lane_accel_relief: float = 0.0


# ============================================================
# Archetypes — weight profiles for different racing styles.
# ============================================================

def archetype_stalker() -> BTConfig:
    """Classic stalker: sits 2nd-4th, kicks at the quarter pole. Balanced."""
    return BTConfig(
        target_lane=-0.60,
        lateral_aggression=0.5,
        w_draft=1.3,
        off_lane_penalty_start=0.06,
        off_lane_tang_penalty_max=0.16,
        off_lane_decel_scale=1.0,
        off_lane_accel_relief=0.03,
    )


def archetype_front_runner() -> BTConfig:
    """Front-runner: pushes to the lead early, tries to hold. May fade."""
    return BTConfig(
        cruise_low=0.72,
        cruise_high=0.85,
        target_lane=-0.80,
        lateral_aggression=0.8,
        kick_phase=0.65,
        kick_early_margin=0.05,
        kick_late_cap=0.88,
        block_min_slowness=0.01,
        pass_cooldown_ticks=40,
        defend_drift=0.20,
        w_pass=1.3,
        w_kick=1.2,
        w_draft=0.5,
        off_lane_penalty_start=0.10,
        off_lane_tang_penalty_scale=0.35,
        off_lane_tang_penalty_max=0.10,
        off_lane_decel_scale=0.75,
        off_lane_accel_relief=0.07,
    )


def archetype_closer() -> BTConfig:
    """Closer: hangs back, makes dramatic late run."""
    return BTConfig(
        cruise_low=0.40,
        cruise_high=0.52,
        target_lane=-0.30,
        lateral_aggression=0.4,
        kick_phase=0.85,
        kick_early_margin=0.05,
        kick_late_cap=0.93,
        conserve_threshold=0.50,
        settle_ticks=50,
        defend_on_score=0.8,
        w_pass=0.7,
        w_kick=1.5,
        w_draft=1.5,
        off_lane_penalty_start=0.04,
        off_lane_tang_penalty_scale=0.65,
        off_lane_tang_penalty_max=0.24,
        off_lane_decel_scale=1.25,
        off_lane_accel_relief=0.0,
    )


def archetype_speedball() -> BTConfig:
    """Aggressive passer: constantly tries to move up through the field."""
    return BTConfig(
        cruise_low=0.60,
        cruise_high=0.75,
        target_lane=-0.20,
        lateral_aggression=0.8,
        kick_phase=0.70,
        kick_early_margin=0.10,
        kick_late_cap=0.88,
        block_min_slowness=0.005,
        pass_min_ticks=30,
        pass_cooldown_ticks=30,
        w_pass=1.5,
        w_kick=0.9,
        w_draft=0.6,
        off_lane_penalty_start=0.08,
        off_lane_tang_penalty_scale=0.38,
        off_lane_tang_penalty_max=0.10,
        off_lane_decel_scale=0.7,
        off_lane_accel_relief=0.09,
    )


def archetype_steady() -> BTConfig:
    """Steady cruiser: narrow band, mid-pack finisher. Rarely passes."""
    return BTConfig(
        cruise_low=0.58,
        cruise_high=0.68,
        target_lane=-0.70,
        lateral_aggression=0.5,
        kick_phase=0.80,
        block_min_slowness=0.08,
        pass_cooldown_ticks=150,
        defend_on_score=0.9,
        w_pass=0.5,
        w_kick=0.8,
        w_draft=1.0,
        off_lane_penalty_start=0.07,
        off_lane_tang_penalty_max=0.14,
        off_lane_decel_scale=1.05,
        off_lane_accel_relief=0.02,
    )


def archetype_drifter() -> BTConfig:
    """Drifter: mid-pack lane, draft-leaning; template for custom tuning."""
    return BTConfig(
        cruise_low=0.52,
        cruise_high=0.65,
        target_lane=-0.45,
        lateral_aggression=0.55,
        kick_phase=0.78,
        w_pass=1.0,
        w_kick=1.05,
        w_draft=1.2,
        off_lane_penalty_start=0.055,
        off_lane_tang_penalty_scale=0.42,
        off_lane_tang_penalty_max=0.14,
        off_lane_decel_scale=0.95,
        off_lane_accel_relief=0.04,
    )


ARCHETYPES = {
    "stalker": archetype_stalker,
    "front-runner": archetype_front_runner,
    "closer": archetype_closer,
    "speedball": archetype_speedball,
    "steady": archetype_steady,
    "drifter": archetype_drifter,
}


class BehaviorTreeStrategy(Strategy):
    """Utility-scored jockey with committed maneuvers and reactive overlays.

    States:
    - CRUISE: maintain speed band; steer toward archetype lane.
    - PASSING: commit wide + extra tangential (committed for pass_min_ticks).
    - KICK: max tangential sprint, pull inside unless blocked.
    - SETTLING: after PASSING, interpolate lane back toward archetype target.
    """

    STATE_CRUISE = 0
    STATE_PASSING = 1
    STATE_KICK = 2
    STATE_SETTLING = 3

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
        self._state_ticks = 0
        self._cooldown_ticks = 0
        self._global_tick = 0
        self._last_transition_tick = -999
        self._defending = False
        self._settle_from_lane = 0.0

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
        self._global_tick += 1

        # KICK is absorbing
        if self._state == self.STATE_KICK:
            self._state_ticks += 1
            return self._apply_defense(
                self._do_kick(obs, lateral_norm), obs, stamina_frac
            )

        # PASSING: committed for pass_min_ticks
        if self._state == self.STATE_PASSING:
            self._state_ticks += 1
            if progress >= cfg.kick_late_cap:
                self._transition(self.STATE_KICK)
                return self._apply_defense(
                    self._do_kick(obs, lateral_norm), obs, stamina_frac
                )
            if self._state_ticks >= cfg.pass_min_ticks and not self._still_blocked(obs):
                self._transition(self.STATE_SETTLING)
                self._settle_from_lane = lateral_norm
            else:
                return self._apply_defense(
                    self._do_pass(stamina_frac), obs, stamina_frac
                )

        # SETTLING: interpolate lane position back toward archetype target
        if self._state == self.STATE_SETTLING:
            self._state_ticks += 1
            if progress >= cfg.kick_late_cap:
                self._transition(self.STATE_KICK)
                return self._apply_defense(
                    self._do_kick(obs, lateral_norm), obs, stamina_frac
                )
            if self._state_ticks >= cfg.settle_ticks:
                self._transition(self.STATE_CRUISE)
                self._cooldown_ticks = cfg.pass_cooldown_ticks
            else:
                return self._apply_defense(
                    self._do_settle(speed_ratio, stamina_frac, lateral_norm),
                    obs, stamina_frac,
                )

        # CRUISE: utility-based action selection
        if self._cooldown_ticks > 0:
            self._cooldown_ticks -= 1

        can_transition = (
            self._global_tick - self._last_transition_tick >= cfg.transition_min_ticks
        )

        kick_u = self._score_kick(progress, stamina_frac)
        pass_u = (
            self._score_pass(obs)
            if (self._cooldown_ticks <= 0 and can_transition)
            else -10
        )
        cruise_u = self._score_cruise(obs, stamina_frac)

        if kick_u >= cruise_u and kick_u >= pass_u and kick_u > 0:
            self._transition(self.STATE_KICK)
            return self._apply_defense(
                self._do_kick(obs, lateral_norm), obs, stamina_frac
            )

        if pass_u > cruise_u and pass_u > 0 and can_transition:
            self._transition(self.STATE_PASSING)
            return self._apply_defense(
                self._do_pass(stamina_frac), obs, stamina_frac
            )

        self._state_ticks += 1
        return self._apply_defense(
            self._do_cruise(speed_ratio, stamina_frac, lateral_norm),
            obs, stamina_frac,
        )

    # -------- State transitions --------

    def _transition(self, new_state: int) -> None:
        self._state = new_state
        self._state_ticks = 0
        self._last_transition_tick = self._global_tick

    # -------- Utility scoring --------

    def _score_cruise(self, obs: "np.ndarray", stamina_frac: float) -> float:
        score = 1.0
        if self._is_drafting(obs):
            score += (0.2 + (1.0 - stamina_frac) * 0.3) * self._cfg.w_draft
        return score

    def _score_pass(self, obs: "np.ndarray") -> float:
        cfg = self._cfg
        best = -10.0
        for s in range(OPPONENT_SLOTS):
            base = _OPP_BASE + s * OPPONENT_SLOT_SIZE
            if obs[base] < 0.5:
                continue
            progress_delta = obs[base + 1]
            tvel_delta = obs[base + 2]
            normal_offset = obs[base + 3]
            if not (0.0 < progress_delta < cfg.block_progress_max):
                continue
            if abs(normal_offset) > cfg.block_lateral_tol:
                continue
            if tvel_delta >= -cfg.block_min_slowness:
                continue
            severity = abs(tvel_delta)
            lateral_cost = abs(normal_offset)
            best = max(best, 0.3 + severity * 5.0 - lateral_cost * 2.0)
        return -10.0 if best < 0 else best * cfg.w_pass

    def _score_kick(self, progress: float, stamina_frac: float) -> float:
        cfg = self._cfg
        remaining = 1.0 - progress
        early_phase = cfg.kick_phase - cfg.kick_early_margin
        late_phase = min(cfg.kick_phase + cfg.kick_early_margin, cfg.kick_late_cap)
        if progress < early_phase:
            return -10.0
        if progress >= late_phase:
            return 10.0
        sustainability = stamina_frac - remaining * 1.5
        if sustainability <= 0:
            return -1.0
        return (0.5 + sustainability * 3.0) * cfg.w_kick

    # -------- Perception helpers --------

    def _is_drafting(self, obs: "np.ndarray") -> bool:
        for s in range(OPPONENT_SLOTS):
            base = _OPP_BASE + s * OPPONENT_SLOT_SIZE
            if obs[base] < 0.5:
                continue
            if (obs[base + 1] > 0.01 and obs[base + 1] < 0.05
                    and abs(obs[base + 3]) < 0.10
                    and obs[base + 2] >= -0.02):
                return True
        return False

    def _still_blocked(self, obs: "np.ndarray") -> bool:
        cfg = self._cfg
        for s in range(OPPONENT_SLOTS):
            base = _OPP_BASE + s * OPPONENT_SLOT_SIZE
            if obs[base] < 0.5:
                continue
            progress_delta = obs[base + 1]
            normal_offset = obs[base + 3]
            if (-0.01 < progress_delta < cfg.block_progress_max
                    and normal_offset < -cfg.pass_clear_lateral):
                return True
        return False

    def _is_blocked_during_kick(self, obs: "np.ndarray") -> bool:
        cfg = self._cfg
        for s in range(OPPONENT_SLOTS):
            base = _OPP_BASE + s * OPPONENT_SLOT_SIZE
            if obs[base] < 0.5:
                continue
            if (obs[base + 1] > 0 and obs[base + 1] < cfg.block_progress_max
                    and abs(obs[base + 3]) < cfg.block_lateral_tol
                    and obs[base + 2] < -cfg.block_min_slowness):
                return True
        return False

    def _compute_threat_score(self, obs: "np.ndarray") -> float:
        max_threat = 0.0
        for s in range(OPPONENT_SLOTS):
            base = _OPP_BASE + s * OPPONENT_SLOT_SIZE
            if obs[base] < 0.5:
                continue
            pd = obs[base + 1]
            tv = obs[base + 2]
            no = obs[base + 3]
            if -0.03 <= pd < 0 and tv > 0.03 and no > 0.05:
                threat = tv * 5.0 + (1.0 - abs(pd) / 0.03) * 0.3
                max_threat = max(max_threat, threat)
        return max_threat

    # -------- Defensive overlay --------

    def _apply_defense(
        self,
        inp: "InputState",
        obs: "np.ndarray",
        stamina_frac: float,
    ) -> "InputState":
        from ..core.types import InputState

        cfg = self._cfg
        threat = self._compute_threat_score(obs)
        if not self._defending and threat > cfg.defend_on_score:
            self._defending = True
        elif self._defending and threat < cfg.defend_off_score:
            self._defending = False
        if not self._defending or stamina_frac < 0.30:
            return inp
        return InputState(
            max(inp.tangential, cfg.defend_tang_min),
            inp.normal + cfg.defend_drift,
        )

    # -------- Shared steering / speed helpers --------

    def _steer_to_lane(self, lateral_norm: float, target_lane: float) -> float:
        err = lateral_norm - target_lane
        if abs(err) < 0.05:
            return 0.0
        return (
            -0.5 * self._cfg.lateral_aggression
            if err > 0
            else 0.5 * self._cfg.lateral_aggression
        )

    def _cruise_speed(self, speed_ratio: float, stamina_frac: float) -> float:
        cfg = self._cfg
        if speed_ratio < cfg.cruise_low - 0.05:
            tang = 0.5
        elif speed_ratio > cfg.cruise_high + 0.05:
            tang = 0.0
        else:
            tang = 0.25
        if stamina_frac < cfg.conserve_threshold:
            tang = min(tang, 0.25)
        return tang

    def _rate_for_lane_convergence(
        self, tang: float, lateral_norm: float, target_lane: float
    ) -> float:
        """Subtract tangential when far from lane so the field does not stay abreast."""
        cfg = self._cfg
        err = abs(lateral_norm - target_lane)
        if err <= cfg.off_lane_penalty_start:
            return tang
        excess = err - cfg.off_lane_penalty_start
        raw = excess * cfg.off_lane_tang_penalty_scale * cfg.off_lane_decel_scale
        penalty = min(cfg.off_lane_tang_penalty_max, raw)
        out = tang - penalty + cfg.off_lane_accel_relief
        return max(0.0, min(tang, out))

    # -------- State actions --------

    def _do_cruise(
        self, speed_ratio: float, stamina_frac: float, lateral_norm: float
    ) -> "InputState":
        from ..core.types import InputState
        cfg = self._cfg
        tang = self._cruise_speed(speed_ratio, stamina_frac)
        tang = self._rate_for_lane_convergence(tang, lateral_norm, cfg.target_lane)
        return InputState(
            tang,
            self._steer_to_lane(lateral_norm, cfg.target_lane),
        )

    def _do_pass(self, stamina_frac: float) -> "InputState":
        from ..core.types import InputState
        tang = 0.75 if stamina_frac > self._cfg.conserve_threshold else 0.5
        return InputState(tang, 0.5)

    def _do_kick(self, obs: "np.ndarray", lateral_norm: float) -> "InputState":
        from ..core.types import InputState
        if self._is_blocked_during_kick(obs):
            return InputState(1.0, 0.5)
        normal = -0.5 if lateral_norm > -0.80 else -0.25
        return InputState(1.0, normal)

    def _do_settle(
        self, speed_ratio: float, stamina_frac: float, lateral_norm: float
    ) -> "InputState":
        from ..core.types import InputState
        cfg = self._cfg
        t = min(self._state_ticks / cfg.settle_ticks, 1.0)
        target = self._settle_from_lane + (cfg.target_lane - self._settle_from_lane) * t
        tang = self._cruise_speed(speed_ratio, stamina_frac)
        tang = self._rate_for_lane_convergence(tang, lateral_norm, target)
        return InputState(
            tang,
            self._steer_to_lane(lateral_norm, target),
        )
