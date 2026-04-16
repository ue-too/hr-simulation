"""Verify behavior-tree config knobs affect outputs as intended.

Tests call BehaviorTreeStrategy helpers / scoring with controlled BTConfig
and synthetic observation vectors. See horse_racing/opponents/JOCKEY.md for
parameter meanings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from horse_racing.core.observation import OBS_SIZE, OPPONENT_SLOT_SIZE, SELF_STATE_SIZE, TRACK_CONTEXT_SIZE
from horse_racing.core.race import Race
from horse_racing.core.track import load_track_json
from horse_racing.opponents.behavior_tree import (
    BTConfig,
    BehaviorTreeStrategy,
)

TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"
_OPP0 = SELF_STATE_SIZE + TRACK_CONTEXT_SIZE  # first opponent slot base index


def _load_segments():
    return load_track_json(TRACKS_DIR / "test_oval.json")


def _strategy(cfg: BTConfig) -> BehaviorTreeStrategy:
    race = Race(_load_segments(), horse_count=2)
    race.start(None)
    return BehaviorTreeStrategy(race, 0, cfg)


def _zeros_obs() -> np.ndarray:
    return np.zeros(OBS_SIZE, dtype=np.float64)


def _filled_opponent_slot(
    obs: np.ndarray,
    slot: int,
    *,
    active: float,
    progress_delta: float,
    tvel_delta: float,
    norm_offset: float,
    nvel_delta: float = 0.0,
) -> None:
    base = _OPP0 + slot * OPPONENT_SLOT_SIZE
    obs[base + 0] = active
    obs[base + 1] = progress_delta
    obs[base + 2] = tvel_delta
    obs[base + 3] = norm_offset
    obs[base + 4] = nvel_delta


class TestCruiseSpeedBand:
    """cruise_low / cruise_high + conserve_threshold."""

    def test_below_band_pushes(self):
        s = _strategy(BTConfig(cruise_low=0.55, cruise_high=0.70, conserve_threshold=0.0))
        assert s._cruise_speed(0.48, 0.50) == pytest.approx(0.5)

    def test_in_band_holds(self):
        s = _strategy(BTConfig(cruise_low=0.55, cruise_high=0.70))
        assert s._cruise_speed(0.62, 0.50) == pytest.approx(0.25)

    def test_above_band_coasts(self):
        s = _strategy(BTConfig(cruise_low=0.55, cruise_high=0.70))
        assert s._cruise_speed(0.80, 0.50) == pytest.approx(0.0)

    def test_conserve_caps_tangential(self):
        s = _strategy(BTConfig(conserve_threshold=0.50))
        assert s._cruise_speed(0.48, 0.40) == pytest.approx(0.25)  # would be 0.5 push otherwise


class TestSteerToLane:
    """target_lane + lateral_aggression."""

    def test_pulls_left_when_too_wide(self):
        s = _strategy(BTConfig(target_lane=-0.80, lateral_aggression=0.6))
        # lateral > target -> err > 0 -> steer negative (inside)
        n = s._steer_to_lane(-0.40, -0.80)
        assert n == pytest.approx(-0.5 * 0.6)

    def test_pulls_right_when_too_inside(self):
        s = _strategy(BTConfig(target_lane=-0.30, lateral_aggression=0.8))
        n = s._steer_to_lane(-0.80, -0.30)
        assert n == pytest.approx(0.5 * 0.8)

    def test_dead_zone_no_steer(self):
        s = _strategy(BTConfig(target_lane=-0.60))
        assert s._steer_to_lane(-0.63, -0.60) == pytest.approx(0.0)


class TestLaneConvergenceKnobs:
    """off_lane_penalty_*, off_lane_decel_scale, off_lane_accel_relief."""

    def _expected_out(self, cfg: BTConfig, tang: float, lateral: float, target: float) -> float:
        err = abs(lateral - target)
        if err <= cfg.off_lane_penalty_start:
            return tang
        excess = err - cfg.off_lane_penalty_start
        raw = excess * cfg.off_lane_tang_penalty_scale * cfg.off_lane_decel_scale
        penalty = min(cfg.off_lane_tang_penalty_max, raw)
        out = tang - penalty + cfg.off_lane_accel_relief
        return max(0.0, min(tang, out))

    def test_matches_closed_form(self):
        cfg = BTConfig(
            off_lane_penalty_start=0.06,
            off_lane_tang_penalty_scale=0.5,
            off_lane_tang_penalty_max=0.18,
            off_lane_decel_scale=1.0,
            off_lane_accel_relief=0.0,
            target_lane=-0.80,
        )
        s = _strategy(cfg)
        lateral = 0.0
        target = -0.80
        tang = 0.25
        got = s._rate_for_lane_convergence(tang, lateral, target)
        assert got == pytest.approx(self._expected_out(cfg, tang, lateral, target))

    def test_decel_scale_increases_penalty(self):
        # Use lateral error where scale=1.0 is below max cap, so scale=2.0 increases penalty.
        base = BTConfig(
            off_lane_penalty_start=0.06,
            off_lane_tang_penalty_scale=0.5,
            off_lane_tang_penalty_max=0.18,
            off_lane_decel_scale=1.0,
            off_lane_accel_relief=0.0,
        )
        strong = BTConfig(**{**base.__dict__, "off_lane_decel_scale": 2.0})
        s1 = _strategy(base)
        s2 = _strategy(strong)
        tang, lat, tgt = 0.25, -0.50, -0.80
        assert s2._rate_for_lane_convergence(tang, lat, tgt) < s1._rate_for_lane_convergence(
            tang, lat, tgt
        )

    def test_accel_relief_increases_tangential(self):
        cfg = BTConfig(
            off_lane_penalty_start=0.06,
            off_lane_tang_penalty_scale=0.5,
            off_lane_tang_penalty_max=0.18,
            off_lane_decel_scale=1.0,
            off_lane_accel_relief=0.0,
        )
        relief = BTConfig(**{**cfg.__dict__, "off_lane_accel_relief": 0.10})
        s1 = _strategy(cfg)
        s2 = _strategy(relief)
        tang, lat, tgt = 0.25, 0.0, -0.80
        assert s2._rate_for_lane_convergence(tang, lat, tgt) > s1._rate_for_lane_convergence(
            tang, lat, tgt
        )

    def test_no_effect_inside_deadband(self):
        cfg = BTConfig(off_lane_penalty_start=0.06, target_lane=-0.80)
        s = _strategy(cfg)
        # 0.05 error: inside start threshold? err=0.05 <= 0.06 -> no penalty
        tang = 0.25
        assert s._rate_for_lane_convergence(tang, -0.75, -0.80) == pytest.approx(tang)


class TestUtilityScores:
    """w_draft, w_pass, w_kick."""

    def test_w_draft_boosts_cruise_when_drafting(self):
        low = _strategy(BTConfig(w_draft=0.5))
        high = _strategy(BTConfig(w_draft=2.0))
        obs = _zeros_obs()
        _filled_opponent_slot(
            obs, 0,
            active=1.0,
            progress_delta=0.03,
            tvel_delta=0.0,
            norm_offset=0.05,
        )
        stamina = 0.5
        assert low._score_cruise(obs, stamina) < high._score_cruise(obs, stamina)

    def test_w_pass_scales_pass_score(self):
        cfg = BTConfig(
            block_progress_max=0.05,
            block_lateral_tol=0.20,
            block_min_slowness=0.02,
            w_pass=0.5,
        )
        s_low = _strategy(cfg)
        s_high = _strategy(BTConfig(**{**cfg.__dict__, "w_pass": 2.0}))
        obs = _zeros_obs()
        _filled_opponent_slot(
            obs, 0,
            active=1.0,
            progress_delta=0.02,
            tvel_delta=-0.05,
            norm_offset=0.05,
        )
        assert s_high._score_pass(obs) > s_low._score_pass(obs)

    def test_w_kick_scales_middle_window(self):
        cfg = BTConfig(kick_phase=0.75, kick_early_margin=0.10, kick_late_cap=0.92, w_kick=0.5)
        s_low = _strategy(cfg)
        s_high = _strategy(BTConfig(**{**cfg.__dict__, "w_kick": 2.0}))
        progress = 0.80
        stamina = 0.80
        assert s_high._score_kick(progress, stamina) > s_low._score_kick(progress, stamina)

    def test_kick_forced_late_regardless_of_w_kick(self):
        cfg = BTConfig(kick_phase=0.75, kick_early_margin=0.10, kick_late_cap=0.92, w_kick=0.1)
        s = _strategy(cfg)
        assert s._score_kick(0.93, 0.05) == pytest.approx(10.0)

    def test_kick_early_window_returns_negative(self):
        s = _strategy(BTConfig(kick_phase=0.75, kick_early_margin=0.10))
        assert s._score_kick(0.60, 1.0) == pytest.approx(-10.0)


class TestBlockAndDefenseKnobs:
    """block_*, pass_clear_lateral, defend_*."""

    def test_block_min_slowness_requires_slower_opponent(self):
        loose = _strategy(BTConfig(block_min_slowness=0.001))
        strict = _strategy(BTConfig(block_min_slowness=0.10))
        obs = _zeros_obs()
        _filled_opponent_slot(
            obs, 0,
            active=1.0,
            progress_delta=0.02,
            tvel_delta=-0.02,
            norm_offset=0.05,
        )
        assert strict._score_pass(obs) == pytest.approx(-10.0)
        assert loose._score_pass(obs) > 0

    def test_pass_clear_lateral_affects_still_blocked(self):
        cfg = BTConfig(pass_clear_lateral=0.25, block_progress_max=0.05)
        s = _strategy(cfg)
        obs = _zeros_obs()
        _filled_opponent_slot(
            obs, 0,
            active=1.0,
            progress_delta=0.02,
            tvel_delta=0.0,
            norm_offset=-0.30,
        )
        assert s._still_blocked(obs) is True

    def test_defend_on_score_triggers_overlay(self):
        cfg = BTConfig(defend_on_score=0.3, defend_off_score=0.1, defend_tang_min=0.5, defend_drift=0.1)
        s = _strategy(cfg)
        obs = _zeros_obs()
        _filled_opponent_slot(
            obs, 0,
            active=1.0,
            progress_delta=-0.01,
            tvel_delta=0.10,
            norm_offset=0.10,
        )
        from horse_racing.core.types import InputState

        inp = InputState(0.25, -0.25)
        out = s._apply_defense(inp, obs, 0.80)
        assert out.tangential >= 0.5
        assert out.normal > inp.normal


class TestKickBlockingVsLane:
    """Kick lane vs wide when blocked."""

    def test_kick_goes_wide_when_blocked(self):
        cfg = BTConfig(block_progress_max=0.05, block_lateral_tol=0.20, block_min_slowness=0.02)
        s = _strategy(cfg)
        obs = _zeros_obs()
        _filled_opponent_slot(
            obs, 0,
            active=1.0,
            progress_delta=0.02,
            tvel_delta=-0.10,
            norm_offset=0.05,
        )
        k = s._do_kick(obs, lateral_norm=-0.50)
        assert k.tangential == pytest.approx(1.0)
        assert k.normal == pytest.approx(0.5)


class TestDoCruiseIntegration:
    """do_cruise combines cruise_speed + rate + steer."""

    def test_high_lateral_error_reduces_tang_vs_on_target(self):
        cfg = BTConfig(
            cruise_low=0.55,
            cruise_high=0.70,
            target_lane=-0.80,
            off_lane_penalty_start=0.06,
            off_lane_tang_penalty_scale=0.5,
            off_lane_tang_penalty_max=0.18,
            off_lane_decel_scale=1.0,
            off_lane_accel_relief=0.0,
        )
        s = _strategy(cfg)
        on = s._do_cruise(0.62, 0.80, lateral_norm=-0.80)
        far = s._do_cruise(0.62, 0.80, lateral_norm=0.0)
        assert far.tangential < on.tangential

