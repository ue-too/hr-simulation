"""Tests for v2 core modules: horse, stamina, physics, observation, reward."""

import math
import random

import numpy as np
import pytest

from horse_racing.core.horse import HorseProfile, default_horse, random_horse, TRAIT_RANGES
from horse_racing.core.stamina import (
    BASE_DRAIN_RATE,
    CRITICAL_THRESHOLD,
    EXCESS_DRAIN_RATE,
    FATIGUE_THRESHOLD,
    StaminaState,
    apply_fatigue,
    compute_drain,
    create_stamina,
)
from horse_racing.core.physics import (
    compute_forward_force,
    compute_lateral_force,
    compute_target_speed,
    fatigue_wobble,
    smooth_force,
    stride_oscillation,
)
from horse_racing.core.types import HorseBody, JockeyAction, JockeyStyle, TrackFrame
from horse_racing.env.observation import OBS_SIZE, build_observation
from horse_racing.env.reward import FINISH_ORDER_BONUS, compute_reward


# ---------------------------------------------------------------------------
# Horse profile
# ---------------------------------------------------------------------------


class TestHorseProfile:
    def test_default_horse_traits_at_midpoint(self):
        h = default_horse()
        for name, (lo, hi) in TRAIT_RANGES.items():
            val = getattr(h, name)
            expected = (lo + hi) / 2
            assert abs(val - expected) < 1e-6, f"{name}: {val} != {expected}"

    def test_efficiency_speed_is_75_percent(self):
        h = HorseProfile(
            top_speed=20.0, acceleration=1.0, stamina_pool=100,
            stamina_efficiency=1.0, cornering_grip_left=1.0,
            cornering_grip_right=1.0, weight=490, climbing_power=1.0,
        )
        assert abs(h.efficiency_speed - 15.0) < 1e-6

    def test_random_horse_within_ranges(self):
        rng = random.Random(42)
        for _ in range(50):
            h = random_horse(rng)
            for name, (lo, hi) in TRAIT_RANGES.items():
                val = getattr(h, name)
                assert lo <= val <= hi, f"{name}={val} out of [{lo}, {hi}]"

    def test_random_horse_has_variance(self):
        rng = random.Random(123)
        horses = [random_horse(rng) for _ in range(20)]
        speeds = [h.top_speed for h in horses]
        assert max(speeds) - min(speeds) > 1.0, "Too little variance in random horses"


# ---------------------------------------------------------------------------
# Stamina
# ---------------------------------------------------------------------------


class TestStamina:
    def test_create_stamina_full(self):
        h = default_horse()
        s = create_stamina(h)
        assert s.current == h.stamina_pool
        assert s.maximum == h.stamina_pool
        assert abs(s.ratio - 1.0) < 1e-6

    def test_drain_reduces_current(self):
        s = StaminaState(current=100, maximum=100)
        s.drain(10)
        assert abs(s.current - 90) < 1e-6

    def test_drain_floors_at_zero(self):
        s = StaminaState(current=5, maximum=100)
        s.drain(20)
        assert s.current == 0.0

    def test_fatigue_threshold(self):
        s = StaminaState(current=29, maximum=100)  # 29% < 30% threshold
        assert s.is_fatigued
        s2 = StaminaState(current=31, maximum=100)
        assert not s2.is_fatigued

    def test_critical_threshold(self):
        s = StaminaState(current=14, maximum=100)  # 14% < 15% threshold
        assert s.is_critical
        s2 = StaminaState(current=16, maximum=100)
        assert not s2.is_critical

    def test_drain_at_efficiency_speed(self):
        """At efficiency speed, only base drain applies (no excess)."""
        h = default_horse()
        eff_speed = h.efficiency_speed
        drain = compute_drain(h, eff_speed, 0.0, 0.0, 1.0)
        expected = BASE_DRAIN_RATE * eff_speed / h.stamina_efficiency
        assert abs(drain - expected) < 1e-6

    def test_drain_above_efficiency_is_quadratic(self):
        """Drain increases quadratically above efficiency speed."""
        h = default_horse()
        eff = h.efficiency_speed
        drain_at_eff = compute_drain(h, eff, 0.0, 0.0, 1.0)
        drain_1_over = compute_drain(h, eff + 1.0, 0.0, 0.0, 1.0)
        drain_2_over = compute_drain(h, eff + 2.0, 0.0, 0.0, 1.0)
        # Quadratic: excess of 2 should cost ~4x the excess of 1
        excess_1 = drain_1_over - drain_at_eff
        excess_2 = drain_2_over - drain_at_eff
        # Not exactly 4x because base drain also increases linearly
        assert excess_2 > excess_1 * 3.0, "Drain should scale super-linearly"

    def test_apply_fatigue_no_effect_above_threshold(self):
        s = StaminaState(current=50, maximum=100)  # 50% > 30%
        fwd, lat = apply_fatigue(s, 100.0, 50.0)
        assert fwd == 100.0
        assert lat == 50.0

    def test_apply_fatigue_reduces_below_threshold(self):
        s = StaminaState(current=20, maximum=100)  # 20% < 30%
        fwd, lat = apply_fatigue(s, 100.0, 50.0)
        assert fwd < 100.0
        assert lat < 50.0

    def test_apply_fatigue_severe_at_critical(self):
        s = StaminaState(current=5, maximum=100)  # 5% < 15%
        fwd, lat = apply_fatigue(s, 100.0, 50.0)
        assert fwd < 40.0, "Should be severely reduced below critical"


# ---------------------------------------------------------------------------
# Physics — action-to-force
# ---------------------------------------------------------------------------


def _make_body() -> HorseBody:
    return HorseBody(
        position=np.array([0.0, 0.0]),
        velocity=np.array([10.0, 0.0]),
        orientation=0.0,
        smoothed_forward_force=0.0,
        smoothed_lateral_force=0.0,
    )


def _make_frame(turn_direction: int = 0) -> TrackFrame:
    return TrackFrame(
        tangential=np.array([1.0, 0.0]),
        normal=np.array([0.0, 1.0]),
        turn_radius=1e7 if turn_direction == 0 else 100.0,
        target_radius=100.0,
        segment_index=0,
        slope=0.0,
        turn_direction=float(turn_direction),
    )


class TestPhysics:
    def test_target_speed_at_zero_effort(self):
        h = default_horse()
        speed = compute_target_speed(JockeyAction(effort=0.0, lane=0.0), h)
        assert abs(speed - h.efficiency_speed) < 1e-6

    def test_target_speed_at_max_effort(self):
        h = default_horse()
        speed = compute_target_speed(JockeyAction(effort=1.0, lane=0.0), h)
        assert abs(speed - h.top_speed) < 1e-6

    def test_target_speed_at_min_effort(self):
        h = default_horse()
        speed = compute_target_speed(JockeyAction(effort=-1.0, lane=0.0), h)
        expected = h.efficiency_speed * 0.5
        assert abs(speed - expected) < 1e-6

    def test_forward_force_positive_when_below_target(self):
        h = default_horse()
        force = compute_forward_force(10.0, 18.0, h)
        assert force > 0

    def test_forward_force_negative_when_above_target(self):
        h = default_horse()
        force = compute_forward_force(20.0, 14.0, h)
        assert force < 0

    def test_lateral_force_uses_left_grip_on_left_turn(self):
        h = HorseProfile(
            top_speed=18, acceleration=1.0, stamina_pool=100,
            stamina_efficiency=1.0, cornering_grip_left=1.5,
            cornering_grip_right=0.5, weight=490, climbing_power=1.0,
        )
        frame_left = _make_frame(turn_direction=1)  # left turn
        frame_right = _make_frame(turn_direction=-1)  # right turn
        action = JockeyAction(effort=0, lane=1.0)

        force_left = compute_lateral_force(action, h, frame_left)
        force_right = compute_lateral_force(action, h, frame_right)
        assert abs(force_left) > abs(force_right), "Left grip should produce more force on left turn"

    def test_stride_oscillation_range(self):
        for t in np.linspace(0, 2, 100):
            val = stride_oscillation(t)
            assert abs(val) <= 0.2 + 1e-6

    def test_fatigue_wobble_zero_when_not_fatigued(self):
        s = StaminaState(current=50, maximum=100)
        assert fatigue_wobble(s, 1.0) == 0.0

    def test_fatigue_wobble_nonzero_when_fatigued(self):
        s = StaminaState(current=10, maximum=100)
        # Wobble is sinusoidal, so check over a few time steps
        vals = [fatigue_wobble(s, t * 0.1) for t in range(20)]
        assert any(v != 0.0 for v in vals), "Should produce some wobble"

    def test_smooth_force_converges(self):
        h = default_horse()
        body = _make_body()
        # Apply same raw force repeatedly — smoothed should converge
        for _ in range(500):
            smooth_force(body, 1000.0, 500.0, h, 0.01)
        assert abs(body.smoothed_forward_force - 1000.0) < 1.0
        assert abs(body.smoothed_lateral_force - 500.0) < 1.0


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class TestObservation:
    def test_obs_size(self):
        assert OBS_SIZE == 63

    def test_build_observation_shape(self):
        """Single horse, basic observation should have correct shape."""
        body = _make_body()
        profile = default_horse()
        stamina = create_stamina(profile)
        frame = _make_frame()
        # Minimal navigator stub
        nav = _make_stub_navigator()
        style = JockeyStyle(risk_tolerance=0.5, tactical_bias=0.0, skill_level=0.5)

        obs = build_observation(
            horse_idx=0,
            bodies=[body],
            profiles=[profile],
            staminas=[stamina],
            frames=[frame],
            navigators=[nav],
            jockey_style=style,
            num_horses=1,
            placement=0,
            sim_time=10.0,
            total_race_time_est=60.0,
        )
        assert obs.shape == (63,)
        assert obs.dtype == np.float32

    def test_obs_stamina_ratio_is_correct(self):
        body = _make_body()
        profile = default_horse()
        stamina = StaminaState(current=50, maximum=100)
        frame = _make_frame()
        nav = _make_stub_navigator()
        style = JockeyStyle(risk_tolerance=0.5, tactical_bias=0.0, skill_level=0.5)

        obs = build_observation(
            horse_idx=0, bodies=[body], profiles=[profile],
            staminas=[stamina], frames=[frame], navigators=[nav],
            jockey_style=style, num_horses=1, placement=0,
            sim_time=0, total_race_time_est=60,
        )
        # Index 6 is stamina_ratio
        assert abs(obs[6] - 0.5) < 1e-6

    def test_obs_no_nans(self):
        body = _make_body()
        profile = default_horse()
        stamina = create_stamina(profile)
        frame = _make_frame()
        nav = _make_stub_navigator()
        style = JockeyStyle(risk_tolerance=0.5, tactical_bias=0.0, skill_level=0.5)

        obs = build_observation(
            horse_idx=0, bodies=[body], profiles=[profile],
            staminas=[stamina], frames=[frame], navigators=[nav],
            jockey_style=style, num_horses=1, placement=0,
            sim_time=0, total_race_time_est=60,
        )
        assert not np.any(np.isnan(obs)), "Observation contains NaN values"


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


class TestReward:
    def test_finish_bonus_first_place(self):
        style = JockeyStyle(risk_tolerance=0.5, tactical_bias=0.0, skill_level=0.5)
        stamina = StaminaState(current=50, maximum=100)
        r = compute_reward(
            placement=0, num_horses=10, progress=1.0, prev_progress=0.99,
            collided=False, finished=True, jockey_style=style, stamina=stamina,
        )
        assert r >= FINISH_ORDER_BONUS[0] - 1.0  # minus alive penalty + other components

    def test_collision_penalty(self):
        style = JockeyStyle(risk_tolerance=0.5, tactical_bias=0.0, skill_level=0.5)
        stamina = StaminaState(current=50, maximum=100)
        r_clean = compute_reward(
            placement=5, num_horses=10, progress=0.5, prev_progress=0.49,
            collided=False, finished=False, jockey_style=style, stamina=stamina,
        )
        r_collide = compute_reward(
            placement=5, num_horses=10, progress=0.5, prev_progress=0.49,
            collided=True, finished=False, jockey_style=style, stamina=stamina,
        )
        assert r_collide < r_clean
        assert abs(r_clean - r_collide - 1.0) < 1e-6  # exactly -1.0 penalty

    def test_progress_nudge_positive(self):
        style = JockeyStyle(risk_tolerance=0.5, tactical_bias=0.0, skill_level=0.5)
        stamina = StaminaState(current=50, maximum=100)
        r_forward = compute_reward(
            placement=0, num_horses=1, progress=0.5, prev_progress=0.49,
            collided=False, finished=False, jockey_style=style, stamina=stamina,
        )
        r_stall = compute_reward(
            placement=0, num_horses=1, progress=0.5, prev_progress=0.5,
            collided=False, finished=False, jockey_style=style, stamina=stamina,
        )
        assert r_forward > r_stall

    def test_front_runner_nudge(self):
        style = JockeyStyle(risk_tolerance=0.5, tactical_bias=-0.8, skill_level=0.5)
        stamina = StaminaState(current=50, maximum=100)
        r_leading = compute_reward(
            placement=0, num_horses=10, progress=0.3, prev_progress=0.29,
            collided=False, finished=False, jockey_style=style, stamina=stamina,
            leading_at_25=True,
        )
        r_not_leading = compute_reward(
            placement=0, num_horses=10, progress=0.3, prev_progress=0.29,
            collided=False, finished=False, jockey_style=style, stamina=stamina,
            leading_at_25=False,
        )
        assert r_leading > r_not_leading

    def test_closer_nudge(self):
        style = JockeyStyle(risk_tolerance=0.5, tactical_bias=0.8, skill_level=0.5)
        stamina = StaminaState(current=50, maximum=100)
        r_gained = compute_reward(
            placement=2, num_horses=10, progress=0.9, prev_progress=0.89,
            collided=False, finished=False, jockey_style=style, stamina=stamina,
            positions_gained_final_25=3,
        )
        r_no_gain = compute_reward(
            placement=2, num_horses=10, progress=0.9, prev_progress=0.89,
            collided=False, finished=False, jockey_style=style, stamina=stamina,
            positions_gained_final_25=0,
        )
        assert r_gained > r_no_gain

    def test_conservative_stamina_penalty(self):
        style = JockeyStyle(risk_tolerance=0.1, tactical_bias=0.0, skill_level=0.5)
        stamina_low = StaminaState(current=15, maximum=100)
        stamina_ok = StaminaState(current=50, maximum=100)
        r_low = compute_reward(
            placement=5, num_horses=10, progress=0.5, prev_progress=0.49,
            collided=False, finished=False, jockey_style=style, stamina=stamina_low,
        )
        r_ok = compute_reward(
            placement=5, num_horses=10, progress=0.5, prev_progress=0.49,
            collided=False, finished=False, jockey_style=style, stamina=stamina_ok,
        )
        assert r_low < r_ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubNavigator:
    """Minimal navigator for observation tests."""

    total_length = 1000.0

    def compute_progress(self, position):
        return 0.5

    def lookahead_segments(self, count):
        from horse_racing.core.types import StraightSegment
        return [
            StraightSegment(
                tracktype="STRAIGHT",
                start_point=(0.0, 0.0),
                end_point=(100.0, 0.0),
                slope=0.0,
            )
            for _ in range(count)
        ]


def _make_stub_navigator():
    return _StubNavigator()
