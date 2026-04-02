"""Tests for the stamina system — fixed pool, no recovery, lateral drains."""

import pytest

from horse_racing.attributes import CoreAttributes
from horse_racing.stamina import HorseRuntimeState, apply_exhaustion, update_stamina


def _make_state(stamina: float = 100.0) -> HorseRuntimeState:
    attrs = CoreAttributes(stamina=stamina)
    return HorseRuntimeState(current_stamina=stamina, base_attributes=attrs)


def _default_eff(**overrides) -> CoreAttributes:
    return CoreAttributes(**overrides)


# ---------------------------------------------------------------------------
# update_stamina: no recovery
# ---------------------------------------------------------------------------


class TestNoRecovery:
    def test_stamina_never_increases(self):
        """With no actions, stamina should only decrease (speed drain)."""
        state = _make_state(100.0)
        eff = _default_eff()
        for _ in range(100):
            prev = state.current_stamina
            update_stamina(state, eff, 0.0, 0.0, 14.0, 14.0, 0.0, float("inf"))
            assert state.current_stamina <= prev

    def test_zero_speed_no_drain(self):
        """At zero speed and no actions, stamina stays the same."""
        state = _make_state(100.0)
        eff = _default_eff()
        update_stamina(state, eff, 0.0, 0.0, 0.0, 0.0, 0.0, float("inf"))
        assert state.current_stamina == 100.0

    def test_stamina_floors_at_zero(self):
        """Stamina cannot go below zero."""
        state = _make_state(0.001)
        eff = _default_eff()
        update_stamina(state, eff, 10.0, 5.0, 20.0, 20.0, 3.0, 50.0)
        assert state.current_stamina == 0.0


# ---------------------------------------------------------------------------
# update_stamina: forward drains
# ---------------------------------------------------------------------------


class TestForwardDrains:
    def test_tangential_push_drains(self):
        """Pushing forward should drain more than coasting."""
        coast = _make_state(100.0)
        push = _make_state(100.0)
        eff = _default_eff()
        update_stamina(coast, eff, 0.0, 0.0, 14.0, 14.0, 0.0, float("inf"))
        update_stamina(push, eff, 5.0, 0.0, 14.0, 14.0, 0.0, float("inf"))
        assert push.current_stamina < coast.current_stamina

    def test_overdrive_drains(self):
        """Exceeding cruise speed should drain extra."""
        cruise = _make_state(100.0)
        over = _make_state(100.0)
        eff = _default_eff()  # cruise_speed=14.25
        update_stamina(cruise, eff, 0.0, 0.0, 14.0, 14.0, 0.0, float("inf"))
        update_stamina(over, eff, 0.0, 0.0, 18.0, 18.0, 0.0, float("inf"))
        assert over.current_stamina < cruise.current_stamina

    def test_speed_drain_proportional(self):
        """Faster speed = more drain (distance tax)."""
        slow = _make_state(100.0)
        fast = _make_state(100.0)
        eff = _default_eff()
        update_stamina(slow, eff, 0.0, 0.0, 10.0, 10.0, 0.0, float("inf"))
        update_stamina(fast, eff, 0.0, 0.0, 18.0, 18.0, 0.0, float("inf"))
        assert fast.current_stamina < slow.current_stamina


# ---------------------------------------------------------------------------
# update_stamina: lateral drains
# ---------------------------------------------------------------------------


class TestLateralDrains:
    def test_lateral_steering_drains(self):
        """Steering laterally should cost stamina."""
        no_steer = _make_state(100.0)
        steer = _make_state(100.0)
        eff = _default_eff()
        update_stamina(no_steer, eff, 0.0, 0.0, 14.0, 14.0, 0.0, float("inf"))
        update_stamina(steer, eff, 0.0, 3.0, 14.0, 14.0, 0.0, float("inf"))
        assert steer.current_stamina < no_steer.current_stamina

    def test_lateral_velocity_drains(self):
        """Sustained lateral drift should cost stamina."""
        no_drift = _make_state(100.0)
        drift = _make_state(100.0)
        eff = _default_eff()
        update_stamina(no_drift, eff, 0.0, 0.0, 14.0, 14.0, 0.0, float("inf"))
        update_stamina(drift, eff, 0.0, 0.0, 14.0, 14.0, 2.0, float("inf"))
        assert drift.current_stamina < no_drift.current_stamina

    def test_negative_steering_also_drains(self):
        """Steering in either direction should drain equally."""
        left = _make_state(100.0)
        right = _make_state(100.0)
        eff = _default_eff()
        update_stamina(left, eff, 0.0, -3.0, 14.0, 14.0, 0.0, float("inf"))
        update_stamina(right, eff, 0.0, 3.0, 14.0, 14.0, 0.0, float("inf"))
        assert left.current_stamina == pytest.approx(right.current_stamina)


# ---------------------------------------------------------------------------
# update_stamina: drain_rate_mult
# ---------------------------------------------------------------------------


class TestDrainRateMult:
    def test_lower_mult_means_less_drain(self):
        """A horse with drain_rate_mult < 1 should drain less."""
        normal = _make_state(100.0)
        efficient = _make_state(100.0)
        eff_normal = _default_eff(drain_rate_mult=1.0)
        eff_efficient = _default_eff(drain_rate_mult=0.7)
        update_stamina(normal, eff_normal, 5.0, 2.0, 16.0, 16.0, 1.0, float("inf"))
        update_stamina(efficient, eff_efficient, 5.0, 2.0, 16.0, 16.0, 1.0, float("inf"))
        assert efficient.current_stamina > normal.current_stamina

    def test_higher_mult_means_more_drain(self):
        """A horse with drain_rate_mult > 1 should drain more."""
        normal = _make_state(100.0)
        wasteful = _make_state(100.0)
        eff_normal = _default_eff(drain_rate_mult=1.0)
        eff_wasteful = _default_eff(drain_rate_mult=1.3)
        update_stamina(normal, eff_normal, 5.0, 0.0, 16.0, 16.0, 0.0, float("inf"))
        update_stamina(wasteful, eff_wasteful, 5.0, 0.0, 16.0, 16.0, 0.0, float("inf"))
        assert wasteful.current_stamina < normal.current_stamina


# ---------------------------------------------------------------------------
# apply_exhaustion: progressive turn_accel
# ---------------------------------------------------------------------------


class TestExhaustion:
    def test_no_effect_above_thresholds(self):
        """Above 30% stamina, no degradation."""
        eff = _default_eff()
        result = apply_exhaustion(eff, 50.0, 100.0)
        assert result.forward_accel == eff.forward_accel
        assert result.max_speed == eff.max_speed
        assert result.turn_accel == eff.turn_accel

    def test_forward_accel_degrades_below_30(self):
        """forward_accel scales linearly below 30%."""
        eff = _default_eff()
        result = apply_exhaustion(eff, 15.0, 100.0)
        assert result.forward_accel == pytest.approx(eff.forward_accel * 0.5)

    def test_max_speed_degrades_below_20(self):
        """max_speed lerps toward cruise below 20%."""
        eff = _default_eff()
        result = apply_exhaustion(eff, 10.0, 100.0)
        expected = eff.cruise_speed + (eff.max_speed - eff.cruise_speed) * 0.5
        assert result.max_speed == pytest.approx(expected)

    def test_turn_accel_progressive_from_25(self):
        """turn_accel degrades progressively starting at 25%."""
        eff = _default_eff()
        # At 25% boundary: should be ~100%
        at_25 = apply_exhaustion(eff, 25.0, 100.0)
        assert at_25.turn_accel == pytest.approx(eff.turn_accel)

        # At 12.5%: should be ~75%
        at_12 = apply_exhaustion(eff, 12.5, 100.0)
        assert at_12.turn_accel == pytest.approx(eff.turn_accel * 0.75)

        # At 0%: should be ~50%
        at_0 = apply_exhaustion(eff, 0.0, 100.0)
        assert at_0.turn_accel == pytest.approx(eff.turn_accel * 0.5)

    def test_turn_accel_monotonically_decreasing(self):
        """Lower stamina should always mean lower turn_accel (below threshold)."""
        eff = _default_eff()
        prev_ta = eff.turn_accel
        for pct in [24, 20, 15, 10, 5, 1]:
            result = apply_exhaustion(eff, float(pct), 100.0)
            assert result.turn_accel <= prev_ta
            prev_ta = result.turn_accel
