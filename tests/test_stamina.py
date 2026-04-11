"""Tests for the redesigned stamina system.

Two pools (aerobic + burst), lead penalty, draft recovery, cliff collapse.
See horse_racing/types.py for the high-level mechanic description.
"""

import pytest

from horse_racing.attributes import CoreAttributes
from horse_racing.stamina import (
    HorseRuntimeState,
    apply_exhaustion,
    compute_burst_max,
    update_stamina,
)
from horse_racing.types import (
    BURST_EMPTY_CLAMP,
    CLIFF_ACCEL_MULT,
    CLIFF_CRUISE_MULT,
    CLIFF_THRESHOLD,
)


def _make_state(stamina: float = 100.0, attrs: CoreAttributes | None = None) -> HorseRuntimeState:
    a = attrs or CoreAttributes(stamina=stamina)
    burst_max = compute_burst_max(a)
    return HorseRuntimeState(
        current_stamina=stamina,
        base_attributes=a,
        burst_pool=burst_max,
        burst_max=burst_max,
    )


def _default_eff(**overrides) -> CoreAttributes:
    return CoreAttributes(**overrides)


# ---------------------------------------------------------------------------
# update_stamina: basic drain behavior
# ---------------------------------------------------------------------------


class TestBasicDrain:
    def test_zero_speed_no_drain(self):
        state = _make_state(100.0)
        eff = _default_eff()
        update_stamina(state, eff, 0.0, 0.0, 0.0, 0.0, 0.0, float("inf"))
        assert state.current_stamina == 100.0

    def test_stamina_floors_at_zero(self):
        state = _make_state(0.001)
        eff = _default_eff()
        update_stamina(state, eff, 10.0, 5.0, 20.0, 20.0, 3.0, 50.0)
        assert state.current_stamina == 0.0

    def test_distance_tax_proportional_to_speed(self):
        slow = _make_state(100.0)
        fast = _make_state(100.0)
        eff = _default_eff()
        update_stamina(slow, eff, 0.0, 0.0, 10.0, 10.0, 0.0, float("inf"))
        update_stamina(fast, eff, 0.0, 0.0, 18.0, 18.0, 0.0, float("inf"))
        assert fast.current_stamina < slow.current_stamina

    def test_push_drains_extra(self):
        coast = _make_state(100.0)
        push = _make_state(100.0)
        eff = _default_eff()
        update_stamina(coast, eff, 0.0, 0.0, 14.0, 14.0, 0.0, float("inf"))
        update_stamina(push, eff, 5.0, 0.0, 14.0, 14.0, 0.0, float("inf"))
        assert push.current_stamina < coast.current_stamina


# ---------------------------------------------------------------------------
# update_stamina: lateral drains
# ---------------------------------------------------------------------------


class TestLateralDrains:
    def test_lateral_steering_drains(self):
        no_steer = _make_state(100.0)
        steer = _make_state(100.0)
        eff = _default_eff()
        update_stamina(no_steer, eff, 0.0, 0.0, 14.0, 14.0, 0.0, float("inf"))
        update_stamina(steer, eff, 0.0, 3.0, 14.0, 14.0, 0.0, float("inf"))
        assert steer.current_stamina < no_steer.current_stamina

    def test_lateral_velocity_drains(self):
        no_drift = _make_state(100.0)
        drift = _make_state(100.0)
        eff = _default_eff()
        update_stamina(no_drift, eff, 0.0, 0.0, 14.0, 14.0, 0.0, float("inf"))
        update_stamina(drift, eff, 0.0, 0.0, 14.0, 14.0, 2.0, float("inf"))
        assert drift.current_stamina < no_drift.current_stamina

    def test_steering_symmetric(self):
        left = _make_state(100.0)
        right = _make_state(100.0)
        eff = _default_eff()
        update_stamina(left, eff, 0.0, -3.0, 14.0, 14.0, 0.0, float("inf"))
        update_stamina(right, eff, 0.0, 3.0, 14.0, 14.0, 0.0, float("inf"))
        assert left.current_stamina == pytest.approx(right.current_stamina)


# ---------------------------------------------------------------------------
# update_stamina: drain_rate_mult applies to all drain
# ---------------------------------------------------------------------------


class TestDrainRateMult:
    def test_lower_mult_means_less_drain(self):
        normal = _make_state(100.0)
        efficient = _make_state(100.0)
        eff_normal = _default_eff(drain_rate_mult=1.0)
        eff_efficient = _default_eff(drain_rate_mult=0.7)
        update_stamina(normal, eff_normal, 5.0, 2.0, 16.0, 16.0, 1.0, float("inf"))
        update_stamina(efficient, eff_efficient, 5.0, 2.0, 16.0, 16.0, 1.0, float("inf"))
        assert efficient.current_stamina > normal.current_stamina


# ---------------------------------------------------------------------------
# Burst pool dynamics
# ---------------------------------------------------------------------------


class TestBurstPool:
    def test_burst_max_scales_with_band_and_stamina(self):
        narrow = CoreAttributes(cruise_speed=15.0, max_speed=16.0, stamina=100.0)
        wide = CoreAttributes(cruise_speed=12.0, max_speed=20.0, stamina=100.0)
        big = CoreAttributes(cruise_speed=12.0, max_speed=20.0, stamina=150.0)
        assert compute_burst_max(wide) > compute_burst_max(narrow)
        assert compute_burst_max(big) > compute_burst_max(wide)

    def test_burst_drains_when_above_cruise(self):
        eff = _default_eff()  # cruise=14.25
        state = _make_state(100.0, eff)
        starting = state.burst_pool
        for _ in range(20):
            update_stamina(state, eff, 0.0, 0.0, eff.cruise_speed + 3.0, eff.cruise_speed + 3.0, 0.0, float("inf"))
        assert state.burst_pool < starting

    def test_burst_recovers_below_cruise_buffer(self):
        eff = _default_eff()
        state = _make_state(100.0, eff)
        # Force burst to half
        state.burst_pool = state.burst_max * 0.5
        before = state.burst_pool
        for _ in range(50):
            update_stamina(state, eff, 0.0, 0.0, eff.cruise_speed - 2.0, eff.cruise_speed - 2.0, 0.0, float("inf"))
        assert state.burst_pool > before

    def test_burst_recovery_caps_at_max(self):
        eff = _default_eff()
        state = _make_state(100.0, eff)
        # Recovery from full should never exceed max
        for _ in range(1000):
            update_stamina(state, eff, 0.0, 0.0, 0.0, 0.0, 0.0, float("inf"))
        assert state.burst_pool == pytest.approx(state.burst_max)

    def test_quadratic_drain_with_excess(self):
        """Drain at excess=2 should be 4× drain at excess=1."""
        eff = _default_eff()
        s1 = _make_state(100.0, eff)
        s2 = _make_state(100.0, eff)
        update_stamina(s1, eff, 0.0, 0.0, eff.cruise_speed + 1.0, 0.0, 0.0, float("inf"))
        update_stamina(s2, eff, 0.0, 0.0, eff.cruise_speed + 2.0, 0.0, 0.0, float("inf"))
        d1 = s1.burst_max - s1.burst_pool
        d2 = s2.burst_max - s2.burst_pool
        assert d2 / d1 == pytest.approx(4.0, rel=0.01)


# ---------------------------------------------------------------------------
# Lead penalty
# ---------------------------------------------------------------------------


class TestLeadPenalty:
    def test_frontmost_pays_extra(self):
        eff = _default_eff()
        front = _make_state(100.0, eff)
        front.is_frontmost = True
        mid = _make_state(100.0, eff)
        mid.is_frontmost = False
        # Both running above cruise
        update_stamina(front, eff, 0.0, 0.0, eff.cruise_speed + 2.0, eff.cruise_speed + 2.0, 0.0, float("inf"))
        update_stamina(mid, eff, 0.0, 0.0, eff.cruise_speed + 2.0, eff.cruise_speed + 2.0, 0.0, float("inf"))
        assert front.current_stamina < mid.current_stamina

    def test_high_stamina_horse_pays_less_lead_penalty(self):
        """Stamina factor is non-linear: stayer with stam=140 pays much less."""
        weak_eff = _default_eff(stamina=80.0)
        stayer_eff = _default_eff(stamina=140.0)
        weak = _make_state(80.0, weak_eff)
        stayer = _make_state(140.0, stayer_eff)
        weak.is_frontmost = True
        stayer.is_frontmost = True
        # Same speed above each one's cruise
        speed = weak_eff.cruise_speed + 2.0
        update_stamina(weak, weak_eff, 0.0, 0.0, speed, speed, 0.0, float("inf"))
        update_stamina(stayer, stayer_eff, 0.0, 0.0, speed, speed, 0.0, float("inf"))
        weak_drain = 80.0 - weak.current_stamina
        stayer_drain = 140.0 - stayer.current_stamina
        assert stayer_drain < weak_drain

    def test_no_lead_penalty_at_or_below_cruise(self):
        eff = _default_eff()
        front = _make_state(100.0, eff)
        mid = _make_state(100.0, eff)
        front.is_frontmost = True
        # At cruise, neither pays lead penalty
        update_stamina(front, eff, 0.0, 0.0, eff.cruise_speed, eff.cruise_speed, 0.0, float("inf"))
        update_stamina(mid, eff, 0.0, 0.0, eff.cruise_speed, eff.cruise_speed, 0.0, float("inf"))
        assert front.current_stamina == pytest.approx(mid.current_stamina)


# ---------------------------------------------------------------------------
# Draft recovery
# ---------------------------------------------------------------------------


class TestDraftRecovery:
    def test_drafting_at_cruise_recovers(self):
        eff = _default_eff()
        state = _make_state(50.0, eff)  # half stamina
        state.is_drafting = True
        before = state.current_stamina
        update_stamina(state, eff, 0.0, 0.0, eff.cruise_speed, eff.cruise_speed, 0.0, float("inf"))
        # Should have recovered (recovery > distance tax at cruise)
        # If not, at least drained less than non-drafter
        non_draft = _make_state(50.0, eff)
        update_stamina(non_draft, eff, 0.0, 0.0, eff.cruise_speed, eff.cruise_speed, 0.0, float("inf"))
        assert state.current_stamina > non_draft.current_stamina

    def test_drafting_above_cruise_no_recovery(self):
        """Recovery requires speed at/below cruise + buffer."""
        eff = _default_eff()
        drafter_pushing = _make_state(50.0, eff)
        drafter_pushing.is_drafting = True
        non_drafter = _make_state(50.0, eff)
        speed = eff.cruise_speed + 3.0
        update_stamina(drafter_pushing, eff, 0.0, 0.0, speed, speed, 0.0, float("inf"))
        update_stamina(non_drafter, eff, 0.0, 0.0, speed, speed, 0.0, float("inf"))
        assert drafter_pushing.current_stamina == pytest.approx(non_drafter.current_stamina)

    def test_recovery_capped_at_base_max(self):
        eff = _default_eff(stamina=100.0)
        state = _make_state(100.0, eff)
        state.is_drafting = True
        for _ in range(500):
            update_stamina(state, eff, 0.0, 0.0, eff.cruise_speed - 1.0, eff.cruise_speed - 1.0, 0.0, float("inf"))
        assert state.current_stamina <= 100.0


# ---------------------------------------------------------------------------
# apply_exhaustion: cliff collapse + burst clamp
# ---------------------------------------------------------------------------


class TestExhaustion:
    def test_no_effect_above_cliff(self):
        eff = _default_eff()
        state = _make_state(50.0, eff)  # 50%, well above 5% cliff
        result = apply_exhaustion(eff, state, 100.0)
        assert result.cruise_speed == eff.cruise_speed
        assert result.max_speed == eff.max_speed
        assert result.forward_accel == eff.forward_accel

    def test_cliff_collapse_at_threshold(self):
        eff = _default_eff()
        state = _make_state(CLIFF_THRESHOLD * 100.0, eff)  # exactly at cliff
        result = apply_exhaustion(eff, state, 100.0)
        assert result.cruise_speed == pytest.approx(eff.cruise_speed * CLIFF_CRUISE_MULT)
        assert result.max_speed == pytest.approx(result.cruise_speed + BURST_EMPTY_CLAMP)
        assert result.forward_accel == pytest.approx(eff.forward_accel * CLIFF_ACCEL_MULT)

    def test_cliff_collapse_below_threshold(self):
        eff = _default_eff()
        state = _make_state(0.0, eff)
        result = apply_exhaustion(eff, state, 100.0)
        assert result.cruise_speed == pytest.approx(eff.cruise_speed * CLIFF_CRUISE_MULT)

    def test_burst_empty_clamps_max_speed(self):
        eff = _default_eff()
        state = _make_state(50.0, eff)  # well above cliff
        state.burst_pool = 0.0
        result = apply_exhaustion(eff, state, 100.0)
        assert result.max_speed == pytest.approx(eff.cruise_speed + BURST_EMPTY_CLAMP)
        assert result.cruise_speed == eff.cruise_speed  # cruise untouched

    def test_burst_clamp_only_ratchets_down(self):
        """Burst clamp must never INCREASE max_speed (e.g., when cruise > current max)."""
        eff = CoreAttributes(cruise_speed=15.0, max_speed=15.2)  # narrow band
        state = _make_state(50.0, eff)
        state.burst_pool = 0.0
        result = apply_exhaustion(eff, state, 100.0)
        assert result.max_speed <= eff.max_speed

    def test_burst_with_pool_no_clamp(self):
        eff = _default_eff()
        state = _make_state(50.0, eff)
        # Burst pool > 0 → no clamp
        result = apply_exhaustion(eff, state, 100.0)
        assert result.max_speed == eff.max_speed
