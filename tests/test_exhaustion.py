import math

import numpy as np
import pytest

from horse_racing.core.attributes import create_default_attributes
from horse_racing.core.exhaustion import apply_exhaustion, effective_ratio
from horse_racing.core.types import Horse


def make_horse(**kwargs) -> Horse:
    attrs = create_default_attributes()
    defaults = dict(
        id=0, color=0, pos=np.array([0.0, 0.0]),
        tangential_vel=0.0, normal_vel=0.0, track_progress=0.0,
        navigator=None, finished=False, finish_order=None,
        base_attributes=attrs, current_stamina=attrs.max_stamina,
        effective_attributes=attrs,
    )
    defaults.update(kwargs)
    return Horse(**defaults)


class TestEffectiveRatio:
    def test_full_stamina_returns_one(self):
        assert effective_ratio(1.0) == pytest.approx(1.0, abs=1e-3)

    def test_zero_stamina_returns_floor(self):
        assert effective_ratio(0.0) == pytest.approx(0.45, abs=1e-3)

    def test_monotonically_increasing(self):
        values = [effective_ratio(p / 100) for p in range(101)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

    def test_knee_region_has_steep_drop(self):
        """Between 30% and 10% stamina, ratio should drop significantly."""
        high = effective_ratio(0.3)
        low = effective_ratio(0.1)
        assert high - low > 0.2

    def test_above_knee_gentle(self):
        """Between 80% and 60% stamina, ratio barely changes."""
        high = effective_ratio(0.8)
        low = effective_ratio(0.6)
        assert high - low < 0.02

    def test_floor_is_respected(self):
        """Even at 0 stamina, ratio never goes below floor."""
        assert effective_ratio(0.0) >= 0.45
        assert effective_ratio(0.01) >= 0.45

    def test_custom_parameters(self):
        r = effective_ratio(0.5, knee=0.5, floor=0.3, k=15)
        assert 0.3 < r < 1.0


class TestApplyExhaustion:
    def test_full_stamina_no_degradation(self):
        horse = make_horse(current_stamina=100.0)
        result = apply_exhaustion(horse)
        base = horse.base_attributes
        assert result.cruise_speed == pytest.approx(base.cruise_speed, abs=0.1)
        assert result.max_speed == pytest.approx(base.max_speed, abs=0.1)
        assert result.forward_accel == pytest.approx(base.forward_accel, abs=0.01)
        assert result.turn_accel == pytest.approx(base.turn_accel, abs=0.01)

    def test_zero_stamina_hits_floor(self):
        horse = make_horse(current_stamina=0.0)
        result = apply_exhaustion(horse)
        base = horse.base_attributes
        assert result.cruise_speed == pytest.approx(base.cruise_speed * 0.45, abs=0.1)
        assert result.max_speed == pytest.approx(base.max_speed * 0.45, abs=0.1)

    def test_half_stamina_mild_degradation(self):
        horse = make_horse(current_stamina=50.0)
        result = apply_exhaustion(horse)
        base = horse.base_attributes
        # At 50% stamina, ratio ~0.94 — still mostly full
        assert result.cruise_speed > base.cruise_speed * 0.9
        assert result.cruise_speed < base.cruise_speed

    def test_low_stamina_significant_degradation(self):
        horse = make_horse(current_stamina=10.0)
        result = apply_exhaustion(horse)
        base = horse.base_attributes
        # At 10% stamina, ratio ~0.50 — heavily degraded
        assert result.cruise_speed < base.cruise_speed * 0.55

    def test_no_iterative_decay(self):
        """New model is stateless — same stamina always gives same result."""
        horse = make_horse(current_stamina=30.0)
        r1 = apply_exhaustion(horse)
        horse.effective_attributes = r1
        r2 = apply_exhaustion(horse)
        assert r1.cruise_speed == pytest.approx(r2.cruise_speed)

    def test_cornering_grip_unchanged(self):
        """cornering_grip is not affected by exhaustion."""
        horse = make_horse(current_stamina=0.0)
        result = apply_exhaustion(horse)
        assert result.cornering_grip == horse.base_attributes.cornering_grip

    def test_weight_unchanged(self):
        horse = make_horse(current_stamina=0.0)
        result = apply_exhaustion(horse)
        assert result.weight == horse.base_attributes.weight
