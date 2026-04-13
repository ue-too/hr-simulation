import numpy as np
import pytest

from horse_racing.core.attributes import create_default_attributes
from horse_racing.core.exhaustion import EXHAUSTION_DECAY, apply_exhaustion
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


class TestApplyExhaustion:
    def test_no_penalty_above_zero_stamina(self):
        horse = make_horse(current_stamina=50.0)
        result = apply_exhaustion(horse)
        assert result.cruise_speed == horse.base_attributes.cruise_speed
        assert result.max_speed == horse.base_attributes.max_speed

    def test_decay_at_zero_stamina(self):
        horse = make_horse(current_stamina=0.0)
        result = apply_exhaustion(horse)
        assert result.cruise_speed < horse.base_attributes.cruise_speed

    def test_decay_approaches_floor(self):
        horse = make_horse(current_stamina=0.0)
        for _ in range(300):
            horse.effective_attributes = apply_exhaustion(horse)
        floor = horse.base_attributes.cruise_speed * 0.4
        assert horse.effective_attributes.cruise_speed == pytest.approx(floor, abs=0.1)

    def test_forward_accel_floor(self):
        horse = make_horse(current_stamina=0.0)
        for _ in range(300):
            horse.effective_attributes = apply_exhaustion(horse)
        floor = horse.base_attributes.forward_accel * 0.15
        assert horse.effective_attributes.forward_accel == pytest.approx(floor, abs=0.01)

    def test_snaps_back_when_stamina_restored(self):
        horse = make_horse(current_stamina=0.0)
        for _ in range(60):
            horse.effective_attributes = apply_exhaustion(horse)
        assert horse.effective_attributes.cruise_speed < horse.base_attributes.cruise_speed
        horse.current_stamina = 50.0
        result = apply_exhaustion(horse)
        assert result.cruise_speed == horse.base_attributes.cruise_speed
