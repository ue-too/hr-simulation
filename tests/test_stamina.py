import math

import numpy as np
import pytest

from horse_racing.core.attributes import create_default_attributes
from horse_racing.core.stamina import (
    OVERDRIVE_DRAIN_RATE,
    STAMINA_DRAIN_RATE,
    LATERAL_STEERING_DRAIN_RATE,
    SPEED_DRAIN_RATE,
    drain_stamina,
)
from horse_racing.core.track_navigator import TrackFrame
from horse_racing.core.types import Horse, InputState


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


def flat_straight() -> TrackFrame:
    return TrackFrame(
        tangential=np.array([1.0, 0.0]),
        normal=np.array([0.0, -1.0]),
        turn_radius=math.inf,
        nominal_radius=math.inf,
        target_radius=math.inf,
        slope=0.0,
    )


class TestDrainStamina:
    def test_no_drain_at_cruise_zero_input(self):
        horse = make_horse(tangential_vel=13.0)
        attrs = horse.base_attributes
        drain_stamina(horse, attrs, InputState(0, 0), flat_straight())
        expected = 100.0 - 13.0 * SPEED_DRAIN_RATE
        assert horse.current_stamina == pytest.approx(expected, abs=0.01)

    def test_overdrive_drain(self):
        horse = make_horse(tangential_vel=18.0)
        attrs = horse.base_attributes
        drain_stamina(horse, attrs, InputState(0, 0), flat_straight())
        overdrive = 5.0 * OVERDRIVE_DRAIN_RATE
        speed_tax = 18.0 * SPEED_DRAIN_RATE
        total = overdrive + speed_tax
        assert horse.current_stamina == pytest.approx(100.0 - total, abs=0.01)

    def test_jockey_push_drain(self):
        horse = make_horse()
        attrs = horse.base_attributes
        drain_stamina(horse, attrs, InputState(1.0, 0), flat_straight())
        push = STAMINA_DRAIN_RATE
        assert horse.current_stamina < 100.0
        assert horse.current_stamina == pytest.approx(100.0 - push, abs=0.01)

    def test_steering_drain(self):
        horse = make_horse()
        attrs = horse.base_attributes
        drain_stamina(horse, attrs, InputState(0, 0.5), flat_straight())
        steer = 0.5 * LATERAL_STEERING_DRAIN_RATE
        assert horse.current_stamina == pytest.approx(100.0 - steer, abs=0.01)

    def test_stamina_floors_at_zero(self):
        horse = make_horse(current_stamina=0.001, tangential_vel=20.0)
        attrs = horse.base_attributes
        drain_stamina(horse, attrs, InputState(1.0, 1.0), flat_straight())
        assert horse.current_stamina == 0.0

    def test_drain_rate_mult_scales(self):
        import dataclasses
        attrs = dataclasses.replace(create_default_attributes(), drain_rate_mult=2.0)
        horse = make_horse(tangential_vel=18.0, base_attributes=attrs, effective_attributes=attrs)
        drain_stamina(horse, attrs, InputState(0, 0), flat_straight())
        horse2 = make_horse(tangential_vel=18.0)
        drain_stamina(horse2, horse2.base_attributes, InputState(0, 0), flat_straight())
        drain_1x = 100.0 - horse2.current_stamina
        drain_2x = 100.0 - horse.current_stamina
        assert drain_2x == pytest.approx(drain_1x * 2.0, abs=0.001)


class TestLastDrain:
    def test_last_drain_recorded(self):
        horse = make_horse(tangential_vel=18.0)
        attrs = horse.base_attributes
        drain_stamina(horse, attrs, InputState(1.0, 0), flat_straight())
        assert horse.last_drain > 0.0

    def test_last_drain_matches_stamina_change(self):
        horse = make_horse(tangential_vel=13.0)
        attrs = horse.base_attributes
        before = horse.current_stamina
        drain_stamina(horse, attrs, InputState(0.5, 0), flat_straight())
        actual_drain = before - horse.current_stamina
        assert horse.last_drain == pytest.approx(actual_drain)

    def test_last_drain_zero_default(self):
        horse = make_horse()
        assert horse.last_drain == 0.0


class TestDrainScale:
    def test_reference_track_returns_one(self):
        """A track that takes exactly REFERENCE_CRUISE_TICKS at cruise speed."""
        from horse_racing.core.stamina import compute_drain_scale, REFERENCE_CRUISE_TICKS
        from horse_racing.core.types import FIXED_DT, PHYS_SUBSTEPS
        cruise_speed = 13.0
        # length = cruise_speed * dt_per_tick * reference_ticks
        dt_per_tick = FIXED_DT * PHYS_SUBSTEPS
        length = cruise_speed * dt_per_tick * REFERENCE_CRUISE_TICKS
        scale = compute_drain_scale(length, cruise_speed)
        assert scale == pytest.approx(1.0, abs=0.01)

    def test_longer_track_reduces_drain(self):
        """A track twice as long should drain at half the rate."""
        from horse_racing.core.stamina import compute_drain_scale, REFERENCE_CRUISE_TICKS
        from horse_racing.core.types import FIXED_DT, PHYS_SUBSTEPS
        cruise_speed = 13.0
        dt_per_tick = FIXED_DT * PHYS_SUBSTEPS
        ref_length = cruise_speed * dt_per_tick * REFERENCE_CRUISE_TICKS
        scale = compute_drain_scale(ref_length * 2, cruise_speed)
        assert scale == pytest.approx(0.5, abs=0.01)

    def test_shorter_track_increases_drain(self):
        """A track half as long should drain at double the rate."""
        from horse_racing.core.stamina import compute_drain_scale, REFERENCE_CRUISE_TICKS
        from horse_racing.core.types import FIXED_DT, PHYS_SUBSTEPS
        cruise_speed = 13.0
        dt_per_tick = FIXED_DT * PHYS_SUBSTEPS
        ref_length = cruise_speed * dt_per_tick * REFERENCE_CRUISE_TICKS
        scale = compute_drain_scale(ref_length / 2, cruise_speed)
        assert scale == pytest.approx(2.0, abs=0.01)

    def test_drain_scale_applied(self):
        """drain_scale=0.5 should halve the drain."""
        h1 = make_horse(tangential_vel=18.0)
        h2 = make_horse(tangential_vel=18.0)
        attrs = h1.base_attributes
        frame = flat_straight()
        drain_stamina(h1, attrs, InputState(1.0, 0), frame, drain_scale=1.0)
        drain_stamina(h2, attrs, InputState(1.0, 0), frame, drain_scale=0.5)
        drain_1x = 100.0 - h1.current_stamina
        drain_half = 100.0 - h2.current_stamina
        assert drain_half == pytest.approx(drain_1x * 0.5, abs=0.001)
