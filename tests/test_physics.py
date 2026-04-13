import math

import numpy as np
import pytest

from horse_racing.core.attributes import create_default_attributes
from horse_racing.core.physics import compute_accelerations, project_velocity
from horse_racing.core.track_navigator import TrackFrame
from horse_racing.core.types import InputState

ZERO_INPUT = InputState(0.0, 0.0)


def straight_frame() -> TrackFrame:
    return TrackFrame(
        tangential=np.array([1.0, 0.0]),
        normal=np.array([0.0, -1.0]),
        turn_radius=math.inf,
        nominal_radius=math.inf,
        target_radius=math.inf,
        slope=0.0,
    )


def curve_frame(turn_radius: float) -> TrackFrame:
    return TrackFrame(
        tangential=np.array([0.0, -1.0]),
        normal=np.array([1.0, 0.0]),
        turn_radius=turn_radius,
        nominal_radius=turn_radius,
        target_radius=turn_radius,
        slope=0.0,
    )


class TestProjectVelocity:
    def test_decomposes_world_velocity(self):
        t, n = project_velocity(np.array([10.0, -3.0]), straight_frame())
        assert t == pytest.approx(10.0)
        assert n == pytest.approx(3.0)

    def test_angled_frame(self):
        s = math.sqrt(0.5)
        frame = TrackFrame(
            tangential=np.array([s, s]),
            normal=np.array([s, -s]),
            turn_radius=math.inf,
            nominal_radius=math.inf,
            target_radius=math.inf,
            slope=0.0,
        )
        t, n = project_velocity(np.array([10.0, 0.0]), frame)
        assert t == pytest.approx(10.0 * s)
        assert n == pytest.approx(10.0 * s)


class TestComputeAccelerations:
    def test_positive_tangential_at_zero_velocity(self):
        attrs = create_default_attributes()
        a_t, _ = compute_accelerations(0, 0, attrs, ZERO_INPUT, straight_frame())
        assert a_t == pytest.approx(26, abs=1)

    def test_negative_tangential_above_cruise(self):
        attrs = create_default_attributes()
        a_t, _ = compute_accelerations(20, 0, attrs, ZERO_INPUT, straight_frame())
        assert a_t < 0

    def test_centripetal_on_curves(self):
        attrs = create_default_attributes()
        _, a_n = compute_accelerations(10, 0, attrs, ZERO_INPUT, curve_frame(100))
        assert a_n == pytest.approx(-1.0, abs=0.1)

    def test_no_centripetal_on_straights(self):
        attrs = create_default_attributes()
        _, a_n = compute_accelerations(10, 0, attrs, ZERO_INPUT, straight_frame())
        assert a_n == pytest.approx(0.0, abs=1e-5)

    def test_normal_damp(self):
        attrs = create_default_attributes()
        _, a_n = compute_accelerations(0, 5, attrs, ZERO_INPUT, straight_frame())
        assert a_n == pytest.approx(-3.0, abs=0.1)

    def test_steering_input(self):
        attrs = create_default_attributes()
        _, a_n = compute_accelerations(0, 0, attrs, InputState(0, 1), straight_frame())
        assert a_n == pytest.approx(3, abs=1)

    def test_uphill_reduces_tangential(self):
        attrs = create_default_attributes()
        uphill = TrackFrame(
            tangential=np.array([1.0, 0.0]),
            normal=np.array([0.0, -1.0]),
            turn_radius=math.inf,
            nominal_radius=math.inf,
            target_radius=math.inf,
            slope=0.05,
        )
        a_flat, _ = compute_accelerations(10, 0, attrs, ZERO_INPUT, straight_frame())
        a_up, _ = compute_accelerations(10, 0, attrs, ZERO_INPUT, uphill)
        assert a_up < a_flat
        assert a_flat - a_up == pytest.approx(9.81 * 0.05, abs=0.01)

    def test_downhill_increases_tangential(self):
        attrs = create_default_attributes()
        downhill = TrackFrame(
            tangential=np.array([1.0, 0.0]),
            normal=np.array([0.0, -1.0]),
            turn_radius=math.inf,
            nominal_radius=math.inf,
            target_radius=math.inf,
            slope=-0.05,
        )
        a_flat, _ = compute_accelerations(10, 0, attrs, ZERO_INPUT, straight_frame())
        a_down, _ = compute_accelerations(10, 0, attrs, ZERO_INPUT, downhill)
        assert a_down > a_flat
        assert a_down - a_flat == pytest.approx(9.81 * 0.05, abs=0.01)

    def test_clamps_input(self):
        attrs = create_default_attributes()
        a_t1, a_n1 = compute_accelerations(0, 0, attrs, InputState(5, -3), straight_frame())
        a_t2, a_n2 = compute_accelerations(0, 0, attrs, InputState(1, -1), straight_frame())
        assert a_t1 == pytest.approx(a_t2)
        assert a_n1 == pytest.approx(a_n2)
