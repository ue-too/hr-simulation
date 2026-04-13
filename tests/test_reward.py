import pytest

from horse_racing.reward import (
    EXHAUSTION_PENALTY,
    PAR_TICKS,
    TIME_BONUS_MAX,
    compute_reward,
)


def test_positive_progress():
    reward = compute_reward(
        prev_progress=0.1, curr_progress=0.15, finish_order=None
    )
    assert reward == pytest.approx(0.05)


def test_no_progress():
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.5, finish_order=None
    )
    assert reward == pytest.approx(0.0)


def test_first_place_bonus():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1
    )
    assert reward == pytest.approx(0.01 + 10.0)


def test_second_place_bonus():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=2
    )
    assert reward == pytest.approx(0.01 + 5.0)


def test_third_place_bonus():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=3
    )
    assert reward == pytest.approx(0.01 + 2.0)


def test_no_bonus_for_fourth():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=4
    )
    assert reward == pytest.approx(0.01)


def test_exhaustion_penalty_when_stamina_zero():
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.51, finish_order=None,
        current_stamina=0.0,
    )
    assert reward == pytest.approx(0.01 + EXHAUSTION_PENALTY)


def test_no_penalty_when_stamina_positive():
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.51, finish_order=None,
        current_stamina=50.0,
    )
    assert reward == pytest.approx(0.01)


def test_time_bonus_at_fast_finish():
    """Finishing well under PAR_TICKS earns a time bonus."""
    tick = 1800
    expected_bonus = TIME_BONUS_MAX * (1.0 - tick / PAR_TICKS)
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        finish_tick=tick,
    )
    assert reward == pytest.approx(0.01 + 10.0 + expected_bonus)


def test_no_time_bonus_at_par():
    """Finishing at exactly PAR_TICKS earns zero time bonus."""
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        finish_tick=PAR_TICKS,
    )
    assert reward == pytest.approx(0.01 + 10.0)


def test_no_time_bonus_when_slow():
    """Finishing slower than PAR_TICKS still earns zero time bonus (clamped)."""
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        finish_tick=PAR_TICKS + 500,
    )
    assert reward == pytest.approx(0.01 + 10.0)


def test_no_time_bonus_without_finish_tick():
    """No time bonus when finish_tick is not provided (backwards compat)."""
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
    )
    assert reward == pytest.approx(0.01 + 10.0)
