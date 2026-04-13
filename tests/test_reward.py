import pytest

from horse_racing.reward import (
    EXHAUSTION_PENALTY,
    STEP_PENALTY,
    compute_reward,
)


def test_positive_progress():
    reward = compute_reward(
        prev_progress=0.1, curr_progress=0.15, finish_order=None
    )
    assert reward == pytest.approx(0.05 + STEP_PENALTY)


def test_no_progress():
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.5, finish_order=None
    )
    assert reward == pytest.approx(STEP_PENALTY)


def test_first_place_bonus():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1
    )
    assert reward == pytest.approx(0.01 + 10.0 + STEP_PENALTY)


def test_second_place_bonus():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=2
    )
    assert reward == pytest.approx(0.01 + 5.0 + STEP_PENALTY)


def test_third_place_bonus():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=3
    )
    assert reward == pytest.approx(0.01 + 2.0 + STEP_PENALTY)


def test_no_bonus_for_fourth():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=4
    )
    assert reward == pytest.approx(0.01 + STEP_PENALTY)


def test_exhaustion_penalty_when_stamina_zero():
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.51, finish_order=None,
        current_stamina=0.0,
    )
    assert reward == pytest.approx(0.01 + STEP_PENALTY + EXHAUSTION_PENALTY)


def test_no_penalty_when_stamina_positive():
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.51, finish_order=None,
        current_stamina=50.0,
    )
    assert reward == pytest.approx(0.01 + STEP_PENALTY)


def test_step_penalty_always_applied():
    """Step penalty applies even with zero progress and no finish."""
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.5, finish_order=None
    )
    assert reward < 0.0
    assert reward == pytest.approx(STEP_PENALTY)
