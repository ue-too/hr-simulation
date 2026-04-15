import pytest

from horse_racing.reward import (
    DEPLETION_PENALTY,
    DEPLETION_THRESHOLD,
    FINISHING_SPEED_BONUS,
    OVERTAKE_BONUS,
    RAIL_COLLISION_PENALTY,
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
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        finishing_speed=9.0, cruise_speed=13.0,
    )
    # progress + placement + speed bonus
    expected = 0.01 + 10.0 + FINISHING_SPEED_BONUS * (9.0 / 13.0)
    assert reward == pytest.approx(expected)


def test_second_place_bonus():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=2,
        finishing_speed=9.0, cruise_speed=13.0,
    )
    expected = 0.01 + 5.0 + FINISHING_SPEED_BONUS * (9.0 / 13.0)
    assert reward == pytest.approx(expected)


def test_third_place_bonus():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=3,
        finishing_speed=9.0, cruise_speed=13.0,
    )
    expected = 0.01 + 2.0 + FINISHING_SPEED_BONUS * (9.0 / 13.0)
    assert reward == pytest.approx(expected)


def test_no_bonus_for_fourth():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=4,
        finishing_speed=9.0, cruise_speed=13.0,
    )
    expected = 0.01 + FINISHING_SPEED_BONUS * (9.0 / 13.0)
    assert reward == pytest.approx(expected)


def test_speed_bonus_higher_when_faster():
    """A horse finishing at 10 m/s gets more bonus than one at 6.6 m/s."""
    fast = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        finishing_speed=10.0, cruise_speed=13.0,
    )
    slow = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        finishing_speed=6.6, cruise_speed=13.0,
    )
    assert fast > slow


def test_speed_bonus_zero_when_stopped():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        finishing_speed=0.0, cruise_speed=13.0,
    )
    # No speed bonus + full depletion penalty (0.6 below threshold)
    expected = 0.01 + 10.0 - DEPLETION_PENALTY * DEPLETION_THRESHOLD
    assert reward == pytest.approx(expected)


def test_depletion_penalty_below_threshold():
    """Finishing below threshold incurs penalty."""
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        finishing_speed=6.5, cruise_speed=13.0,  # 0.5× cruise
    )
    speed_ratio = 6.5 / 13.0
    expected = 0.01 + 10.0 + FINISHING_SPEED_BONUS * speed_ratio - DEPLETION_PENALTY * (DEPLETION_THRESHOLD - speed_ratio)
    assert reward == pytest.approx(expected)


def test_no_depletion_penalty_above_threshold():
    """Finishing above threshold has no penalty."""
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        finishing_speed=9.1, cruise_speed=13.0,  # 0.7× cruise
    )
    speed_ratio = 9.1 / 13.0
    expected = 0.01 + 10.0 + FINISHING_SPEED_BONUS * speed_ratio
    assert reward == pytest.approx(expected)


def test_no_speed_bonus_mid_race():
    """Speed bonus only triggers at finish."""
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.51, finish_order=None,
        finishing_speed=14.0, cruise_speed=13.0,
    )
    assert reward == pytest.approx(0.01)


def test_no_exhaustion_penalty():
    """Zero stamina mid-race has no direct penalty."""
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.51, finish_order=None,
    )
    assert reward == pytest.approx(0.01)


def test_overtake_bonus():
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.51, finish_order=None,
        overtakes=2,
    )
    assert reward == pytest.approx(0.01 + OVERTAKE_BONUS * 2)


def test_no_overtake_bonus_when_zero():
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.51, finish_order=None,
        overtakes=0,
    )
    assert reward == pytest.approx(0.01)


def test_overtake_bonus_at_finish():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        finishing_speed=9.0, cruise_speed=13.0,
        overtakes=1,
    )
    expected = 0.01 + OVERTAKE_BONUS + 10.0 + FINISHING_SPEED_BONUS * (9.0 / 13.0)
    assert reward == pytest.approx(expected)


def test_rail_collision_penalty():
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.51, finish_order=None,
        rail_contact=True,
    )
    assert reward == pytest.approx(0.01 - RAIL_COLLISION_PENALTY)


def test_no_rail_penalty_when_no_contact():
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.51, finish_order=None,
        rail_contact=False,
    )
    assert reward == pytest.approx(0.01)
