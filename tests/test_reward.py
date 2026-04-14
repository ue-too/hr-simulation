import pytest

from horse_racing.reward import OVERTAKE_BONUS, RAIL_COLLISION_PENALTY, STAMINA_EFFICIENCY_BONUS, compute_reward


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
        current_stamina=0.0, max_stamina=100.0,
    )
    # progress + placement + efficiency (used all stamina)
    assert reward == pytest.approx(0.01 + 10.0 + STAMINA_EFFICIENCY_BONUS)


def test_second_place_bonus():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=2,
        current_stamina=0.0, max_stamina=100.0,
    )
    assert reward == pytest.approx(0.01 + 5.0 + STAMINA_EFFICIENCY_BONUS)


def test_third_place_bonus():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=3,
        current_stamina=0.0, max_stamina=100.0,
    )
    assert reward == pytest.approx(0.01 + 2.0 + STAMINA_EFFICIENCY_BONUS)


def test_no_bonus_for_fourth():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=4,
        current_stamina=0.0, max_stamina=100.0,
    )
    assert reward == pytest.approx(0.01 + STAMINA_EFFICIENCY_BONUS)


def test_efficiency_bonus_scales_with_stamina_used():
    # Finish with 50% stamina = half the efficiency bonus
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        current_stamina=50.0, max_stamina=100.0,
    )
    expected = 0.01 + 10.0 + STAMINA_EFFICIENCY_BONUS * 0.5
    assert reward == pytest.approx(expected)


def test_efficiency_bonus_zero_when_full_stamina():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        current_stamina=100.0, max_stamina=100.0,
    )
    assert reward == pytest.approx(0.01 + 10.0)


def test_no_efficiency_bonus_mid_race():
    """Efficiency bonus only triggers at finish (finish_order is not None)."""
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.51, finish_order=None,
        current_stamina=10.0, max_stamina=100.0,
    )
    assert reward == pytest.approx(0.01)


def test_no_exhaustion_penalty():
    """Old exhaustion penalty is removed — zero stamina mid-race has no direct penalty."""
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.51, finish_order=None,
        current_stamina=0.0, max_stamina=100.0,
    )
    assert reward == pytest.approx(0.01)


def test_overtake_bonus():
    """Passing opponents gives a small bonus."""
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
    """Overtake bonus stacks with finish bonus."""
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        current_stamina=0.0, max_stamina=100.0,
        overtakes=1,
    )
    expected = 0.01 + OVERTAKE_BONUS + 10.0 + STAMINA_EFFICIENCY_BONUS
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
