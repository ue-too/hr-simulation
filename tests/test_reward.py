import pytest

from horse_racing.reward import (
    DEPLETION_PENALTY,
    DEPLETION_THRESHOLD,
    FINISHING_SPEED_BONUS,
    OVERTAKE_BONUS,
    PACING_BONUS,
    POSITION_BONUS,
    RAIL_COLLISION_PENALTY,
    SMOOTHNESS_PENALTY,
    SPEED_BAND_BONUS,
    SPEED_BAND_HIGH,
    SPEED_BAND_LOW,
    SPEED_BAND_PHASE,
    _ideal_stamina,
    compute_reward,
)


def _base(prev_prog=0.5, curr_prog=0.51, **kw):
    """Call compute_reward with neutral defaults for new params."""
    defaults = dict(
        finish_order=None,
        stamina_frac=_ideal_stamina(curr_prog),
        speed_ratio=(SPEED_BAND_LOW + SPEED_BAND_HIGH) / 2,  # in band
        rank=3,  # mid-pack
        horse_count=4,
        prev_tang=0.5,
        curr_tang=0.5,
        prev_norm=0.0,
        curr_norm=0.0,
    )
    defaults.update(kw)
    return compute_reward(prev_prog, curr_prog, **defaults)


# --- Progress ---

def test_positive_progress():
    r = _base(prev_prog=0.1, curr_prog=0.15)
    # progress (0.05) + pacing + speed band + position (in pack)
    assert r > 0.05


def test_no_progress():
    r = _base(prev_prog=0.5, curr_prog=0.5)
    # no progress, but still gets pacing/speed/position bonuses
    assert r >= 0


# --- Placement ---

def test_first_place_bonus():
    r = _base(prev_prog=0.99, curr_prog=1.0, finish_order=1,
              finishing_speed=9.0, cruise_speed=13.0)
    # Should include 10.0 placement
    assert r > 10.0


def test_placement_ordering():
    r1 = _base(prev_prog=0.99, curr_prog=1.0, finish_order=1,
               finishing_speed=9.0, cruise_speed=13.0)
    r2 = _base(prev_prog=0.99, curr_prog=1.0, finish_order=2,
               finishing_speed=9.0, cruise_speed=13.0)
    r3 = _base(prev_prog=0.99, curr_prog=1.0, finish_order=3,
               finishing_speed=9.0, cruise_speed=13.0)
    assert r1 > r2 > r3


# --- Speed bonus ---

def test_speed_bonus_higher_when_faster():
    fast = _base(prev_prog=0.99, curr_prog=1.0, finish_order=1,
                 finishing_speed=10.0, cruise_speed=13.0)
    slow = _base(prev_prog=0.99, curr_prog=1.0, finish_order=1,
                 finishing_speed=6.6, cruise_speed=13.0)
    assert fast > slow


def test_depletion_penalty():
    # Finishing at 0.5× cruise triggers penalty
    r = _base(prev_prog=0.99, curr_prog=1.0, finish_order=1,
              finishing_speed=6.5, cruise_speed=13.0)
    # Without penalty would be higher
    r_no_penalty = _base(prev_prog=0.99, curr_prog=1.0, finish_order=1,
                         finishing_speed=9.1, cruise_speed=13.0)
    # Paced finish should beat depleted finish at same placement
    assert r_no_penalty > r


# --- Pacing ---

def test_pacing_bonus_above_ideal():
    r_above = _base(stamina_frac=0.8)  # well above ideal at 50%
    r_below = _base(stamina_frac=0.2)  # well below ideal at 50%
    assert r_above > r_below


def test_pacing_penalty_below_ideal():
    ideal = _ideal_stamina(0.5)
    r = _base(stamina_frac=0.1)  # far below ideal
    r_at = _base(stamina_frac=ideal)  # at ideal
    assert r < r_at


# --- Speed band ---

def test_speed_band_reward_in_band():
    """Speed in cruise band during first 70% gets bonus."""
    r_in = _base(curr_prog=0.3, speed_ratio=0.72)
    r_out = _base(curr_prog=0.3, speed_ratio=0.95)
    assert r_in > r_out


def test_speed_band_no_effect_after_phase():
    """Speed band doesn't apply in final 30%."""
    r_fast = _base(curr_prog=0.85, speed_ratio=0.95)
    r_cruise = _base(curr_prog=0.85, speed_ratio=0.72)
    # Neither should get speed band bonus/penalty after 70%
    # Difference should be small (no speed band component)
    assert abs(r_fast - r_cruise) < SPEED_BAND_BONUS + 0.001


def test_speed_band_penalizes_overshoot():
    """Pushing too fast early gets penalized."""
    r_cruise = _base(curr_prog=0.3, speed_ratio=0.72)
    r_push = _base(curr_prog=0.3, speed_ratio=0.95)
    assert r_cruise > r_push


# --- Tactical positioning ---

def test_position_stalk_early():
    """Being in pack (2nd-4th) is rewarded early."""
    r_pack = _base(curr_prog=0.2, rank=3)
    r_lead = _base(curr_prog=0.2, rank=1)
    assert r_pack > r_lead


def test_position_kick_late():
    """Leading is rewarded in final stretch."""
    r_lead = _base(curr_prog=0.85, rank=1)
    r_back = _base(curr_prog=0.85, rank=4)
    assert r_lead > r_back


def test_position_no_effect_mid():
    """Middle phase (50-75%) has no position bonus."""
    r_lead = _base(curr_prog=0.6, rank=1)
    r_back = _base(curr_prog=0.6, rank=4)
    # Should be similar (no position component)
    assert abs(r_lead - r_back) < POSITION_BONUS + 0.001


# --- Smoothness ---

def test_smoothness_penalty_large_tang_change():
    """Large tangential flip is penalized."""
    r_smooth = _base(prev_tang=0.5, curr_tang=0.75)
    r_flip = _base(prev_tang=-0.25, curr_tang=1.0)  # 1.25 change
    assert r_smooth > r_flip


def test_smoothness_no_penalty_small_change():
    """Small tangential changes are not penalized."""
    r1 = _base(prev_tang=0.5, curr_tang=0.75)  # 0.25 change
    r2 = _base(prev_tang=0.5, curr_tang=0.5)   # no change
    # Difference should be negligible (no smoothness component)
    assert abs(r1 - r2) < 0.001


# --- Overtake ---

def test_overtake_bonus():
    r_pass = _base(overtakes=2)
    r_none = _base(overtakes=0)
    assert r_pass == pytest.approx(r_none + OVERTAKE_BONUS * 2)


# --- Rail ---

def test_rail_collision_penalty():
    r_contact = _base(rail_contact=True)
    r_clear = _base(rail_contact=False)
    assert r_clear == pytest.approx(r_contact + RAIL_COLLISION_PENALTY)
