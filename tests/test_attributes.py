"""Tests for attributes and modifier resolution."""

from horse_racing.attributes import CoreAttributes, TRAIT_RANGES, resolve_effective
from horse_racing.modifiers import ActiveModifier


def test_default_attributes():
    attrs = CoreAttributes()
    assert attrs.cruise_speed == 13.0
    assert attrs.weight == 500.0


def test_resolve_no_modifiers():
    base = CoreAttributes()
    result = resolve_effective(base, [])
    assert result.cruise_speed == base.cruise_speed
    assert result.weight == base.weight


def test_resolve_flat_modifier():
    base = CoreAttributes(cruise_speed=13.0)
    # front_runner gives flat +1.5 to cruise_speed at full strength
    active = [ActiveModifier(id="front_runner", strength=1.0)]
    result = resolve_effective(base, active)
    assert result.cruise_speed > base.cruise_speed
    # 13.0 + 1.5 * 1.0 = 14.5
    assert abs(result.cruise_speed - 14.5) < 0.01


def test_resolve_pct_modifier():
    base = CoreAttributes(forward_accel=1.0)
    # gate_speed gives +25% forward_accel at full strength
    active = [ActiveModifier(id="gate_speed", strength=1.0)]
    result = resolve_effective(base, active)
    assert abs(result.forward_accel - 1.25) < 0.01


def test_resolve_clamps_to_range():
    base = CoreAttributes(cruise_speed=17.5)
    # front_runner flat +1.5, could push to 19.0 but max is 18.0
    active = [ActiveModifier(id="front_runner", strength=1.0)]
    result = resolve_effective(base, active)
    assert result.cruise_speed <= TRAIT_RANGES["cruise_speed"][1]


def test_trait_ranges_cover_all_attributes():
    attrs = CoreAttributes()
    for trait in TRAIT_RANGES:
        assert hasattr(attrs, trait), f"CoreAttributes missing trait: {trait}"
