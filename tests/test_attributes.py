from horse_racing.core.attributes import (
    CoreAttributes,
    TRAIT_RANGES,
    F_T_MAX,
    F_N_MAX,
    create_default_attributes,
)


def test_default_attributes_values():
    attrs = create_default_attributes()
    assert attrs.cruise_speed == 13
    assert attrs.max_speed == 20
    assert attrs.forward_accel == 1.0
    assert attrs.turn_accel == 1.0
    assert attrs.cornering_grip == 1.0
    assert attrs.max_stamina == 100
    assert attrs.drain_rate_mult == 1.0
    assert attrs.weight == 500


def test_trait_ranges_has_all_fields():
    attrs = create_default_attributes()
    for field in [
        "cruise_speed", "max_speed", "forward_accel", "turn_accel",
        "cornering_grip", "max_stamina", "drain_rate_mult", "weight",
    ]:
        assert field in TRAIT_RANGES
        lo, hi = TRAIT_RANGES[field]
        assert lo < hi
        val = getattr(attrs, field)
        assert lo <= val <= hi


def test_force_caps():
    assert F_T_MAX == 5
    assert F_N_MAX == 3


def test_core_attributes_is_dataclass():
    attrs = create_default_attributes()
    import dataclasses
    modified = dataclasses.replace(attrs, cruise_speed=15)
    assert modified.cruise_speed == 15
    assert attrs.cruise_speed == 13
