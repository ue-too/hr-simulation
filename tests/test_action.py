from horse_racing.action import (
    NORMAL_LEVELS,
    NUM_ACTIONS,
    NUM_NORMAL,
    NUM_TANGENTIAL,
    TANGENTIAL_LEVELS,
    decode_action,
)


def test_tangential_levels():
    assert TANGENTIAL_LEVELS == [-0.25, 0, 0.25, 0.5, 0.75, 1]


def test_normal_levels():
    assert NORMAL_LEVELS == [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]


def test_axis_counts():
    assert NUM_TANGENTIAL == 6
    assert NUM_NORMAL == 9


def test_num_actions():
    assert NUM_ACTIONS == 54


def test_decode_index_0():
    t, n = decode_action(0)
    assert t == -0.25
    assert n == -1


def test_decode_cruise():
    """Cruise = tangential 0, normal 0 → index 1*9+4 = 13."""
    t, n = decode_action(13)
    assert t == 0
    assert n == 0


def test_decode_last():
    t, n = decode_action(53)
    assert t == 1
    assert n == 1


def test_decode_075_push():
    """0.75 tangential, 0 normal → index 4*9+4 = 40."""
    t, n = decode_action(40)
    assert t == 0.75
    assert n == 0


def test_all_actions_valid():
    for i in range(NUM_ACTIONS):
        t, n = decode_action(i)
        assert t in TANGENTIAL_LEVELS
        assert n in NORMAL_LEVELS
