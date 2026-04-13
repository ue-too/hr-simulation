from horse_racing.action import (
    NORMAL_LEVELS,
    NUM_ACTIONS,
    NUM_NORMAL,
    NUM_TANGENTIAL,
    TANGENTIAL_LEVELS,
    decode_action,
)


def test_tangential_levels():
    assert TANGENTIAL_LEVELS == [-1, -0.5, 0, 0.25, 0.5, 0.75, 1]


def test_normal_levels():
    assert NORMAL_LEVELS == [-1, -0.5, 0, 0.5, 1]


def test_axis_counts():
    assert NUM_TANGENTIAL == 7
    assert NUM_NORMAL == 5


def test_num_actions():
    assert NUM_ACTIONS == 35


def test_decode_index_0():
    t, n = decode_action(0)
    assert t == -1
    assert n == -1


def test_decode_cruise():
    """Cruise = tangential 0, normal 0 → index 2*5+2 = 12."""
    t, n = decode_action(12)
    assert t == 0
    assert n == 0


def test_decode_last():
    t, n = decode_action(34)
    assert t == 1
    assert n == 1


def test_decode_075_push():
    """0.75 tangential, 0 normal → index 5*5+2 = 27."""
    t, n = decode_action(27)
    assert t == 0.75
    assert n == 0


def test_all_actions_valid():
    for i in range(NUM_ACTIONS):
        t, n = decode_action(i)
        assert t in TANGENTIAL_LEVELS
        assert n in NORMAL_LEVELS
