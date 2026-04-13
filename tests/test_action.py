from horse_racing.action import (
    ACTION_LEVELS,
    LEVELS_PER_AXIS,
    NUM_ACTIONS,
    decode_action,
)


def test_action_levels():
    assert ACTION_LEVELS == [-1, -0.5, 0, 0.5, 1]


def test_levels_per_axis():
    assert LEVELS_PER_AXIS == 5


def test_num_actions():
    assert NUM_ACTIONS == 25


def test_decode_index_0():
    t, n = decode_action(0)
    assert t == -1
    assert n == -1


def test_decode_index_12():
    t, n = decode_action(12)
    assert t == 0
    assert n == 0


def test_decode_index_24():
    t, n = decode_action(24)
    assert t == 1
    assert n == 1


def test_decode_index_21():
    t, n = decode_action(21)
    assert t == 1
    assert n == -0.5


def test_all_25_actions_valid():
    for i in range(NUM_ACTIONS):
        t, n = decode_action(i)
        assert t in ACTION_LEVELS
        assert n in ACTION_LEVELS
