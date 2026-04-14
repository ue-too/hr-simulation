"""Discrete 6×9 action space — asymmetric braking (light brake only)."""

TANGENTIAL_LEVELS: list[float] = [-0.25, 0, 0.25, 0.5, 0.75, 1]
NORMAL_LEVELS: list[float] = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
NUM_TANGENTIAL = len(TANGENTIAL_LEVELS)  # 6
NUM_NORMAL = len(NORMAL_LEVELS)          # 9
NUM_ACTIONS = NUM_TANGENTIAL * NUM_NORMAL  # 54


def decode_action(index: int) -> tuple[float, float]:
    """Decode a flat action index (0-80) into (tangential, normal) pair.

    Layout: index = tangential_level * NUM_NORMAL + normal_level
    """
    ti = index // NUM_NORMAL
    ni = index % NUM_NORMAL
    return TANGENTIAL_LEVELS[ti], NORMAL_LEVELS[ni]
