"""Discrete 5×5 action space — mirrors TS onnx-jockey.ts."""

ACTION_LEVELS: list[float] = [-1, -0.5, 0, 0.5, 1]
LEVELS_PER_AXIS = len(ACTION_LEVELS)  # 5
NUM_ACTIONS = LEVELS_PER_AXIS * LEVELS_PER_AXIS  # 25


def decode_action(index: int) -> tuple[float, float]:
    """Decode a flat action index (0-24) into (tangential, normal) pair.

    Layout: index = tangential_level * LEVELS_PER_AXIS + normal_level
    """
    ti = index // LEVELS_PER_AXIS
    ni = index % LEVELS_PER_AXIS
    return ACTION_LEVELS[ti], ACTION_LEVELS[ni]
