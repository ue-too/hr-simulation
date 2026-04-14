"""Scripted opponent strategies for single-agent training."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.types import Horse, InputState


class Strategy(ABC):
    @abstractmethod
    def act(self, progress: float) -> int:
        """Return a discrete action index [0-53] based on current progress."""
        ...

    def act_continuous(self, horse: Horse) -> InputState | None:
        """Return continuous (tangential, normal) input, or None to use act()."""
        return None


def _jitter_action(base_index: int) -> int:
    """Add ±1 tangential level jitter to an action index.

    Keeps the same normal component but shifts tangential by -1, 0, or +1
    level with equal probability. Clamps to valid range.
    """
    from ..action import NUM_NORMAL, NUM_TANGENTIAL

    ti = base_index // NUM_NORMAL
    ni = base_index % NUM_NORMAL
    ti = max(0, min(NUM_TANGENTIAL - 1, ti + random.choice([-1, 0, 1])))
    return ti * NUM_NORMAL + ni


class CruiseStrategy(Strategy):
    """Always cruise: action (0, 0) = index 13."""

    def act(self, progress: float) -> int:
        return 13


class PushEarlyStrategy(Strategy):
    """Push hard early, then cruise. Switch point randomized [0.2, 0.4]."""

    def __init__(self, switch: float | None = None):
        self._switch = switch if switch is not None else random.uniform(0.2, 0.4)

    def act(self, progress: float) -> int:
        return _jitter_action(49) if progress < self._switch else _jitter_action(13)


class PushLateStrategy(Strategy):
    """Cruise then push hard late. Switch point randomized [0.6, 0.8]."""

    def __init__(self, switch: float | None = None):
        self._switch = switch if switch is not None else random.uniform(0.6, 0.8)

    def act(self, progress: float) -> int:
        return _jitter_action(49) if progress >= self._switch else _jitter_action(13)


class FullPushStrategy(Strategy):
    """Push +1.0 the entire race — fast but exhausts mid-race."""

    def act(self, progress: float) -> int:
        return _jitter_action(49)


class SteadyPushStrategy(Strategy):
    """Push +0.5 the entire race — faster than cruise, moderate drain."""

    def act(self, progress: float) -> int:
        return _jitter_action(31)


class SteadyThenSprintStrategy(Strategy):
    """Push +0.5 then sprint +1.0 in the final stretch."""

    def __init__(self, switch_progress: float | None = None):
        self._switch = switch_progress if switch_progress is not None else random.uniform(0.4, 0.8)

    def act(self, progress: float) -> int:
        return _jitter_action(49) if progress >= self._switch else _jitter_action(31)


class EarlySprint50Strategy(SteadyThenSprintStrategy):
    """Steady then sprint at 50% — aggressive."""

    def __init__(self):
        super().__init__(switch_progress=0.5)


class LateSprint80Strategy(SteadyThenSprintStrategy):
    """Steady then sprint at 80% — conservative."""

    def __init__(self):
        super().__init__(switch_progress=0.8)


class LaneHolderStrategy(Strategy):
    """Combines a pacing strategy with lane-holding behavior.

    Uses proportional control on lateral offset to hold a target lane
    position. Creates a physical obstacle the agent must overtake.
    """

    _LANE_K = 0.15  # proportional gain for lane correction

    def __init__(self, pacing: Strategy, target_offset: float):
        self._pacing = pacing
        self._target_offset = target_offset

    def act(self, progress: float) -> int:
        return self._pacing.act(progress)

    def act_continuous(self, horse: Horse) -> InputState:
        from ..action import decode_action
        from ..core.types import InputState

        tang_action = self._pacing.act(horse.track_progress)
        tang, _ = decode_action(tang_action)

        current_offset = horse.navigator.lateral_offset(horse.pos)
        error = self._target_offset - current_offset
        normal = max(-1.0, min(1.0, error * self._LANE_K))

        return InputState(tang, normal)


class InsideLaneStrategy(LaneHolderStrategy):
    """Smart pacer holding inside lane (offset -5.0m)."""

    def __init__(self):
        super().__init__(SteadyThenSprintStrategy(), target_offset=-5.0)


class OutsideLaneStrategy(LaneHolderStrategy):
    """Smart pacer holding outside lane (offset +5.0m)."""

    def __init__(self):
        super().__init__(SteadyThenSprintStrategy(), target_offset=5.0)


class CenterLaneStrategy(LaneHolderStrategy):
    """Smart pacer holding center lane (offset 0.0m)."""

    def __init__(self):
        super().__init__(SteadyThenSprintStrategy(), target_offset=0.0)


_STRATEGIES = [
    CruiseStrategy,
    PushEarlyStrategy,
    PushLateStrategy,
    FullPushStrategy,
    SteadyPushStrategy,
    SteadyThenSprintStrategy,
    EarlySprint50Strategy,
    LateSprint80Strategy,
    InsideLaneStrategy,
    OutsideLaneStrategy,
    CenterLaneStrategy,
]


def random_strategy() -> Strategy:
    """Return a randomly chosen strategy."""
    return random.choice(_STRATEGIES)()
