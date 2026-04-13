"""Scripted opponent strategies for single-agent training."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod


class Strategy(ABC):
    @abstractmethod
    def act(self, progress: float) -> int:
        """Return a discrete action index [0-24] based on current progress."""
        ...


class CruiseStrategy(Strategy):
    """Always cruise: action (0, 0) = index 12."""

    def act(self, progress: float) -> int:
        return 12


class PushEarlyStrategy(Strategy):
    """Push hard for the first 30%, then cruise."""

    def act(self, progress: float) -> int:
        return 22 if progress < 0.3 else 12  # (1, 0) or (0, 0)


class PushLateStrategy(Strategy):
    """Cruise until 70%, then push hard."""

    def act(self, progress: float) -> int:
        return 22 if progress >= 0.7 else 12  # (1, 0) or (0, 0)


class FullPushStrategy(Strategy):
    """Push +1.0 the entire race — fast but exhausts mid-race."""

    def act(self, progress: float) -> int:
        return 22  # (1, 0)


class SteadyPushStrategy(Strategy):
    """Push +0.5 the entire race — faster than cruise, moderate drain."""

    def act(self, progress: float) -> int:
        return 17  # (0.5, 0)


class SteadyThenSprintStrategy(Strategy):
    """Push +0.5 then sprint +1.0 in the final stretch. ~1935-1983 ticks."""

    def __init__(self, switch_progress: float = 0.6):
        self._switch = switch_progress

    def act(self, progress: float) -> int:
        return 22 if progress >= self._switch else 17  # (1.0, 0) or (0.5, 0)


class EarlySprint50Strategy(SteadyThenSprintStrategy):
    """Steady then sprint at 50% — aggressive, finishes ~1935 ticks."""

    def __init__(self):
        super().__init__(switch_progress=0.5)


class LateSprint80Strategy(SteadyThenSprintStrategy):
    """Steady then sprint at 80% — conservative, finishes ~1983 ticks."""

    def __init__(self):
        super().__init__(switch_progress=0.8)


_STRATEGIES = [
    CruiseStrategy,
    PushEarlyStrategy,
    PushLateStrategy,
    FullPushStrategy,
    SteadyPushStrategy,
    EarlySprint50Strategy,
    LateSprint80Strategy,
]


def random_strategy() -> Strategy:
    """Return a randomly chosen strategy."""
    return random.choice(_STRATEGIES)()
