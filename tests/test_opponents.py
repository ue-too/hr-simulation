from horse_racing.opponents.scripted import (
    CruiseStrategy,
    PushEarlyStrategy,
    PushLateStrategy,
    random_strategy,
)


class TestCruiseStrategy:
    def test_always_returns_cruise_action(self):
        s = CruiseStrategy()
        assert s.act(0.0) == 12
        assert s.act(0.5) == 12
        assert s.act(1.0) == 12


class TestPushEarlyStrategy:
    def test_pushes_before_30_percent(self):
        s = PushEarlyStrategy()
        assert s.act(0.0) == 22  # (1, 0)
        assert s.act(0.29) == 22

    def test_cruises_after_30_percent(self):
        s = PushEarlyStrategy()
        assert s.act(0.3) == 12  # (0, 0)
        assert s.act(0.8) == 12


class TestPushLateStrategy:
    def test_cruises_before_70_percent(self):
        s = PushLateStrategy()
        assert s.act(0.0) == 12
        assert s.act(0.69) == 12

    def test_pushes_after_70_percent(self):
        s = PushLateStrategy()
        assert s.act(0.7) == 22  # (1, 0)
        assert s.act(0.9) == 22


class TestRandomStrategy:
    def test_returns_one_of_three_strategies(self):
        strategies = set()
        for _ in range(100):
            s = random_strategy()
            strategies.add(type(s).__name__)
        assert strategies == {"CruiseStrategy", "PushEarlyStrategy", "PushLateStrategy"}
