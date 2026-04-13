from horse_racing.opponents.scripted import (
    CruiseStrategy,
    EarlySprint50Strategy,
    LateSprint80Strategy,
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


class TestEarlySprint50Strategy:
    def test_pushes_half_before_50(self):
        s = EarlySprint50Strategy()
        assert s.act(0.0) == 17  # (0.5, 0)
        assert s.act(0.49) == 17

    def test_sprints_after_50(self):
        s = EarlySprint50Strategy()
        assert s.act(0.5) == 22  # (1.0, 0)
        assert s.act(0.9) == 22


class TestLateSprint80Strategy:
    def test_pushes_half_before_80(self):
        s = LateSprint80Strategy()
        assert s.act(0.0) == 17  # (0.5, 0)
        assert s.act(0.79) == 17

    def test_sprints_after_80(self):
        s = LateSprint80Strategy()
        assert s.act(0.8) == 22  # (1.0, 0)
        assert s.act(0.9) == 22


class TestRandomStrategy:
    def test_returns_one_of_seven_strategies(self):
        strategies = set()
        for _ in range(500):
            s = random_strategy()
            strategies.add(type(s).__name__)
        assert strategies == {
            "CruiseStrategy",
            "PushEarlyStrategy",
            "PushLateStrategy",
            "FullPushStrategy",
            "SteadyPushStrategy",
            "EarlySprint50Strategy",
            "LateSprint80Strategy",
        }
