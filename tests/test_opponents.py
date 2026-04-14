from horse_racing.opponents.scripted import (
    CenterLaneStrategy,
    CruiseStrategy,
    EarlySprint50Strategy,
    InsideLaneStrategy,
    LateSprint80Strategy,
    OutsideLaneStrategy,
    PushEarlyStrategy,
    PushLateStrategy,
    random_strategy,
)


class TestCruiseStrategy:
    def test_always_returns_cruise_action(self):
        s = CruiseStrategy()
        assert s.act(0.0) == 13
        assert s.act(0.5) == 13
        assert s.act(1.0) == 13


class TestPushEarlyStrategy:
    def test_pushes_before_30_percent(self):
        s = PushEarlyStrategy()
        assert s.act(0.0) == 49
        assert s.act(0.29) == 49

    def test_cruises_after_30_percent(self):
        s = PushEarlyStrategy()
        assert s.act(0.3) == 13
        assert s.act(0.8) == 13


class TestPushLateStrategy:
    def test_cruises_before_70_percent(self):
        s = PushLateStrategy()
        assert s.act(0.0) == 13
        assert s.act(0.69) == 13

    def test_pushes_after_70_percent(self):
        s = PushLateStrategy()
        assert s.act(0.7) == 49
        assert s.act(0.9) == 49


class TestEarlySprint50Strategy:
    def test_pushes_half_before_50(self):
        s = EarlySprint50Strategy()
        assert s.act(0.0) == 31
        assert s.act(0.49) == 31

    def test_sprints_after_50(self):
        s = EarlySprint50Strategy()
        assert s.act(0.5) == 49
        assert s.act(0.9) == 49


class TestLateSprint80Strategy:
    def test_pushes_half_before_80(self):
        s = LateSprint80Strategy()
        assert s.act(0.0) == 31
        assert s.act(0.79) == 31

    def test_sprints_after_80(self):
        s = LateSprint80Strategy()
        assert s.act(0.8) == 49
        assert s.act(0.9) == 49


class TestLaneHolderStrategies:
    def test_inside_lane_act_returns_pacing_action(self):
        s = InsideLaneStrategy()
        assert s.act(0.0) == 31
        assert s.act(0.5) == 49

    def test_outside_lane_act_returns_pacing_action(self):
        s = OutsideLaneStrategy()
        assert s.act(0.0) == 31
        assert s.act(0.5) == 49

    def test_center_lane_act_returns_pacing_action(self):
        s = CenterLaneStrategy()
        assert s.act(0.0) == 31
        assert s.act(0.5) == 49


class TestRandomStrategy:
    def test_returns_one_of_ten_strategies(self):
        strategies = set()
        for _ in range(1000):
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
            "InsideLaneStrategy",
            "OutsideLaneStrategy",
            "CenterLaneStrategy",
        }
