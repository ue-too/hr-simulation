from horse_racing.action import NUM_ACTIONS
from horse_racing.opponents.scripted import (
    CenterLaneStrategy,
    CruiseStrategy,
    EarlySprint50Strategy,
    FullPushStrategy,
    InsideLaneStrategy,
    LateSprint80Strategy,
    OutsideLaneStrategy,
    PushEarlyStrategy,
    PushLateStrategy,
    SteadyPushStrategy,
    SteadyThenSprintStrategy,
    random_strategy,
)


def _tangential_index(action: int) -> int:
    """Extract tangential level index from flat action."""
    return action // 9


class TestCruiseStrategy:
    def test_always_returns_cruise_region(self):
        s = CruiseStrategy()
        # Cruise base is t=0 (index 1), no jitter on cruise
        assert s.act(0.0) == 13
        assert s.act(0.5) == 13
        assert s.act(1.0) == 13


class TestPushEarlyStrategy:
    def test_pushes_before_switch(self):
        s = PushEarlyStrategy(switch=0.3)
        actions = [s.act(0.0) for _ in range(50)]
        # All should be high tangential (push region, t=4 or 5 with jitter)
        assert all(_tangential_index(a) >= 3 for a in actions)

    def test_cruises_after_switch(self):
        s = PushEarlyStrategy(switch=0.3)
        actions = [s.act(0.5) for _ in range(50)]
        # All should be low tangential (cruise region, t=0 or 1 with jitter)
        assert all(_tangential_index(a) <= 2 for a in actions)

    def test_random_switch_in_range(self):
        switches = set()
        for _ in range(100):
            s = PushEarlyStrategy()
            switches.add(s._switch)
        assert all(0.2 <= sw <= 0.4 for sw in switches)
        assert len(switches) > 5  # not all the same


class TestPushLateStrategy:
    def test_cruises_before_switch(self):
        s = PushLateStrategy(switch=0.7)
        actions = [s.act(0.5) for _ in range(50)]
        assert all(_tangential_index(a) <= 2 for a in actions)

    def test_pushes_after_switch(self):
        s = PushLateStrategy(switch=0.7)
        actions = [s.act(0.8) for _ in range(50)]
        assert all(_tangential_index(a) >= 3 for a in actions)

    def test_random_switch_in_range(self):
        switches = set()
        for _ in range(100):
            s = PushLateStrategy()
            switches.add(s._switch)
        assert all(0.6 <= sw <= 0.8 for sw in switches)


class TestFullPushStrategy:
    def test_always_high_tangential(self):
        s = FullPushStrategy()
        actions = [s.act(p / 10) for p in range(11) for _ in range(5)]
        assert all(_tangential_index(a) >= 3 for a in actions)


class TestSteadyPushStrategy:
    def test_always_medium_tangential(self):
        s = SteadyPushStrategy()
        actions = [s.act(p / 10) for p in range(11) for _ in range(5)]
        # Steady base is t=3 (index 3), jitter gives 2-4
        assert all(1 <= _tangential_index(a) <= 4 for a in actions)


class TestSteadyThenSprintStrategy:
    def test_steady_before_switch(self):
        s = SteadyThenSprintStrategy(switch_progress=0.6)
        actions = [s.act(0.3) for _ in range(50)]
        assert all(1 <= _tangential_index(a) <= 4 for a in actions)

    def test_sprints_after_switch(self):
        s = SteadyThenSprintStrategy(switch_progress=0.6)
        actions = [s.act(0.8) for _ in range(50)]
        assert all(_tangential_index(a) >= 3 for a in actions)

    def test_random_switch_in_range(self):
        switches = set()
        for _ in range(100):
            s = SteadyThenSprintStrategy()
            switches.add(s._switch)
        assert all(0.4 <= sw <= 0.8 for sw in switches)


class TestEarlySprint50Strategy:
    def test_fixed_switch_at_50(self):
        s = EarlySprint50Strategy()
        assert s._switch == 0.5


class TestLateSprint80Strategy:
    def test_fixed_switch_at_80(self):
        s = LateSprint80Strategy()
        assert s._switch == 0.8


class TestLaneHolderStrategies:
    def test_inside_lane_act_returns_pacing_action(self):
        s = InsideLaneStrategy()
        # Before switch: steady region; after switch: push region
        early_actions = [s.act(0.0) for _ in range(20)]
        late_actions = [s.act(0.9) for _ in range(20)]
        assert all(1 <= _tangential_index(a) <= 4 for a in early_actions)
        assert all(_tangential_index(a) >= 3 for a in late_actions)

    def test_outside_lane_act_returns_pacing_action(self):
        s = OutsideLaneStrategy()
        early_actions = [s.act(0.0) for _ in range(20)]
        late_actions = [s.act(0.9) for _ in range(20)]
        assert all(1 <= _tangential_index(a) <= 4 for a in early_actions)
        assert all(_tangential_index(a) >= 3 for a in late_actions)

    def test_center_lane_act_returns_pacing_action(self):
        s = CenterLaneStrategy()
        early_actions = [s.act(0.0) for _ in range(20)]
        late_actions = [s.act(0.9) for _ in range(20)]
        assert all(1 <= _tangential_index(a) <= 4 for a in early_actions)
        assert all(_tangential_index(a) >= 3 for a in late_actions)


class TestRandomStrategy:
    def test_returns_one_of_eleven_strategies(self):
        strategies = set()
        for _ in range(2000):
            s = random_strategy()
            strategies.add(type(s).__name__)
        assert strategies == {
            "CruiseStrategy",
            "PushEarlyStrategy",
            "PushLateStrategy",
            "FullPushStrategy",
            "SteadyPushStrategy",
            "SteadyThenSprintStrategy",
            "EarlySprint50Strategy",
            "LateSprint80Strategy",
            "InsideLaneStrategy",
            "OutsideLaneStrategy",
            "CenterLaneStrategy",
        }

    def test_all_actions_in_valid_range(self):
        for _ in range(500):
            s = random_strategy()
            action = s.act(random.random())
            assert 0 <= action < NUM_ACTIONS


import random
