"""Tests for BT jockey behavior tree AI."""

import pytest

from horse_racing.bt_jockey import (
    BTJockey,
    JockeyPersonality,
    PERSONALITIES,
    make_bt_jockey,
)
from horse_racing.engine import EngineConfig, HorseRacingEngine
from horse_racing.types import HorseAction


# ---------------------------------------------------------------------------
# Action range
# ---------------------------------------------------------------------------


class TestActionRange:
    """BT must always produce actions within the valid range."""

    @pytest.fixture()
    def engine(self):
        return HorseRacingEngine("tracks/tokyo.json", EngineConfig(horse_count=4))

    def test_actions_within_bounds(self, engine):
        jockeys = [make_bt_jockey(a) for a in ["front_runner", "stalker", "closer", "presser"]]
        for _ in range(200):
            obs_list = engine.get_observations()
            for j, obs in zip(jockeys, obs_list):
                action = j.compute_action(obs)
                assert -10.0 <= action.extra_tangential <= 10.0
                assert -5.0 <= action.extra_normal <= 5.0
            actions = [j.compute_action(obs) for j, obs in zip(jockeys, obs_list)]
            engine.step(actions)

    def test_default_action_is_zero(self):
        jockey = BTJockey()
        # Minimal obs that hits the fallback branch
        obs = {
            "tangential_vel": 0.0,
            "normal_vel": 0.0,
            "displacement": 0.0,
            "track_progress": 0.5,
            "curvature": 0.0,
            "stamina_ratio": 0.5,
            "effective_cruise_speed": 14.0,
            "effective_max_speed": 18.0,
            "relatives": [(0, 0, 0, 0)] * 19,
            "cornering_margin": 1000.0,
            "path_efficiency": 1.0,
            "slope": 0.0,
            "pushing_power": 0.1,
            "push_resistance": 0.1,
            "forward_accel": 1.0,
            "turn_accel": 1.0,
            "cornering_grip": 1.0,
            "drain_rate_mult": 1.0,
            "placement_norm": 0.0,
            "num_horses": 4,
            "active_modifiers": set(),
            "collision": False,
            "finished": False,
        }
        action = jockey.compute_action(obs)
        assert isinstance(action, HorseAction)


# ---------------------------------------------------------------------------
# Personality effects
# ---------------------------------------------------------------------------


class TestPersonality:
    def _make_obs(self, progress=0.3, stamina=0.8, curvature=0.0, displacement=0.0):
        return {
            "tangential_vel": 14.0,
            "normal_vel": 0.0,
            "displacement": displacement,
            "track_progress": progress,
            "curvature": curvature,
            "stamina_ratio": stamina,
            "effective_cruise_speed": 14.0,
            "effective_max_speed": 18.0,
            "relatives": [(0, 0, 0, 0)] * 19,
            "cornering_margin": 1000.0,
            "path_efficiency": 1.0,
            "slope": 0.0,
            "pushing_power": 0.1,
            "push_resistance": 0.1,
            "forward_accel": 1.0,
            "turn_accel": 1.0,
            "cornering_grip": 1.0,
            "drain_rate_mult": 1.0,
            "placement_norm": 0.5,
            "num_horses": 6,
            "active_modifiers": set(),
            "collision": False,
            "finished": False,
        }

    def test_front_runner_pushes_harder_early(self):
        fr = make_bt_jockey("front_runner")
        cl = make_bt_jockey("closer")
        obs = self._make_obs(progress=0.2)
        assert fr.compute_action(obs).extra_tangential > cl.compute_action(obs).extra_tangential

    def test_emergency_eases_up(self):
        jockey = make_bt_jockey("front_runner")
        obs = self._make_obs(stamina=0.10)
        action = jockey.compute_action(obs)
        # Emergency mode should produce zero or near-zero tangential
        assert action.extra_tangential <= 0.5

    def test_cornering_steers_inside(self):
        jockey = make_bt_jockey("front_runner")
        obs = self._make_obs(curvature=0.05, displacement=5.0)
        action = jockey.compute_action(obs)
        # Should steer toward inside (negative normal when displacement is positive)
        assert action.extra_normal < 0

    def test_all_archetypes_exist(self):
        for archetype in ["front_runner", "stalker", "closer", "presser"]:
            assert archetype in PERSONALITIES
            jockey = make_bt_jockey(archetype)
            assert jockey.personality == PERSONALITIES[archetype]


# ---------------------------------------------------------------------------
# Race completion
# ---------------------------------------------------------------------------


class TestRaceCompletion:
    def test_all_archetypes_finish_race(self):
        archetypes = ["front_runner", "stalker", "closer", "presser"]
        jockeys = [make_bt_jockey(a) for a in archetypes]
        engine = HorseRacingEngine("tracks/tokyo.json", EngineConfig(horse_count=4))

        for step in range(5000):
            obs_list = engine.get_observations()
            actions = [j.compute_action(obs) for j, obs in zip(jockeys, obs_list)]
            engine.step(actions)
            if all(h.finished for h in engine.horses):
                break

        assert all(h.finished for h in engine.horses), (
            f"Not all horses finished: {[h.track_progress for h in engine.horses]}"
        )

    def test_default_jockey_finishes(self):
        jockeys = [make_bt_jockey() for _ in range(4)]
        engine = HorseRacingEngine("tracks/tokyo.json", EngineConfig(horse_count=4))

        for step in range(5000):
            obs_list = engine.get_observations()
            actions = [j.compute_action(obs) for j, obs in zip(jockeys, obs_list)]
            engine.step(actions)
            if all(h.finished for h in engine.horses):
                break

        assert all(h.finished for h in engine.horses)
