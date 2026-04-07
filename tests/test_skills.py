"""Tests for composable jockey skill reward functions."""
from __future__ import annotations

import pytest

from horse_racing.skills import (
    ARCHETYPE_SKILL_PRESETS,
    _SKILL_FN,
    compute_skill_bonus,
)
from horse_racing.types import SKILL_IDS


def _make_obs(
    *,
    tangential_vel: float = 15.0,
    normal_vel: float = 0.0,
    displacement: float = 0.0,
    track_progress: float = 0.5,
    curvature: float = 0.0,
    stamina_ratio: float = 0.8,
    effective_cruise_speed: float = 14.25,
    effective_max_speed: float = 19.0,
    active_modifiers: set | None = None,
    relatives: list | None = None,
    finished: bool = False,
    collision: bool = False,
    _placement: int = 2,
) -> dict:
    if relatives is None:
        relatives = [(5.0, 1.0, 0.5, 0.0)] + [(0.0, 0.0, 0.0, 0.0)] * 18
    return {
        "tangential_vel": tangential_vel,
        "normal_vel": normal_vel,
        "displacement": displacement,
        "track_progress": track_progress,
        "curvature": curvature,
        "stamina_ratio": stamina_ratio,
        "effective_cruise_speed": effective_cruise_speed,
        "effective_max_speed": effective_max_speed,
        "active_modifiers": active_modifiers or set(),
        "relatives": relatives,
        "finished": finished,
        "collision": collision,
        "_placement": _placement,
    }


class TestSkillRegistry:
    def test_all_skills_have_functions(self):
        for sid in SKILL_IDS:
            assert sid in _SKILL_FN, f"Missing function for skill {sid}"

    def test_archetype_presets_use_valid_skills(self):
        for arch, skills in ARCHETYPE_SKILL_PRESETS.items():
            for sid in skills:
                assert sid in SKILL_IDS, f"Archetype {arch} uses unknown skill {sid}"


class TestStaminaManagement:
    def test_above_baseline_gives_bonus(self):
        obs = _make_obs(track_progress=0.5, stamina_ratio=0.7)
        prev = _make_obs(track_progress=0.45, stamina_ratio=0.75)
        bonus = _SKILL_FN["stamina_management"](obs, prev, 2, 4)
        assert bonus > 0, "Should reward being above linear baseline"

    def test_below_baseline_gives_penalty(self):
        obs = _make_obs(track_progress=0.5, stamina_ratio=0.2)
        prev = _make_obs(track_progress=0.45, stamina_ratio=0.25)
        bonus = _SKILL_FN["stamina_management"](obs, prev, 2, 4)
        assert bonus < 0, "Should penalize overspending"

    def test_finish_reserves_bonus(self):
        obs = _make_obs(track_progress=0.9, stamina_ratio=0.3)
        prev = _make_obs(track_progress=0.88, stamina_ratio=0.32)
        bonus = _SKILL_FN["stamina_management"](obs, prev, 2, 4)
        assert bonus > 0, "Should reward having reserves near finish"


class TestSprintTiming:
    def test_early_sprint_penalty(self):
        obs = _make_obs(track_progress=0.3, tangential_vel=18.0)
        prev = _make_obs(track_progress=0.28, tangential_vel=17.0)
        bonus = _SKILL_FN["sprint_timing"](obs, prev, 2, 4)
        assert bonus < 0, "Should penalize sprinting before 60%"

    def test_late_sprint_reward(self):
        obs = _make_obs(track_progress=0.85, tangential_vel=18.0, stamina_ratio=0.5)
        prev = _make_obs(track_progress=0.83, tangential_vel=17.0)
        bonus = _SKILL_FN["sprint_timing"](obs, prev, 2, 4)
        assert bonus > 0, "Should reward sprinting after 75%"


class TestOvertake:
    def test_gaining_position_gives_bonus(self):
        obs = _make_obs(_placement=2)
        prev = _make_obs(_placement=3)
        bonus = _SKILL_FN["overtake"](obs, prev, 2, 4)
        assert bonus > 0, "Should reward gaining positions"

    def test_losing_position_gives_penalty(self):
        obs = _make_obs(_placement=3)
        prev = _make_obs(_placement=2)
        bonus = _SKILL_FN["overtake"](obs, prev, 3, 4)
        assert bonus < 0, "Should penalize losing positions"


class TestDraftingExploit:
    def test_drafting_early_gives_bonus(self):
        obs = _make_obs(track_progress=0.3, active_modifiers={"drafting"})
        prev = _make_obs(track_progress=0.28)
        bonus = _SKILL_FN["drafting_exploit"](obs, prev, 2, 4)
        assert bonus > 0, "Should reward drafting early"

    def test_high_speed_late_gives_bonus(self):
        obs = _make_obs(track_progress=0.85, tangential_vel=18.0)
        prev = _make_obs(track_progress=0.83)
        bonus = _SKILL_FN["drafting_exploit"](obs, prev, 2, 4)
        assert bonus > 0, "Should reward high speed late"


class TestCorneringLine:
    def test_no_bonus_on_straights(self):
        obs = _make_obs(curvature=0.0, displacement=-3.0)
        prev = _make_obs(curvature=0.0)
        bonus = _SKILL_FN["cornering_line"](obs, prev, 2, 4)
        assert bonus == 0.0, "Should return 0 on straights"

    def test_inside_line_gives_bonus(self):
        obs = _make_obs(curvature=0.05, displacement=-3.0, tangential_vel=15.0)
        prev = _make_obs(curvature=0.05)
        bonus = _SKILL_FN["cornering_line"](obs, prev, 2, 4)
        assert bonus > 0, "Should reward inside lines on curves"

    def test_outside_line_gives_penalty(self):
        obs = _make_obs(curvature=0.05, displacement=3.0, tangential_vel=10.0)
        prev = _make_obs(curvature=0.05)
        bonus = _SKILL_FN["cornering_line"](obs, prev, 2, 4)
        assert bonus < 0, "Should penalize wide lines on curves"

    def test_clamped_output(self):
        obs = _make_obs(curvature=1.0, displacement=-10.0, tangential_vel=20.0)
        prev = _make_obs(curvature=1.0)
        bonus = _SKILL_FN["cornering_line"](obs, prev, 2, 4)
        assert -0.15 <= bonus <= 0.3, f"Bonus {bonus} outside clamped range"


class TestPacePressure:
    def test_above_cruise_gives_bonus(self):
        obs = _make_obs(tangential_vel=17.0, stamina_ratio=0.6)
        prev = _make_obs(tangential_vel=16.5)
        bonus = _SKILL_FN["pace_pressure"](obs, prev, 2, 4)
        assert bonus > 0, "Should reward pushing above cruise"

    def test_below_cruise_gives_penalty(self):
        obs = _make_obs(tangential_vel=12.0)
        prev = _make_obs(tangential_vel=12.0)
        bonus = _SKILL_FN["pace_pressure"](obs, prev, 2, 4)
        assert bonus < 0, "Should penalize coasting below cruise"


class TestComposability:
    def test_multiple_skills_sum(self):
        obs = _make_obs(track_progress=0.85, tangential_vel=18.0, stamina_ratio=0.5)
        prev = _make_obs(track_progress=0.83, tangential_vel=17.0)
        skills = {"sprint_timing", "pace_pressure"}
        combined = compute_skill_bonus(skills, obs, prev, 2, 4)
        individual_sum = sum(
            _SKILL_FN[s](obs, prev, 2, 4) for s in skills
        )
        assert abs(combined - individual_sum) < 1e-6

    def test_all_skills_combined_reasonable(self):
        obs = _make_obs(
            track_progress=0.85, tangential_vel=18.0, stamina_ratio=0.5,
            curvature=0.05, displacement=-2.0,
            active_modifiers={"drafting"},
        )
        prev = _make_obs(
            track_progress=0.83, tangential_vel=17.0,
            _placement=3,
        )
        all_skills = set(SKILL_IDS)
        bonus = compute_skill_bonus(all_skills, obs, prev, 2, 4)
        assert abs(bonus) < 2.0, f"All skills combined {bonus} seems too large"

    def test_empty_skills_returns_zero(self):
        obs = _make_obs()
        prev = _make_obs()
        assert compute_skill_bonus(set(), obs, prev, 2, 4) == 0.0

    def test_unknown_skill_ignored(self):
        obs = _make_obs()
        prev = _make_obs()
        bonus = compute_skill_bonus({"nonexistent_skill"}, obs, prev, 2, 4)
        assert bonus == 0.0


class TestSkillRewardScale:
    def test_scaled_bonus_meaningful(self):
        """Skill bonus with 10x scale should produce >1.0 reward difference."""
        from horse_racing.reward import compute_reward

        obs_curr = _make_obs(track_progress=0.85, tangential_vel=18.0, stamina_ratio=0.5)
        obs_prev = _make_obs(track_progress=0.83, tangential_vel=17.0)

        reward_no_skills = compute_reward(
            obs_prev, obs_curr, False, active_skills=None,
        )
        reward_with_skills = compute_reward(
            obs_prev, obs_curr, False,
            active_skills={"sprint_timing", "pace_pressure"},
            skill_reward_scale=10.0,
        )
        diff = reward_with_skills - reward_no_skills
        assert diff > 1.0, f"Scaled skill diff {diff} too small to influence learning"

    def test_scale_1_matches_unscaled(self):
        """Scale of 1.0 should match the raw bonus * tick_scale."""
        obs_curr = _make_obs(track_progress=0.85, tangential_vel=18.0, stamina_ratio=0.5)
        obs_prev = _make_obs(track_progress=0.83, tangential_vel=17.0)
        skills = {"sprint_timing", "pace_pressure"}

        raw_bonus = compute_skill_bonus(skills, obs_curr, obs_prev, 2, 4)

        from horse_racing.reward import compute_reward, REF_TICKS
        delta_progress = 0.85 - 0.83
        tick_scale = delta_progress * REF_TICKS
        reward_no = compute_reward(obs_prev, obs_curr, False, active_skills=None)
        reward_s1 = compute_reward(
            obs_prev, obs_curr, False, active_skills=skills, skill_reward_scale=1.0,
        )
        assert abs((reward_s1 - reward_no) - raw_bonus * tick_scale) < 1e-6
