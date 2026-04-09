"""Behavior-tree AI jockey — rule-based racing agent.

Produces HorseAction(extra_tangential, extra_normal) from observation dicts.
Parameterized by JockeyPersonality for archetype variety.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from horse_racing.types import HorseAction


# ---------------------------------------------------------------------------
# Personality — tuneable parameters that define racing style
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JockeyPersonality:
    """Parameters that shape BT decision-making without changing tree structure."""

    early_effort: float = 0.4       # tangential push in early race (0–1 scale)
    kick_progress: float = 0.75     # progress threshold to start final kick
    stamina_reserve: float = 0.25   # min stamina ratio before kicking
    inside_bias: float = 0.5        # tendency to take inside line on curves (0–1)
    overtake_aggression: float = 0.5  # how hard to push when overtaking (0–1)
    draft_seeking: float = 0.5      # tendency to position for drafting (0–1)


# Archetype presets
PERSONALITIES: dict[str, JockeyPersonality] = {
    "front_runner": JockeyPersonality(
        early_effort=0.8,
        kick_progress=0.60,
        stamina_reserve=0.15,
        inside_bias=0.6,
        overtake_aggression=0.7,
        draft_seeking=0.2,
    ),
    "stalker": JockeyPersonality(
        early_effort=0.35,
        kick_progress=0.72,
        stamina_reserve=0.25,
        inside_bias=0.5,
        overtake_aggression=0.5,
        draft_seeking=0.8,
    ),
    "closer": JockeyPersonality(
        early_effort=0.2,
        kick_progress=0.85,
        stamina_reserve=0.35,
        inside_bias=0.4,
        overtake_aggression=0.6,
        draft_seeking=0.6,
    ),
    "presser": JockeyPersonality(
        early_effort=0.6,
        kick_progress=0.70,
        stamina_reserve=0.20,
        inside_bias=0.5,
        overtake_aggression=0.8,
        draft_seeking=0.3,
    ),
    # Degenerate personalities for RL robustness training
    "full_throttle": JockeyPersonality(
        early_effort=1.0,
        kick_progress=0.0,
        stamina_reserve=0.0,
        inside_bias=0.3,
        overtake_aggression=0.9,
        draft_seeking=0.0,
    ),
    "passive": JockeyPersonality(
        early_effort=0.0,
        kick_progress=1.0,
        stamina_reserve=0.0,
        inside_bias=0.3,
        overtake_aggression=0.0,
        draft_seeking=0.0,
    ),
    "blocker": JockeyPersonality(
        early_effort=0.4,
        kick_progress=0.80,
        stamina_reserve=0.20,
        inside_bias=0.9,
        overtake_aggression=0.2,
        draft_seeking=0.0,
    ),
}

DEFAULT_PERSONALITY = JockeyPersonality()


# ---------------------------------------------------------------------------
# BT node types
# ---------------------------------------------------------------------------


@dataclass
class ActionOutput:
    """Continuous action output from a leaf or blending node."""

    tangential: float = 0.0
    normal: float = 0.0
    weight: float = 1.0


class BTNode:
    """Base class for behavior tree nodes."""

    def tick(self, obs: dict[str, Any], p: JockeyPersonality) -> ActionOutput | None:
        """Evaluate and return an ActionOutput, or None if this branch doesn't apply."""
        raise NotImplementedError


class Selector(BTNode):
    """Tries children in order, returns first non-None result."""

    def __init__(self, children: list[BTNode]):
        self.children = children

    def tick(self, obs: dict[str, Any], p: JockeyPersonality) -> ActionOutput | None:
        for child in self.children:
            result = child.tick(obs, p)
            if result is not None:
                return result
        return None


class Condition(BTNode):
    """Gates a child node behind a boolean check."""

    def __init__(self, predicate, child: BTNode):
        self.predicate = predicate
        self.child = child

    def tick(self, obs: dict[str, Any], p: JockeyPersonality) -> ActionOutput | None:
        if self.predicate(obs, p):
            return self.child.tick(obs, p)
        return None


class Blend(BTNode):
    """Runs all children and blends their outputs by weight."""

    def __init__(self, children: list[BTNode]):
        self.children = children

    def tick(self, obs: dict[str, Any], p: JockeyPersonality) -> ActionOutput | None:
        results = []
        for child in self.children:
            r = child.tick(obs, p)
            if r is not None:
                results.append(r)
        if not results:
            return None
        total_w = sum(r.weight for r in results)
        if total_w < 1e-9:
            return ActionOutput()
        tang = sum(r.tangential * r.weight for r in results) / total_w
        norm = sum(r.normal * r.weight for r in results) / total_w
        return ActionOutput(tangential=tang, normal=norm, weight=total_w)


class Leaf(BTNode):
    """Leaf node wrapping a callable."""

    def __init__(self, fn):
        self.fn = fn

    def tick(self, obs: dict[str, Any], p: JockeyPersonality) -> ActionOutput | None:
        return self.fn(obs, p)


# ---------------------------------------------------------------------------
# Leaf behaviors
# ---------------------------------------------------------------------------


def _emergency_brake(obs: dict, p: JockeyPersonality) -> ActionOutput | None:
    """Ease up when stamina is critically low — let auto-cruise handle speed."""
    # Don't brake (negative tangential slows below cruise which wastes time).
    # Just stop pushing — auto-cruise will maintain cruise speed.
    return ActionOutput(tangential=0.0, normal=0.0, weight=1.0)


def _gate_break(obs: dict, p: JockeyPersonality) -> ActionOutput | None:
    """Sprint off the starting gate."""
    effort = 2.0 + p.early_effort * 3.0  # 2–5 range
    return ActionOutput(tangential=effort, normal=0.0, weight=1.0)


def _pace_control(obs: dict, p: JockeyPersonality) -> ActionOutput | None:
    """Control forward effort based on personality and race phase.

    Auto-cruise already drives toward cruise_speed (~75% top speed).
    BT adds small nudges above/below cruise — even +1 tangential is
    a meaningful push because it's multiplied by forward_accel.
    """
    progress = obs["track_progress"]
    stamina = obs["stamina_ratio"]

    # Base effort: small push above auto-cruise
    # front_runner (0.8) → 2.4, closer (0.2) → 0.6
    base = p.early_effort * 3.0

    # Ramp up slightly as race progresses
    phase_mult = 1.0 + progress * 0.3  # 1.0 at start, 1.3 at end
    effort = base * phase_mult

    # Back off if stamina is draining faster than expected
    expected_stamina = 1.0 - progress * 0.85  # expect ~15% reserve at finish
    if stamina < expected_stamina - 0.1:
        effort *= 0.3  # strong conservation
    elif stamina < expected_stamina:
        effort *= 0.7  # mild conservation

    return ActionOutput(tangential=effort, normal=0.0, weight=0.6)


def _cornering_line(obs: dict, p: JockeyPersonality) -> ActionOutput | None:
    """Steer toward inside line on curves."""
    curvature = obs["curvature"]
    displacement = obs["displacement"]

    if abs(curvature) < 1e-4:
        # On straights, pre-position for upcoming curve
        # Signed curvature: positive = CCW (inside = negative disp),
        # negative = CW (inside = positive disp)
        next_curv = obs.get("next_curvature", 0.0)
        if abs(next_curv) > 1e-6:
            # Target inside: negative displacement for positive curvature, positive for negative
            inside_sign = -1.0 if next_curv > 0 else 1.0
            target_disp = inside_sign * 6.0 * p.inside_bias
            # Proportional control toward target — naturally gives zero at target,
            # steers inward when outside, and gently corrects if overshooting.
            error = target_disp - displacement
            steer = error * 0.3
            return ActionOutput(tangential=0.0, normal=_clamp(steer, -2.0, 2.0), weight=0.3)
        # Otherwise gently drift toward center
        if abs(displacement) > 2.0:
            correction = -displacement * 0.3
            return ActionOutput(tangential=0.0, normal=_clamp(correction, -2.0, 2.0), weight=0.2)
        return None

    # On curves, steer toward inside (negative displacement = inside)
    # Target displacement based on inside_bias
    target_disp = -8.0 * p.inside_bias  # -8 at max bias, 0 at min
    error = target_disp - displacement
    steer = error * 0.4  # proportional control
    return ActionOutput(tangential=0.0, normal=_clamp(steer, -4.0, 4.0), weight=0.4)


def _drafting(obs: dict, p: JockeyPersonality) -> ActionOutput | None:
    """Position behind horse ahead to exploit drafting."""
    relatives = obs["relatives"]

    # Find nearest horse ahead (first in sorted list with positive tangential offset)
    for tang_off, norm_off, rel_tv, rel_nv in relatives:
        if tang_off > 3.0 and tang_off < 20.0:
            # Horse is ahead — try to align behind it
            # Lateral: steer to match their normal offset
            lateral_correction = norm_off * 0.3 * p.draft_seeking
            # Tangential: match their speed (relative vel ~0) but stay slightly behind
            speed_match = rel_tv * 0.2  # positive rel_tv means they're faster
            return ActionOutput(
                tangential=speed_match,
                normal=_clamp(lateral_correction, -2.0, 2.0),
                weight=0.3 * p.draft_seeking,
            )
        if tang_off > 20.0:
            break  # too far ahead, no point drafting

    return None  # no draftable horse found


def _kick(obs: dict, p: JockeyPersonality) -> ActionOutput | None:
    """Final stretch push — increase effort if stamina allows."""
    stamina = obs["stamina_ratio"]
    progress = obs["track_progress"]

    if stamina < p.stamina_reserve * 0.5:
        # Critically low — ease up to just finish
        return ActionOutput(tangential=0.5, normal=0.0, weight=0.6)

    if stamina < p.stamina_reserve:
        # Low but manageable — moderate push
        return ActionOutput(tangential=2.0, normal=0.0, weight=0.6)

    # Scale effort by remaining stamina and proximity to finish
    remaining_race = max(1.0 - progress, 0.02)
    intensity = min(1.0, stamina / (remaining_race * 2.0))
    effort = 3.0 + intensity * 4.0  # 3–7 range
    return ActionOutput(tangential=effort, normal=0.0, weight=0.6)


def _overtake(obs: dict, p: JockeyPersonality) -> ActionOutput | None:
    """Full overtake: detect blockage, pull out, pass, cut back.

    Phases:
    1. Blocked: slower horse ahead in same lane → steer to side with more room
    2. Alongside: side-by-side → accelerate past while holding lane offset
    3. Clear: no blockage → return None (let other behaviors take over)
    """
    if p.overtake_aggression < 0.05:
        return None

    relatives = obs["relatives"]
    displacement = obs["displacement"]
    track_half_w = 10.0  # approximate TRACK_HALF_WIDTH

    # Phase 1: detect horse blocking ahead (in front, same lane, slower)
    for tang_off, norm_off, rel_tv, rel_nv in relatives:
        if tang_off > 2.0 and tang_off < 15.0 and abs(norm_off) < 3.0 and rel_tv < -0.5:
            # Blocked — pull out to side with more room
            room_inside = displacement + track_half_w   # how far to inner rail
            room_outside = track_half_w - displacement   # how far to outer rail
            if room_outside > room_inside:
                steer = 2.0 * p.overtake_aggression  # go outside (positive normal)
            else:
                steer = -2.0 * p.overtake_aggression  # go inside (negative normal)
            return ActionOutput(
                tangential=2.5 * p.overtake_aggression,
                normal=_clamp(steer, -4.0, 4.0),
                weight=0.5 * p.overtake_aggression,
            )

    # Phase 2: alongside — accelerate past while holding offset
    for tang_off, norm_off, rel_tv, rel_nv in relatives:
        if abs(tang_off) < 5.0 and abs(norm_off) < 6.0:
            # Hold current lateral offset (don't steer into them)
            hold_steer = 0.0
            if abs(norm_off) < 2.5:
                # Too close laterally — nudge away
                hold_steer = 1.5 * p.overtake_aggression * (1.0 if norm_off < 0 else -1.0)
            return ActionOutput(
                tangential=3.0 * p.overtake_aggression,
                normal=_clamp(hold_steer, -3.0, 3.0),
                weight=0.4 * p.overtake_aggression,
            )

    return None  # clear — no overtake needed


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------


def build_default_tree() -> BTNode:
    """Construct the standard racing behavior tree."""
    return Selector([
        # Emergency: stamina critically low
        Condition(
            lambda obs, p: obs["stamina_ratio"] < 0.15,
            Leaf(_emergency_brake),
        ),

        # Gate break: first 8% of race
        Condition(
            lambda obs, p: obs["track_progress"] < 0.08,
            Leaf(_gate_break),
        ),

        # Early race: pace + overtake + cornering
        Condition(
            lambda obs, p: obs["track_progress"] < 0.40,
            Blend([
                Leaf(_pace_control),
                Leaf(_overtake),
                Leaf(_cornering_line),
            ]),
        ),

        # Mid race: pace + overtake + drafting + cornering
        Condition(
            lambda obs, p: obs["track_progress"] < p.kick_progress,
            Blend([
                Leaf(_pace_control),
                Leaf(_overtake),
                Leaf(_drafting),
                Leaf(_cornering_line),
            ]),
        ),

        # Final stretch: kick + overtake + cornering
        Condition(
            lambda obs, p: obs["track_progress"] >= p.kick_progress,
            Blend([
                Leaf(_kick),
                Leaf(_overtake),
                Leaf(_cornering_line),
            ]),
        ),

        # Fallback: cruise
        Leaf(lambda obs, p: ActionOutput(0.0, 0.0, 1.0)),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class BTJockey:
    """Behavior-tree AI jockey that produces HorseAction from observations."""

    def __init__(self, personality: JockeyPersonality | None = None):
        self.personality = personality or DEFAULT_PERSONALITY
        self._tree = build_default_tree()

    def compute_action(self, obs: dict[str, Any]) -> HorseAction:
        """Given an observation dict, return a clamped HorseAction."""
        result = self._tree.tick(obs, self.personality)
        if result is None:
            return HorseAction()
        return HorseAction(
            extra_tangential=_clamp(result.tangential, -10.0, 10.0),
            extra_normal=_clamp(result.normal, -5.0, 5.0),
        )


def make_bt_jockey(archetype: str | None = None) -> BTJockey:
    """Create a BTJockey with the given archetype personality."""
    personality = PERSONALITIES.get(archetype or "", DEFAULT_PERSONALITY)
    return BTJockey(personality)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
