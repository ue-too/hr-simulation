"""Observation vector builder — mirrors TS observation.ts."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .attributes import TRAIT_RANGES, CoreAttributes
from .types import MAX_HORSES, TRACK_HALF_WIDTH, Horse

if TYPE_CHECKING:
    from .race import Race

LOOKAHEAD_DISTANCES = [25, 50, 100, 200]

SELF_STATE_SIZE = 15
TRACK_CONTEXT_SIZE = 2 + len(LOOKAHEAD_DISTANCES) * 2  # 10
OPPONENT_SLOT_SIZE = 5
OPPONENT_SLOTS = MAX_HORSES - 1  # 23
OBS_SIZE = SELF_STATE_SIZE + TRACK_CONTEXT_SIZE + OPPONENT_SLOTS * OPPONENT_SLOT_SIZE  # 140


def normalize_trait(value: float, key: str) -> float:
    lo, hi = TRAIT_RANGES[key]
    if hi - lo < 1e-12:
        return 0.0
    return (value - lo) / (hi - lo)


def curvature(turn_radius: float) -> float:
    return 1.0 / turn_radius if turn_radius < 1e6 else 0.0


def _normal_offset(
    opponent: Horse, self_horse: Horse, self_frame_normal: np.ndarray
) -> float:
    dx = opponent.pos[0] - self_horse.pos[0]
    dy = opponent.pos[1] - self_horse.pos[1]
    projection = dx * self_frame_normal[0] + dy * self_frame_normal[1]
    return projection / TRACK_HALF_WIDTH


def build_observations(race: Race) -> list[np.ndarray]:
    """Build a 140-float observation vector per horse."""
    horses = race.state.horses
    result: list[np.ndarray] = []

    for self_horse in horses:
        obs = np.zeros(OBS_SIZE, dtype=np.float64)
        base = self_horse.base_attributes
        eff = self_horse.effective_attributes
        frame = self_horse.navigator.get_track_frame(self_horse.pos)

        obs[0] = self_horse.track_progress
        obs[1] = self_horse.tangential_vel / base.max_speed
        obs[2] = self_horse.normal_vel / base.max_speed
        obs[3] = self_horse.current_stamina / base.max_stamina
        obs[4] = eff.cruise_speed / base.cruise_speed
        obs[5] = eff.max_speed / base.max_speed
        obs[6] = eff.forward_accel / base.forward_accel
        obs[7] = eff.turn_accel / base.turn_accel
        obs[8] = normalize_trait(base.cruise_speed, "cruise_speed")
        obs[9] = normalize_trait(base.max_speed, "max_speed")
        obs[10] = normalize_trait(base.forward_accel, "forward_accel")
        obs[11] = normalize_trait(base.turn_accel, "turn_accel")
        obs[12] = normalize_trait(base.cornering_grip, "cornering_grip")
        obs[13] = normalize_trait(base.weight, "weight")
        obs[14] = self_horse.last_drain / base.max_stamina

        obs[15] = curvature(frame.turn_radius)
        obs[16] = frame.slope

        for i, dist in enumerate(LOOKAHEAD_DISTANCES):
            lookahead = self_horse.navigator.sample_track_ahead(self_horse.pos, dist)
            obs[17 + i * 2] = curvature(lookahead.turn_radius)
            obs[17 + i * 2 + 1] = lookahead.slope

        opponents = sorted(
            [h for h in horses if h.id != self_horse.id],
            key=lambda h: abs(h.track_progress - self_horse.track_progress),
        )

        opponent_base = SELF_STATE_SIZE + TRACK_CONTEXT_SIZE
        for s in range(OPPONENT_SLOTS):
            offset = opponent_base + s * OPPONENT_SLOT_SIZE
            if s < len(opponents):
                opp = opponents[s]
                obs[offset + 0] = 1.0
                obs[offset + 1] = opp.track_progress - self_horse.track_progress
                obs[offset + 2] = (opp.tangential_vel - self_horse.tangential_vel) / base.max_speed
                obs[offset + 3] = _normal_offset(opp, self_horse, frame.normal)
                obs[offset + 4] = (opp.normal_vel - self_horse.normal_vel) / base.max_speed

        result.append(obs)

    return result
