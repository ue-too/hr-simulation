"""Race state machine — mirrors TS race.ts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .attributes import create_default_attributes
from .collision import CollisionWorld
from .exhaustion import apply_exhaustion
from .physics import step_physics
from .stamina import drain_stamina
from .track import TrackSegment
from .track_navigator import TrackNavigator
from .types import (
    FIXED_DT,
    Horse,
    InputState,
    MAX_HORSES,
    PHYS_SUBSTEPS,
    TRACK_HALF_WIDTH,
)

_BASE_COLORS = [
    0xC9A227, 0x4169E1, 0xE53935, 0x43A047, 0x8E24AA, 0xF57C00, 0x00897B,
    0xC62828, 0x1565C0, 0x6A1B9A, 0xEF6C00, 0x2E7D32, 0xAD1457, 0x00838F,
    0x4E342E, 0x37474F, 0xFDD835, 0x7CB342, 0x039BE5, 0xD81B60, 0x00ACC1,
    0x5D4037, 0x546E7A, 0xFF8F00,
]


@dataclass
class RaceState:
    phase: str  # "gate" | "running" | "finished"
    horses: list[Horse]
    player_horse_id: int | None
    tick: int
    finish_order: list[int]


def spawn_horses(
    segments: list[TrackSegment],
    horse_count: int = 4,
) -> list[Horse]:
    if len(segments) == 0:
        raise ValueError("spawn_horses: track has no segments")
    count = max(1, min(MAX_HORSES, horse_count))
    first = segments[0]
    start_point = first.start_point.copy()

    probe = TrackNavigator(segments, half_track_width=TRACK_HALF_WIDTH)
    frame = probe.get_track_frame(start_point)

    lane_spacing = (TRACK_HALF_WIDTH * 2 * 0.8) / (count - 1) if count > 1 else 0.0

    horses: list[Horse] = []
    for i in range(count):
        lane_offset = (-TRACK_HALF_WIDTH * 0.8 + i * lane_spacing) if count > 1 else 0.0
        pos = start_point + frame.normal * lane_offset
        attrs = create_default_attributes()
        horses.append(Horse(
            id=i,
            color=_BASE_COLORS[i % len(_BASE_COLORS)],
            pos=pos.copy(),
            tangential_vel=0.0,
            normal_vel=0.0,
            track_progress=0.0,
            navigator=TrackNavigator(segments, half_track_width=TRACK_HALF_WIDTH),
            finished=False,
            finish_order=None,
            base_attributes=attrs,
            current_stamina=attrs.max_stamina,
            effective_attributes=attrs,
        ))
    return horses


class Race:
    def __init__(self, segments: list[TrackSegment], horse_count: int = 4):
        self._segments = segments
        self._horse_count = horse_count
        self.state = RaceState(
            phase="gate",
            horses=spawn_horses(segments, horse_count),
            player_horse_id=None,
            tick=0,
            finish_order=[],
        )
        self._collision_world = CollisionWorld(segments, TRACK_HALF_WIDTH)
        self._add_horse_bodies()

    def _add_horse_bodies(self) -> None:
        for h in self.state.horses:
            frame = h.navigator.get_track_frame(h.pos)
            self._collision_world.add_horse(
                h.id, h.pos, frame, h.base_attributes.weight
            )

    def start(self, player_horse_id: int | None) -> None:
        if self.state.phase != "gate":
            return
        self.state.player_horse_id = player_horse_id
        self.state.phase = "running"
        self.state.tick = 0

    def tick(self, inputs: dict[int, InputState]) -> None:
        if self.state.phase != "running":
            return

        zero_input = InputState(0.0, 0.0)

        for h in self.state.horses:
            if not h.finished:
                h.effective_attributes = apply_exhaustion(h)

        step_physics(
            self.state.horses, inputs, self._collision_world, PHYS_SUBSTEPS, FIXED_DT
        )

        for h in self.state.horses:
            if not h.finished:
                frame = h.navigator.get_track_frame(h.pos)
                horse_input = inputs.get(h.id, zero_input)
                drain_stamina(h, h.effective_attributes, horse_input, frame)

        for h in self.state.horses:
            if not h.finished and h.track_progress >= 1.0:
                h.finished = True
                h.finish_order = len(self.state.finish_order) + 1
                self.state.finish_order.append(h.id)
                self._collision_world.remove_horse(h.id)

        player_id = self.state.player_horse_id
        is_player_mode = player_id is not None
        if is_player_mode:
            player = self.state.horses[player_id]
            if player.finished:
                self.state.phase = "finished"
        else:
            if all(h.finished for h in self.state.horses):
                self.state.phase = "finished"

        self.state.tick += 1

    def reset(self) -> None:
        self.state = RaceState(
            phase="gate",
            horses=spawn_horses(self._segments, self._horse_count),
            player_horse_id=None,
            tick=0,
            finish_order=[],
        )
        self._collision_world = CollisionWorld(self._segments, TRACK_HALF_WIDTH)
        self._add_horse_bodies()
