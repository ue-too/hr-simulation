"""Drain-only stamina model — mirrors TS stamina.ts."""

import math

from .attributes import CoreAttributes
from .track_navigator import TrackFrame
from .types import Horse, InputState

OVERDRIVE_DRAIN_RATE = 0.01
STAMINA_DRAIN_RATE = 0.015
LATERAL_STEERING_DRAIN_RATE = 0.006
CORNERING_DRAIN_RATE = 0.002
SPEED_DRAIN_RATE = 0.002
LATERAL_VELOCITY_DRAIN_RATE = 0.0008
GRIP_FORCE_BASELINE = 2.0


def drain_stamina(
    horse: Horse,
    attrs: CoreAttributes,
    inp: InputState,
    frame: TrackFrame,
) -> None:
    """Drain stamina based on current effort. Mutates horse.current_stamina."""
    drain = 0.0

    if horse.tangential_vel > attrs.cruise_speed:
        drain += (horse.tangential_vel - attrs.cruise_speed) * OVERDRIVE_DRAIN_RATE

    if inp.tangential > 0:
        drain += abs(inp.tangential) * STAMINA_DRAIN_RATE

    if inp.normal != 0:
        drain += abs(inp.normal) * LATERAL_STEERING_DRAIN_RATE

    if frame.turn_radius < 1e6 and horse.tangential_vel > 0:
        required_force = (horse.tangential_vel ** 2) / frame.turn_radius
        tolerated_force = attrs.cornering_grip * GRIP_FORCE_BASELINE
        if required_force > tolerated_force:
            drain += (required_force - tolerated_force) * CORNERING_DRAIN_RATE

    drain += abs(horse.tangential_vel) * SPEED_DRAIN_RATE
    drain += abs(horse.normal_vel) * LATERAL_VELOCITY_DRAIN_RATE
    drain *= attrs.drain_rate_mult
    horse.current_stamina = max(0.0, horse.current_stamina - drain)
