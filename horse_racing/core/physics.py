"""Physics engine — mirrors TS physics.ts."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .attributes import CoreAttributes, F_N_MAX, F_T_MAX
from .track_navigator import TrackFrame
from .types import (
    C_DRAG,
    Horse,
    InputState,
    K_CRUISE,
    NORMAL_DAMP,
)

if TYPE_CHECKING:
    from .collision import CollisionWorld


def project_velocity(
    world_vel: np.ndarray, frame: TrackFrame
) -> tuple[float, float]:
    """Project world-space velocity onto track-relative components.

    Returns (tangential_vel, normal_vel).
    """
    tang = float(np.dot(world_vel, frame.tangential))
    norm = float(np.dot(world_vel, frame.normal))
    return tang, norm


def compute_cruise_force(current_vel: float, cruise_speed: float) -> float:
    """Proportional cruise controller — mirrors TS cruise.ts."""
    return K_CRUISE * (cruise_speed - current_vel)


def compute_accelerations(
    tangential_vel: float,
    normal_vel: float,
    attrs: CoreAttributes,
    inp: InputState,
    frame: TrackFrame,
) -> tuple[float, float]:
    """Compute track-relative accelerations for a single horse.

    Returns (tangential_accel, normal_accel) in m/s².
    """
    clamped_tan = max(-1.0, min(1.0, inp.tangential))
    clamped_nor = max(-1.0, min(1.0, inp.normal))

    # --- Tangential ---
    a_t = compute_cruise_force(tangential_vel, attrs.cruise_speed)
    a_t += clamped_tan * F_T_MAX * attrs.forward_accel
    a_t -= C_DRAG * tangential_vel
    a_t -= 9.81 * frame.slope
    if tangential_vel >= attrs.max_speed and a_t > 0:
        a_t = 0.0

    # --- Normal ---
    a_n = 0.0
    if frame.turn_radius < 1e6 and frame.turn_radius > 1e-3:
        a_n -= (tangential_vel * tangential_vel) / frame.turn_radius
    a_n -= NORMAL_DAMP * normal_vel
    a_n += clamped_nor * F_N_MAX * attrs.turn_accel
    a_n -= C_DRAG * normal_vel

    return a_t, a_n


def step_physics(
    horses: list[Horse],
    inputs: dict[int, InputState],
    collision_world: CollisionWorld | None,
    substeps: int,
    dt: float,
) -> None:
    """Run physics substeps for all horses.

    Per substep:
    1. Compute forces and Euler integrate velocity + position
    2. Collision resolve (if collision_world provided)
    3. Sync horse state (navigator, progress, track-relative velocities)
    """
    zero_input = InputState(0.0, 0.0)

    for _ in range(substeps):
        # Build world velocities for each horse before the substep
        world_vels: dict[int, np.ndarray] = {}

        for h in horses:
            if h.finished:
                continue
            frame = h.navigator.get_track_frame(h.pos)
            inp = inputs.get(h.id, zero_input)
            a_t, a_n = compute_accelerations(
                h.tangential_vel, h.normal_vel, h.effective_attributes, inp, frame
            )

            # Reconstruct world velocity from track-relative components
            vx = h.tangential_vel * frame.tangential[0] + h.normal_vel * frame.normal[0]
            vy = h.tangential_vel * frame.tangential[1] + h.normal_vel * frame.normal[1]

            # Convert track-relative acceleration to world-space
            ax = a_t * frame.tangential[0] + a_n * frame.normal[0]
            ay = a_t * frame.tangential[1] + a_n * frame.normal[1]

            # Euler integrate
            vx += ax * dt
            vy += ay * dt
            h.pos[0] += vx * dt
            h.pos[1] += vy * dt

            world_vels[h.id] = np.array([vx, vy])

            # Update collision body if world exists
            if collision_world is not None:
                collision_world.set_horse_state(h.id, h.pos, np.array([vx, vy]), frame)

        # Collision resolve
        if collision_world is not None:
            collision_world.step(dt)

        # Sync back
        for h in horses:
            if h.finished:
                continue

            if collision_world is not None:
                pos, vel = collision_world.get_horse_state(h.id)
                h.pos[0] = pos[0]
                h.pos[1] = pos[1]
                world_vel = vel
            else:
                world_vel = world_vels.get(h.id, np.zeros(2))

            h.navigator.update_segment(h.pos)
            h.track_progress = h.navigator.compute_progress(h.pos)

            frame = h.navigator.get_track_frame(h.pos)
            h.tangential_vel, h.normal_vel = project_velocity(world_vel, frame)
