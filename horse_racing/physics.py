"""Simple 2D physics: integration, collision detection, and resolution."""

from __future__ import annotations

import math

import numpy as np

from horse_racing.types import (
    HORSE_RADIUS,
    RAIL_THICKNESS,
    TRACK_HALF_WIDTH,
    CurveSegment,
    HorseBody,
    StraightSegment,
    TrackSegment,
)


def _vec2(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=np.float64)


# ---------------------------------------------------------------------------
# Integration step
# ---------------------------------------------------------------------------


def integrate(body: HorseBody, mass: float, dt: float) -> None:
    """Semi-implicit Euler: update velocity from force, then position from velocity."""
    body.velocity += (body.force / mass) * dt
    body.position += body.velocity * dt


# ---------------------------------------------------------------------------
# Horse-horse collision
# ---------------------------------------------------------------------------


def resolve_horse_collisions(bodies: list[HorseBody], masses: list[float]) -> list[bool]:
    """Resolve circle-circle collisions between all horse pairs.

    Matches JS engine: fixed 50/50 position correction, restitution=0.4,
    impulse always applied (even when not approaching).

    Returns a list of booleans indicating which horses were involved in a collision.
    """
    n = len(bodies)
    collided = [False] * n

    for i in range(n):
        for j in range(i + 1, n):
            delta = bodies[j].position - bodies[i].position
            dist = float(np.linalg.norm(delta))
            min_dist = HORSE_RADIUS * 2

            if dist < min_dist and dist > 1e-6:
                collided[i] = True
                collided[j] = True

                normal = delta / dist
                overlap = min_dist - dist

                # Position correction: fixed 50/50 split (matches JS engine)
                bodies[i].position -= normal * (overlap / 2)
                bodies[j].position += normal * (overlap / 2)

                # Velocity impulse — always applied (JS engine has no approach check)
                # normal points from i to j; relative velocity = A - B in JS convention
                rel_vel = bodies[i].velocity - bodies[j].velocity
                vel_along_normal = float(np.dot(rel_vel, normal))

                e = 0.4  # restitution matches JS engine
                inv_mass_i = 1.0 / masses[i]
                inv_mass_j = 1.0 / masses[j]
                j_impulse = -(1 + e) * vel_along_normal / (inv_mass_i + inv_mass_j)

                bodies[i].velocity += normal * (j_impulse * inv_mass_i)
                bodies[j].velocity -= normal * (j_impulse * inv_mass_j)

    return collided


# ---------------------------------------------------------------------------
# Wall collision — circle vs track boundaries
# ---------------------------------------------------------------------------


def resolve_all_collisions(
    bodies: list[HorseBody],
    masses: list[float],
    segments: list[TrackSegment],
    segment_indices: list[int],
) -> list[bool]:
    """Detect and resolve all collisions in a single pass, matching JS engine.

    JS behavior: broadphase detects overlaps at CURRENT positions, then
    narrowphase resolves all detected pairs. Overlaps created by resolution
    (e.g., horse pushed into wall) are NOT detected until the next substep.

    The order is: wall pairs for each body first (from rigidBodyList iteration
    where walls precede horses), then horse-horse pairs.
    """
    n = len(bodies)
    collided = [False] * n

    # Build collision pairs from current positions (broadphase equivalent)
    # JS order: iterate bodies in insertion order. Walls come first in rigidBodyMap,
    # but walls only detect against horses. In practice, for each body the broadphase
    # retrieves nearby bodies. We simulate this with:
    # 1. Detect wall overlaps for each horse at current positions
    # 2. Detect horse-horse overlaps at current positions
    # Then resolve all in the detected order.

    # Phase 1: Detect wall collision pairs at current positions
    wall_pairs: list[tuple[int, np.ndarray, float]] = []  # (horse_idx, normal, depth)
    for idx in range(n):
        body = bodies[idx]
        seg = segments[segment_indices[idx]]
        wall_hit = _detect_wall_collision(body, seg)
        if wall_hit is not None:
            wall_pairs.append((idx, wall_hit[0], wall_hit[1]))

    # Phase 2: Detect horse-horse collision pairs at current positions
    horse_pairs: list[tuple[int, int, np.ndarray, float]] = []  # (i, j, normal, depth)
    for i in range(n):
        for j in range(i + 1, n):
            delta = bodies[j].position - bodies[i].position
            dist = float(np.linalg.norm(delta))
            min_dist = HORSE_RADIUS * 2
            if dist < min_dist and dist > 1e-6:
                normal = delta / dist
                depth = min_dist - dist
                horse_pairs.append((i, j, normal, depth))

    # Resolve all detected pairs (wall pairs from body iteration, then horse pairs)
    # In JS, walls are in rigidBodyMap before horses, so wall-horse pairs
    # are generated first during broadphase iteration.
    for idx, normal, depth in wall_pairs:
        _resolve_wall_impulse(bodies[idx], normal, depth)

    for i, j, normal, depth in horse_pairs:
        # Recompute overlap with current positions (may have changed from wall resolution)
        delta = bodies[j].position - bodies[i].position
        dist = float(np.linalg.norm(delta))
        min_dist = HORSE_RADIUS * 2
        if dist < min_dist and dist > 1e-6:
            collided[i] = True
            collided[j] = True
            normal = delta / dist
            overlap = min_dist - dist

            bodies[i].position -= normal * (overlap / 2)
            bodies[j].position += normal * (overlap / 2)

            rel_vel = bodies[i].velocity - bodies[j].velocity
            vel_along_normal = float(np.dot(rel_vel, normal))
            e = 0.4
            inv_mass_i = 1.0 / masses[i]
            inv_mass_j = 1.0 / masses[j]
            j_impulse = -(1 + e) * vel_along_normal / (inv_mass_i + inv_mass_j)
            bodies[i].velocity += normal * (j_impulse * inv_mass_i)
            bodies[j].velocity -= normal * (j_impulse * inv_mass_j)

    return collided


def _detect_wall_collision(
    body: HorseBody, seg: TrackSegment
) -> tuple[np.ndarray, float] | None:
    """Detect wall collision at current position. Returns (normal, depth) or None."""
    if isinstance(seg, StraightSegment):
        return _detect_straight_wall(body, seg)
    else:
        return _detect_curve_wall(body, seg)


def _detect_straight_wall(
    body: HorseBody, seg: StraightSegment
) -> tuple[np.ndarray, float] | None:
    fwd = _vec2(
        seg.end_point[0] - seg.start_point[0],
        seg.end_point[1] - seg.start_point[1],
    )
    length = float(np.linalg.norm(fwd))
    if length < 1e-6:
        return None
    fwd = fwd / length
    outward = _vec2(fwd[1], -fwd[0])

    to_horse = body.position - _vec2(*seg.start_point)
    lateral = float(np.dot(to_horse, outward))

    wall_limit = TRACK_HALF_WIDTH - HORSE_RADIUS
    if lateral > wall_limit:
        return (-outward, lateral - wall_limit)
    elif lateral < -wall_limit:
        return (outward, -wall_limit - lateral)
    return None


def _detect_curve_wall(
    body: HorseBody, seg: CurveSegment
) -> tuple[np.ndarray, float] | None:
    center = _vec2(*seg.center)
    to_horse = body.position - center
    dist = float(np.linalg.norm(to_horse))
    if dist < 1e-6:
        return None

    normal = to_horse / dist

    outer_limit = seg.radius + TRACK_HALF_WIDTH - HORSE_RADIUS
    if dist > outer_limit:
        return (-normal, dist - outer_limit)

    inner_limit = seg.radius - TRACK_HALF_WIDTH + HORSE_RADIUS
    if inner_limit > 0 and dist < inner_limit:
        return (normal, inner_limit - dist)

    return None


def resolve_wall_collisions(
    bodies: list[HorseBody],
    segments: list[TrackSegment],
    segment_indices: list[int],
) -> None:
    """Legacy wall collision resolution (separate pass). Use resolve_all_collisions instead."""
    for idx, body in enumerate(bodies):
        seg = segments[segment_indices[idx]]
        if isinstance(seg, StraightSegment):
            hit = _detect_straight_wall(body, seg)
        else:
            hit = _detect_curve_wall(body, seg)
        if hit is not None:
            _resolve_wall_impulse(body, hit[0], hit[1])


WALL_RESTITUTION: float = 0.4


def _resolve_wall_impulse(body: HorseBody, normal: np.ndarray, depth: float) -> None:
    """Resolve collision between horse and static wall, matching JS engine.

    Position correction: full depth (wall is static, horse absorbs all correction).
    Velocity: impulse with restitution 0.4, always applied (no approach check).
    """
    # Position correction: horse moves full depth (static wall gets depth/2 twice)
    body.position += normal * depth

    # Velocity impulse: same formula as resolveCollision with inverseMassB=0
    vel_along_normal = float(np.dot(body.velocity, normal))
    # deltaVel = -(1+e) * vel_n * normal
    body.velocity += normal * (-(1 + WALL_RESTITUTION) * vel_along_normal)


def _resolve_straight_walls(body: HorseBody, seg: StraightSegment) -> None:
    fwd = _vec2(
        seg.end_point[0] - seg.start_point[0],
        seg.end_point[1] - seg.start_point[1],
    )
    length = float(np.linalg.norm(fwd))
    if length < 1e-6:
        return
    fwd = fwd / length
    outward = _vec2(fwd[1], -fwd[0])  # rotate -90

    to_horse = body.position - _vec2(*seg.start_point)
    lateral = float(np.dot(to_horse, outward))

    # Outer wall: horse circle touches rail at TRACK_HALF_WIDTH
    wall_limit = TRACK_HALF_WIDTH - HORSE_RADIUS
    if lateral > wall_limit:
        depth = lateral - wall_limit
        _resolve_wall_impulse(body, -outward, depth)  # push inward
    elif lateral < -wall_limit:
        depth = -wall_limit - lateral
        _resolve_wall_impulse(body, outward, depth)  # push outward


def _resolve_curve_walls(body: HorseBody, seg: CurveSegment) -> None:
    center = _vec2(*seg.center)
    to_horse = body.position - center
    dist = float(np.linalg.norm(to_horse))
    if dist < 1e-6:
        return

    normal = to_horse / dist  # points outward from center

    # Outer wall
    outer_limit = seg.radius + TRACK_HALF_WIDTH - HORSE_RADIUS
    if dist > outer_limit:
        depth = dist - outer_limit
        _resolve_wall_impulse(body, -normal, depth)  # push inward

    # Inner wall
    inner_limit = seg.radius - TRACK_HALF_WIDTH + HORSE_RADIUS
    if inner_limit > 0 and dist < inner_limit:
        depth = inner_limit - dist
        _resolve_wall_impulse(body, normal, depth)  # push outward
