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


def resolve_horse_collisions(
    bodies: list[HorseBody],
    masses: list[float],
    pushing_powers: list[float] | None = None,
    push_resistances: list[float] | None = None,
) -> list[bool]:
    """Resolve circle-circle collisions between all horse pairs.

    Matches JS engine: fixed 50/50 position correction, restitution=0.4,
    impulse always applied (even when not approaching).

    Pushing power and push resistance scale effective mass for collision
    impulse only (not position correction or normal physics).

    Returns a list of booleans indicating which horses were involved in a collision.
    """
    n = len(bodies)
    collided = [False] * n

    # Compute effective collision masses
    collision_masses = []
    for i in range(n):
        pp = pushing_powers[i] if pushing_powers else 0.5
        pr = push_resistances[i] if push_resistances else 0.5
        collision_masses.append(masses[i] * (1.0 + pp) * (1.0 + pr))

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

                # Velocity impulse using effective collision mass
                rel_vel = bodies[i].velocity - bodies[j].velocity
                vel_along_normal = float(np.dot(rel_vel, normal))

                e = 0.4  # restitution matches JS engine
                inv_mass_i = 1.0 / collision_masses[i]
                inv_mass_j = 1.0 / collision_masses[j]
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


# ---------------------------------------------------------------------------
# Rail-based wall collision — circle vs explicit rail paths
# ---------------------------------------------------------------------------


def _nearest_point_on_straight(
    pos: np.ndarray, seg: StraightSegment
) -> tuple[np.ndarray, float]:
    """Find nearest point on a straight segment to pos. Returns (point, distance)."""
    start = _vec2(*seg.start_point)
    end = _vec2(*seg.end_point)
    ab = end - start
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-12:
        return start, float(np.linalg.norm(pos - start))

    t = float(np.dot(pos - start, ab)) / ab_len_sq
    t = max(0.0, min(1.0, t))
    nearest = start + ab * t
    return nearest, float(np.linalg.norm(pos - nearest))


def _nearest_point_on_curve(
    pos: np.ndarray, seg: CurveSegment
) -> tuple[np.ndarray, float]:
    """Find nearest point on a curve segment arc to pos. Returns (point, distance)."""
    center = _vec2(*seg.center)
    to_pos = pos - center
    dist_to_center = float(np.linalg.norm(to_pos))

    if dist_to_center < 1e-6:
        # Position is at the center — nearest is start point
        start = _vec2(*seg.start_point)
        return start, float(np.linalg.norm(pos - start))

    # Angle of pos relative to center
    pos_angle = math.atan2(float(to_pos[1]), float(to_pos[0]))

    # Start angle of the arc
    start_angle = math.atan2(
        seg.start_point[1] - seg.center[1],
        seg.start_point[0] - seg.center[0],
    )

    # Check if pos_angle falls within the arc span
    # Normalize angle difference to determine if within arc
    span = seg.angle_span
    if span >= 0:
        # CCW arc: angles go from start_angle to start_angle + span
        diff = _normalize_angle(pos_angle - start_angle)
        if diff < 0:
            diff += 2 * math.pi
        in_arc = diff <= span
    else:
        # CW arc: angles go from start_angle to start_angle + span (negative)
        diff = _normalize_angle(pos_angle - start_angle)
        if diff > 0:
            diff -= 2 * math.pi
        in_arc = diff >= span

    if in_arc:
        # Project onto the arc at the horse's angle
        direction = to_pos / dist_to_center
        nearest = center + direction * seg.radius
        return nearest, abs(dist_to_center - seg.radius)
    else:
        # Clamp to the nearest arc endpoint
        start_pt = _vec2(*seg.start_point)
        end_pt = _vec2(*seg.end_point)
        d_start = float(np.linalg.norm(pos - start_pt))
        d_end = float(np.linalg.norm(pos - end_pt))
        if d_start <= d_end:
            return start_pt, d_start
        return end_pt, d_end


def _normalize_angle(a: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _detect_rail_collision(
    body: HorseBody,
    rail_segments: list[TrackSegment],
    rail_bboxes: list[tuple[float, float, float, float]],
) -> tuple[np.ndarray, float] | None:
    """Detect collision between horse and explicit rail path.

    Returns (normal, depth) for the deepest penetration, or None.
    Normal points from rail toward horse (push direction).
    """
    px, py = float(body.position[0]), float(body.position[1])
    best_depth = 0.0
    best_normal: np.ndarray | None = None
    margin = HORSE_RADIUS + 1.0  # expand bbox for safety

    for i, seg in enumerate(rail_segments):
        # Bounding box culling
        bmin_x, bmin_y, bmax_x, bmax_y = rail_bboxes[i]
        if px < bmin_x - margin or px > bmax_x + margin:
            continue
        if py < bmin_y - margin or py > bmax_y + margin:
            continue

        if isinstance(seg, StraightSegment):
            nearest, dist = _nearest_point_on_straight(body.position, seg)
        else:
            nearest, dist = _nearest_point_on_curve(body.position, seg)

        if dist < HORSE_RADIUS:
            depth = HORSE_RADIUS - dist
            if depth > best_depth:
                best_depth = depth
                if dist > 1e-6:
                    best_normal = (body.position - nearest) / dist
                else:
                    # Horse is exactly on rail — push along arbitrary direction
                    best_normal = _vec2(1.0, 0.0)

    if best_normal is not None:
        return (best_normal, best_depth)
    return None


def resolve_wall_collisions(
    bodies: list[HorseBody],
    segments: list[TrackSegment],
    segment_indices: list[int],
    inner_rails: list[TrackSegment] | None = None,
    outer_rails: list[TrackSegment] | None = None,
    inner_bboxes: list[tuple[float, float, float, float]] | None = None,
    outer_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> None:
    """Resolve wall collisions for all horses.

    When explicit rail data is provided (inner_rails, outer_rails, bboxes),
    collision is checked against those paths. Otherwise falls back to
    centerline-derived boundaries.
    """
    use_rails = (
        inner_rails is not None
        and outer_rails is not None
        and inner_bboxes is not None
        and outer_bboxes is not None
    )

    for idx, body in enumerate(bodies):
        if use_rails:
            hit = _detect_rail_collision(body, inner_rails, inner_bboxes)
            if hit is not None:
                _resolve_wall_impulse(body, hit[0], hit[1])
            hit = _detect_rail_collision(body, outer_rails, outer_bboxes)
            if hit is not None:
                _resolve_wall_impulse(body, hit[0], hit[1])
        else:
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
