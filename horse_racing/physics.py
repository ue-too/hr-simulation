"""Simple 2D physics: integration, collision detection, and resolution.

Horses are oriented rectangles (HORSE_LENGTH × HORSE_WIDTH). Collision
between horses uses separating-axis test on the two OBBs. Wall collision
treats the horse as a point offset by HORSE_HALF_WIDTH from the track edge
(lateral clearance only — length is negligible for wall contact).
"""

from __future__ import annotations

import math

import numpy as np

from horse_racing.types import (
    HORSE_HALF_LENGTH,
    HORSE_HALF_WIDTH,
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
# Oriented bounding box (OBB) helpers
# ---------------------------------------------------------------------------


def _obb_corners(pos: np.ndarray, orientation: float,
                 half_l: float, half_w: float) -> np.ndarray:
    """Return 4 corners of an OBB as (4, 2) array."""
    c = math.cos(orientation)
    s = math.sin(orientation)
    fwd = np.array([c, s])
    side = np.array([-s, c])
    fl = fwd * half_l
    sw = side * half_w
    return np.array([
        pos + fl + sw,
        pos + fl - sw,
        pos - fl - sw,
        pos - fl + sw,
    ])


def _obb_axes(orientation: float) -> list[np.ndarray]:
    """Return 2 unique edge normals (separating axes) for an OBB."""
    c = math.cos(orientation)
    s = math.sin(orientation)
    return [np.array([c, s]), np.array([-s, c])]


def _project(corners: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    """Project corners onto axis, return (min, max)."""
    dots = corners @ axis
    return float(dots.min()), float(dots.max())


def _obb_overlap(
    pos_a: np.ndarray, ori_a: float,
    pos_b: np.ndarray, ori_b: float,
    half_l: float = HORSE_HALF_LENGTH,
    half_w: float = HORSE_HALF_WIDTH,
) -> tuple[np.ndarray, float] | None:
    """SAT overlap test between two identical OBBs.

    Returns (normal_from_a_to_b, overlap_depth) or None if no overlap.
    """
    corners_a = _obb_corners(pos_a, ori_a, half_l, half_w)
    corners_b = _obb_corners(pos_b, ori_b, half_l, half_w)

    axes = _obb_axes(ori_a) + _obb_axes(ori_b)

    min_overlap = float("inf")
    min_axis = None

    for axis in axes:
        min_a, max_a = _project(corners_a, axis)
        min_b, max_b = _project(corners_b, axis)

        if max_a < min_b or max_b < min_a:
            return None  # separating axis found

        overlap = min(max_a - min_b, max_b - min_a)
        if overlap < min_overlap:
            min_overlap = overlap
            min_axis = axis

    if min_axis is None:
        return None

    # Ensure normal points from A to B
    delta = pos_b - pos_a
    if float(np.dot(min_axis, delta)) < 0:
        min_axis = -min_axis

    return (min_axis, min_overlap)


# ---------------------------------------------------------------------------
# Horse-horse collision
# ---------------------------------------------------------------------------


def resolve_horse_collisions(
    bodies: list[HorseBody],
    masses: list[float],
    pushing_powers: list[float] | None = None,
    push_resistances: list[float] | None = None,
) -> list[bool]:
    """Resolve OBB collisions between all horse pairs.

    Uses separating-axis theorem for oriented rectangles. Position correction
    is 50/50 split, velocity impulse uses restitution=0.4.

    Returns a list of booleans indicating which horses were involved in a collision.
    """
    n = len(bodies)
    collided = [False] * n

    collision_masses = []
    for i in range(n):
        pp = pushing_powers[i] if pushing_powers else 0.5
        pr = push_resistances[i] if push_resistances else 0.5
        collision_masses.append(masses[i] * (1.0 + pp) * (1.0 + pr))

    for i in range(n):
        for j in range(i + 1, n):
            # Quick distance check before expensive SAT
            delta = bodies[j].position - bodies[i].position
            dist = float(np.linalg.norm(delta))
            max_extent = HORSE_HALF_LENGTH + HORSE_HALF_WIDTH  # diagonal bound
            if dist > max_extent * 2:
                continue

            result = _obb_overlap(
                bodies[i].position, bodies[i].orientation,
                bodies[j].position, bodies[j].orientation,
            )
            if result is None:
                continue

            normal, overlap = result
            collided[i] = True
            collided[j] = True

            # Position correction: fixed 50/50 split
            bodies[i].position -= normal * (overlap / 2)
            bodies[j].position += normal * (overlap / 2)

            # Velocity impulse
            rel_vel = bodies[i].velocity - bodies[j].velocity
            vel_along_normal = float(np.dot(rel_vel, normal))

            e = 0.4
            inv_mass_i = 1.0 / collision_masses[i]
            inv_mass_j = 1.0 / collision_masses[j]
            j_impulse = -(1 + e) * vel_along_normal / (inv_mass_i + inv_mass_j)

            bodies[i].velocity += normal * (j_impulse * inv_mass_i)
            bodies[j].velocity -= normal * (j_impulse * inv_mass_j)

    return collided


# ---------------------------------------------------------------------------
# Wall collision — rectangle vs track boundaries
# ---------------------------------------------------------------------------


def resolve_all_collisions(
    bodies: list[HorseBody],
    masses: list[float],
    segments: list[TrackSegment],
    segment_indices: list[int],
) -> list[bool]:
    """Detect and resolve all collisions in a single pass.

    Order: wall collisions first, then horse-horse collisions.
    """
    n = len(bodies)
    collided = [False] * n

    # Phase 1: Detect wall collision pairs at current positions
    wall_pairs: list[tuple[int, np.ndarray, float]] = []
    for idx in range(n):
        body = bodies[idx]
        seg = segments[segment_indices[idx]]
        wall_hit = _detect_wall_collision(body, seg)
        if wall_hit is not None:
            wall_pairs.append((idx, wall_hit[0], wall_hit[1]))

    # Phase 2: Detect horse-horse collision pairs at current positions
    horse_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            delta = bodies[j].position - bodies[i].position
            dist = float(np.linalg.norm(delta))
            max_extent = HORSE_HALF_LENGTH + HORSE_HALF_WIDTH
            if dist < max_extent * 2:
                horse_pairs.append((i, j))

    # Resolve walls first
    for idx, normal, depth in wall_pairs:
        _resolve_wall_impulse(bodies[idx], normal, depth)

    # Resolve horse-horse
    for i, j in horse_pairs:
        result = _obb_overlap(
            bodies[i].position, bodies[i].orientation,
            bodies[j].position, bodies[j].orientation,
        )
        if result is not None:
            normal, overlap = result
            collided[i] = True
            collided[j] = True

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

    wall_limit = TRACK_HALF_WIDTH - HORSE_HALF_WIDTH
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

    outer_limit = seg.radius + TRACK_HALF_WIDTH - HORSE_HALF_WIDTH
    if dist > outer_limit:
        return (-normal, dist - outer_limit)

    inner_limit = seg.radius - TRACK_HALF_WIDTH + HORSE_HALF_WIDTH
    if inner_limit > 0 and dist < inner_limit:
        return (normal, inner_limit - dist)

    return None


# ---------------------------------------------------------------------------
# Rail-based wall collision — point vs explicit rail paths
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
        start = _vec2(*seg.start_point)
        return start, float(np.linalg.norm(pos - start))

    pos_angle = math.atan2(float(to_pos[1]), float(to_pos[0]))

    start_angle = math.atan2(
        seg.start_point[1] - seg.center[1],
        seg.start_point[0] - seg.center[0],
    )

    span = seg.angle_span
    if span >= 0:
        diff = _normalize_angle(pos_angle - start_angle)
        if diff < 0:
            diff += 2 * math.pi
        in_arc = diff <= span
    else:
        diff = _normalize_angle(pos_angle - start_angle)
        if diff > 0:
            diff -= 2 * math.pi
        in_arc = diff >= span

    if in_arc:
        direction = to_pos / dist_to_center
        nearest = center + direction * seg.radius
        return nearest, abs(dist_to_center - seg.radius)
    else:
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
    margin = HORSE_HALF_WIDTH + 1.0  # expand bbox for safety

    for i, seg in enumerate(rail_segments):
        bmin_x, bmin_y, bmax_x, bmax_y = rail_bboxes[i]
        if px < bmin_x - margin or px > bmax_x + margin:
            continue
        if py < bmin_y - margin or py > bmax_y + margin:
            continue

        if isinstance(seg, StraightSegment):
            nearest, dist = _nearest_point_on_straight(body.position, seg)
        else:
            nearest, dist = _nearest_point_on_curve(body.position, seg)

        if dist < HORSE_HALF_WIDTH:
            depth = HORSE_HALF_WIDTH - dist
            if depth > best_depth:
                best_depth = depth
                if dist > 1e-6:
                    best_normal = (body.position - nearest) / dist
                else:
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
    """Resolve wall collisions for all horses."""
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
    """Resolve collision between horse and static wall.

    Position correction: full depth. Velocity: impulse with restitution 0.4.
    """
    body.position += normal * depth
    vel_along_normal = float(np.dot(body.velocity, normal))
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
    outward = _vec2(fwd[1], -fwd[0])

    to_horse = body.position - _vec2(*seg.start_point)
    lateral = float(np.dot(to_horse, outward))

    wall_limit = TRACK_HALF_WIDTH - HORSE_HALF_WIDTH
    if lateral > wall_limit:
        depth = lateral - wall_limit
        _resolve_wall_impulse(body, -outward, depth)
    elif lateral < -wall_limit:
        depth = -wall_limit - lateral
        _resolve_wall_impulse(body, outward, depth)


def _resolve_curve_walls(body: HorseBody, seg: CurveSegment) -> None:
    center = _vec2(*seg.center)
    to_horse = body.position - center
    dist = float(np.linalg.norm(to_horse))
    if dist < 1e-6:
        return

    normal = to_horse / dist

    outer_limit = seg.radius + TRACK_HALF_WIDTH - HORSE_HALF_WIDTH
    if dist > outer_limit:
        depth = dist - outer_limit
        _resolve_wall_impulse(body, -normal, depth)

    inner_limit = seg.radius - TRACK_HALF_WIDTH + HORSE_HALF_WIDTH
    if inner_limit > 0 and dist < inner_limit:
        depth = inner_limit - dist
        _resolve_wall_impulse(body, normal, depth)
