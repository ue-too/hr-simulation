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
    flx, fly = c * half_l, s * half_l
    swx, swy = -s * half_w, c * half_w
    px, py = float(pos[0]), float(pos[1])
    out = np.empty((4, 2), dtype=np.float64)
    out[0, 0] = px + flx + swx; out[0, 1] = py + fly + swy
    out[1, 0] = px + flx - swx; out[1, 1] = py + fly - swy
    out[2, 0] = px - flx - swx; out[2, 1] = py - fly - swy
    out[3, 0] = px - flx + swx; out[3, 1] = py - fly + swy
    return out


def _obb_axes(orientation: float) -> list[np.ndarray]:
    """Return 2 unique edge normals (separating axes) for an OBB."""
    c = math.cos(orientation)
    s = math.sin(orientation)
    return [np.array([c, s]), np.array([-s, c])]


def _project(corners: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    """Project corners onto axis, return (min, max)."""
    ax, ay = float(axis[0]), float(axis[1])
    d0 = corners[0, 0] * ax + corners[0, 1] * ay
    d1 = corners[1, 0] * ax + corners[1, 1] * ay
    d2 = corners[2, 0] * ax + corners[2, 1] * ay
    d3 = corners[3, 0] * ax + corners[3, 1] * ay
    lo = d0; hi = d0
    if d1 < lo: lo = d1
    elif d1 > hi: hi = d1
    if d2 < lo: lo = d2
    elif d2 > hi: hi = d2
    if d3 < lo: lo = d3
    elif d3 > hi: hi = d3
    return lo, hi


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

    max_extent_2 = (HORSE_HALF_LENGTH + HORSE_HALF_WIDTH) * 2

    for i in range(n):
        for j in range(i + 1, n):
            # Quick distance check before expensive SAT
            dx = float(bodies[j].position[0] - bodies[i].position[0])
            dy = float(bodies[j].position[1] - bodies[i].position[1])
            if abs(dx) > max_extent_2 or abs(dy) > max_extent_2:
                continue
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > max_extent_2:
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
    max_extent_2 = (HORSE_HALF_LENGTH + HORSE_HALF_WIDTH) * 2
    for i in range(n):
        for j in range(i + 1, n):
            dx = float(bodies[j].position[0] - bodies[i].position[0])
            dy = float(bodies[j].position[1] - bodies[i].position[1])
            if abs(dx) > max_extent_2 or abs(dy) > max_extent_2:
                continue
            if math.sqrt(dx * dx + dy * dy) < max_extent_2:
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
    fx = seg.end_point[0] - seg.start_point[0]
    fy = seg.end_point[1] - seg.start_point[1]
    length = math.sqrt(fx * fx + fy * fy)
    if length < 1e-6:
        return None
    inv_len = 1.0 / length
    fx *= inv_len
    fy *= inv_len
    # outward = (fy, -fx)
    ox, oy = fy, -fx

    thx = float(body.position[0]) - seg.start_point[0]
    thy = float(body.position[1]) - seg.start_point[1]
    lateral = thx * ox + thy * oy

    wall_limit = TRACK_HALF_WIDTH - HORSE_HALF_WIDTH
    if lateral > wall_limit:
        return (_vec2(-ox, -oy), lateral - wall_limit)
    elif lateral < -wall_limit:
        return (_vec2(ox, oy), -wall_limit - lateral)
    return None


def _detect_curve_wall(
    body: HorseBody, seg: CurveSegment
) -> tuple[np.ndarray, float] | None:
    tx = float(body.position[0]) - seg.center[0]
    ty = float(body.position[1]) - seg.center[1]
    dist = math.sqrt(tx * tx + ty * ty)
    if dist < 1e-6:
        return None

    inv_dist = 1.0 / dist
    nx, ny = tx * inv_dist, ty * inv_dist

    outer_limit = seg.radius + TRACK_HALF_WIDTH - HORSE_HALF_WIDTH
    if dist > outer_limit:
        return (_vec2(-nx, -ny), dist - outer_limit)

    inner_limit = seg.radius - TRACK_HALF_WIDTH + HORSE_HALF_WIDTH
    if inner_limit > 0 and dist < inner_limit:
        return (_vec2(nx, ny), inner_limit - dist)

    return None


# ---------------------------------------------------------------------------
# Rail-based wall collision — point vs explicit rail paths
# ---------------------------------------------------------------------------


def _nearest_point_on_straight(
    pos: np.ndarray, seg: StraightSegment
) -> tuple[np.ndarray, float]:
    """Find nearest point on a straight segment to pos. Returns (point, distance)."""
    px, py = float(pos[0]), float(pos[1])
    sx, sy = seg.start_point
    ex, ey = seg.end_point
    abx, aby = ex - sx, ey - sy
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < 1e-12:
        dx, dy = px - sx, py - sy
        return _vec2(sx, sy), math.sqrt(dx * dx + dy * dy)

    t = ((px - sx) * abx + (py - sy) * aby) / ab_len_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nx, ny = sx + abx * t, sy + aby * t
    dx, dy = px - nx, py - ny
    return _vec2(nx, ny), math.sqrt(dx * dx + dy * dy)


def _nearest_point_on_curve(
    pos: np.ndarray, seg: CurveSegment
) -> tuple[np.ndarray, float]:
    """Find nearest point on a curve segment arc to pos. Returns (point, distance)."""
    px, py = float(pos[0]), float(pos[1])
    cx, cy = seg.center
    tpx, tpy = px - cx, py - cy
    dist_to_center = math.sqrt(tpx * tpx + tpy * tpy)

    if dist_to_center < 1e-6:
        sx, sy = seg.start_point
        dx, dy = px - sx, py - sy
        return _vec2(sx, sy), math.sqrt(dx * dx + dy * dy)

    pos_angle = math.atan2(tpy, tpx)

    start_angle = math.atan2(
        seg.start_point[1] - cy,
        seg.start_point[0] - cx,
    )

    span = seg.angle_span
    diff = pos_angle - start_angle
    # Normalize to [-pi, pi]
    if diff > math.pi:
        diff -= 2 * math.pi
    elif diff < -math.pi:
        diff += 2 * math.pi

    if span >= 0:
        if diff < 0:
            diff += 2 * math.pi
        in_arc = diff <= span
    else:
        if diff > 0:
            diff -= 2 * math.pi
        in_arc = diff >= span

    if in_arc:
        ratio = seg.radius / dist_to_center
        nx, ny = cx + tpx * ratio, cy + tpy * ratio
        return _vec2(nx, ny), abs(dist_to_center - seg.radius)
    else:
        sx, sy = seg.start_point
        ex, ey = seg.end_point
        dsx, dsy = px - sx, py - sy
        dex, dey = px - ex, py - ey
        d_start = math.sqrt(dsx * dsx + dsy * dsy)
        d_end = math.sqrt(dex * dex + dey * dey)
        if d_start <= d_end:
            return _vec2(sx, sy), d_start
        return _vec2(ex, ey), d_end


def _normalize_angle(a: float) -> float:
    """Normalize angle to [-pi, pi]."""
    if a > math.pi:
        a -= 2 * math.pi
    elif a < -math.pi:
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
    best_nx = 1.0
    best_ny = 0.0
    found = False
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
                found = True
                if dist > 1e-6:
                    inv_dist = 1.0 / dist
                    best_nx = (px - float(nearest[0])) * inv_dist
                    best_ny = (py - float(nearest[1])) * inv_dist
                else:
                    best_nx, best_ny = 1.0, 0.0

    if found:
        return (_vec2(best_nx, best_ny), best_depth)
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
    fx = seg.end_point[0] - seg.start_point[0]
    fy = seg.end_point[1] - seg.start_point[1]
    length = math.sqrt(fx * fx + fy * fy)
    if length < 1e-6:
        return
    inv_len = 1.0 / length
    fx *= inv_len
    fy *= inv_len
    ox, oy = fy, -fx

    thx = float(body.position[0]) - seg.start_point[0]
    thy = float(body.position[1]) - seg.start_point[1]
    lateral = thx * ox + thy * oy

    wall_limit = TRACK_HALF_WIDTH - HORSE_HALF_WIDTH
    if lateral > wall_limit:
        _resolve_wall_impulse(body, _vec2(-ox, -oy), lateral - wall_limit)
    elif lateral < -wall_limit:
        _resolve_wall_impulse(body, _vec2(ox, oy), -wall_limit - lateral)


def _resolve_curve_walls(body: HorseBody, seg: CurveSegment) -> None:
    tx = float(body.position[0]) - seg.center[0]
    ty = float(body.position[1]) - seg.center[1]
    dist = math.sqrt(tx * tx + ty * ty)
    if dist < 1e-6:
        return

    inv_dist = 1.0 / dist
    nx, ny = tx * inv_dist, ty * inv_dist

    outer_limit = seg.radius + TRACK_HALF_WIDTH - HORSE_HALF_WIDTH
    if dist > outer_limit:
        _resolve_wall_impulse(body, _vec2(-nx, -ny), dist - outer_limit)

    inner_limit = seg.radius - TRACK_HALF_WIDTH + HORSE_HALF_WIDTH
    if inner_limit > 0 and dist < inner_limit:
        _resolve_wall_impulse(body, _vec2(nx, ny), inner_limit - dist)
