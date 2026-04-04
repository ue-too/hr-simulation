"""Physics engine — action-to-force translation, integration, collision, drafting, slope.

The jockey provides high-level intent (effort + lane). The horse's body determines
the physical response. Movement smoothing creates organic feel.
"""

from __future__ import annotations

import math

import numpy as np

from horse_racing.core.horse import HorseProfile
from horse_racing.core.stamina import StaminaState, apply_fatigue, compute_drain
from horse_racing.core.types import (
    DRAFT_DISTANCE,
    DRAFT_SPEED_BONUS,
    FATIGUE_WOBBLE_SCALE,
    HORSE_HALF_LENGTH,
    HORSE_HALF_WIDTH,
    NORMAL_DAMP,
    RESPONSE_TAU,
    STRIDE_AMPLITUDE,
    STRIDE_FREQUENCY,
    WALL_RESTITUTION,
    CurveSegment,
    HorseBody,
    JockeyAction,
    StraightSegment,
    TrackFrame,
    TrackSegment,
)


def _vec2(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=np.float64)


# ---------------------------------------------------------------------------
# Action-to-force translation
# ---------------------------------------------------------------------------


def compute_target_speed(
    action: JockeyAction,
    profile: HorseProfile,
) -> float:
    """Convert effort level to a target speed based on horse attributes."""
    eff_speed = profile.efficiency_speed
    if action.effort >= 0:
        # 0 = cruise at efficiency_speed, 1 = push to top_speed
        return eff_speed + action.effort * (profile.top_speed - eff_speed)
    else:
        # -1 = ease up to ~50% of efficiency_speed
        return eff_speed * (1.0 + action.effort * 0.5)


def compute_forward_force(
    current_speed: float,
    target_speed: float,
    profile: HorseProfile,
) -> float:
    """Compute forward force to reach target speed."""
    speed_diff = target_speed - current_speed
    return profile.acceleration * speed_diff * profile.weight


def compute_lateral_force(
    action: JockeyAction,
    profile: HorseProfile,
    frame: TrackFrame,
) -> float:
    """Compute lateral force from lane action + horse cornering grip."""
    # Select left or right grip based on turn direction
    if frame.turn_direction > 0:
        grip = profile.cornering_grip_left
    elif frame.turn_direction < 0:
        grip = profile.cornering_grip_right
    else:
        # Straight: use average of both grips for lane changes
        grip = (profile.cornering_grip_left + profile.cornering_grip_right) / 2

    return action.lane * grip * profile.weight * 0.5


# ---------------------------------------------------------------------------
# Movement smoothing
# ---------------------------------------------------------------------------


def smooth_force(
    body: HorseBody,
    raw_forward: float,
    raw_lateral: float,
    profile: HorseProfile,
    dt: float,
) -> tuple[float, float]:
    """Apply exponential moving average to forces for response lag.

    Higher acceleration horses respond faster (shorter effective tau).
    """
    tau = RESPONSE_TAU / profile.acceleration  # fast horses respond quicker
    alpha = min(1.0, dt / tau) if tau > 0 else 1.0

    body.smoothed_forward_force += alpha * (raw_forward - body.smoothed_forward_force)
    body.smoothed_lateral_force += alpha * (raw_lateral - body.smoothed_lateral_force)

    return body.smoothed_forward_force, body.smoothed_lateral_force


def stride_oscillation(time: float) -> float:
    """Subtle forward speed oscillation simulating gallop rhythm."""
    return STRIDE_AMPLITUDE * math.sin(2.0 * math.pi * STRIDE_FREQUENCY * time)


def fatigue_wobble(stamina: StaminaState, time: float) -> float:
    """Lateral drift when exhausted. Returns lateral force perturbation."""
    if not stamina.is_fatigued:
        return 0.0
    # Scale wobble by how far below fatigue threshold
    severity = 1.0 - stamina.ratio / 0.3
    # Low-frequency Perlin-like wobble using two sine waves
    wobble = (
        math.sin(1.7 * time) * 0.6
        + math.sin(3.1 * time + 0.8) * 0.4
    )
    return FATIGUE_WOBBLE_SCALE * severity * wobble


# ---------------------------------------------------------------------------
# Slope physics
# ---------------------------------------------------------------------------

GRAVITY: float = 9.81


def slope_force(
    profile: HorseProfile,
    slope: float,
) -> float:
    """Force from slope. Uphill = negative (slows down), downhill = positive.

    climbing_power scales the penalty: high climbing_power reduces uphill cost.
    """
    if abs(slope) < 1e-6:
        return 0.0
    # Force = -mass * g * sin(slope_angle) / climbing_power
    # For small slopes, sin(atan(slope)) ≈ slope / sqrt(1 + slope^2)
    slope_sin = slope / math.sqrt(1.0 + slope * slope)
    return -profile.weight * GRAVITY * slope_sin / profile.climbing_power


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------


def compute_drafting_bonus(
    horse_idx: int,
    bodies: list[HorseBody],
    frames: list[TrackFrame],
    progresses: list[float],
    total_length: float,
) -> float:
    """Check if horse is drafting behind another. Returns speed bonus (m/s)."""
    my_progress = progresses[horse_idx]
    my_pos = bodies[horse_idx].position

    for j, other_body in enumerate(bodies):
        if j == horse_idx:
            continue
        # Must be ahead
        if progresses[j] <= my_progress:
            continue
        # Must be within draft distance
        dx = float(other_body.position[0] - my_pos[0])
        dy = float(other_body.position[1] - my_pos[1])
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < DRAFT_DISTANCE:
            # Closer = more benefit, linear falloff
            benefit = DRAFT_SPEED_BONUS * (1.0 - dist / DRAFT_DISTANCE)
            return benefit

    return 0.0


# ---------------------------------------------------------------------------
# Centripetal force for curves
# ---------------------------------------------------------------------------


def centripetal_force(
    body: HorseBody,
    frame: TrackFrame,
    mass: float,
) -> np.ndarray:
    """Auto-apply centripetal acceleration to keep horse on curved track."""
    if frame.turn_radius == float("inf") or frame.turn_radius < 1e-6:
        return _vec2(0.0, 0.0)

    # Tangential speed
    tang_speed = float(np.dot(body.velocity, frame.tangential))

    # Required centripetal acceleration: v^2 / r, directed inward (toward center)
    centripetal_accel = tang_speed * tang_speed / frame.turn_radius
    # Inward direction = -normal (normal points outward)
    return -frame.normal * centripetal_accel * mass


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def integrate(body: HorseBody, mass: float, dt: float) -> None:
    """Semi-implicit Euler: velocity from force, then position from velocity."""
    body.velocity += (body.force / mass) * dt
    body.position += body.velocity * dt


# ---------------------------------------------------------------------------
# Collision detection and resolution (reused from v1)
# ---------------------------------------------------------------------------


def _obb_corners(pos: np.ndarray, orientation: float,
                 half_l: float, half_w: float) -> np.ndarray:
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
    c = math.cos(orientation)
    s = math.sin(orientation)
    return [np.array([c, s]), np.array([-s, c])]


def _project(corners: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    dots = corners @ axis
    return float(dots.min()), float(dots.max())


def _obb_overlap(
    pos_a: np.ndarray, ori_a: float,
    pos_b: np.ndarray, ori_b: float,
    half_l: float = HORSE_HALF_LENGTH,
    half_w: float = HORSE_HALF_WIDTH,
) -> tuple[np.ndarray, float] | None:
    corners_a = _obb_corners(pos_a, ori_a, half_l, half_w)
    corners_b = _obb_corners(pos_b, ori_b, half_l, half_w)

    axes = _obb_axes(ori_a) + _obb_axes(ori_b)

    min_overlap = float("inf")
    min_axis = None

    for axis in axes:
        min_a, max_a = _project(corners_a, axis)
        min_b, max_b = _project(corners_b, axis)

        if max_a < min_b or max_b < min_a:
            return None

        overlap = min(max_a - min_b, max_b - min_a)
        if overlap < min_overlap:
            min_overlap = overlap
            min_axis = axis

    if min_axis is None:
        return None

    delta = pos_b - pos_a
    if float(np.dot(min_axis, delta)) < 0:
        min_axis = -min_axis

    return (min_axis, min_overlap)


def resolve_horse_collisions(
    bodies: list[HorseBody],
    masses: list[float],
) -> list[bool]:
    """Resolve OBB collisions between all horse pairs.

    Weight determines collision impulse — heavy horses push light horses.
    """
    n = len(bodies)
    collided = [False] * n

    max_extent_2 = (HORSE_HALF_LENGTH + HORSE_HALF_WIDTH) * 2

    for i in range(n):
        for j in range(i + 1, n):
            dx = float(bodies[j].position[0] - bodies[i].position[0])
            dy = float(bodies[j].position[1] - bodies[i].position[1])
            if abs(dx) > max_extent_2 or abs(dy) > max_extent_2:
                continue
            if math.sqrt(dx * dx + dy * dy) > max_extent_2:
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

            # Position correction: 50/50 split
            bodies[i].position -= normal * (overlap / 2)
            bodies[j].position += normal * (overlap / 2)

            # Velocity impulse using actual mass
            rel_vel = bodies[i].velocity - bodies[j].velocity
            vel_along_normal = float(np.dot(rel_vel, normal))
            e = 0.4
            inv_mass_i = 1.0 / masses[i]
            inv_mass_j = 1.0 / masses[j]
            j_impulse = -(1 + e) * vel_along_normal / (inv_mass_i + inv_mass_j)
            bodies[i].velocity += normal * (j_impulse * inv_mass_i)
            bodies[j].velocity -= normal * (j_impulse * inv_mass_j)

    return collided


# ---------------------------------------------------------------------------
# Wall collision
# ---------------------------------------------------------------------------


def _detect_straight_wall(
    body: HorseBody, seg: StraightSegment, half_width: float,
) -> tuple[np.ndarray, float] | None:
    fx = seg.end_point[0] - seg.start_point[0]
    fy = seg.end_point[1] - seg.start_point[1]
    length = math.sqrt(fx * fx + fy * fy)
    if length < 1e-6:
        return None
    inv_len = 1.0 / length
    fx *= inv_len
    fy *= inv_len
    ox, oy = fy, -fx

    thx = float(body.position[0]) - seg.start_point[0]
    thy = float(body.position[1]) - seg.start_point[1]
    lateral = thx * ox + thy * oy

    wall_limit = half_width - HORSE_HALF_WIDTH
    if lateral > wall_limit:
        return (_vec2(-ox, -oy), lateral - wall_limit)
    elif lateral < -wall_limit:
        return (_vec2(ox, oy), -wall_limit - lateral)
    return None


def _detect_curve_wall(
    body: HorseBody, seg: CurveSegment, half_width: float,
) -> tuple[np.ndarray, float] | None:
    tx = float(body.position[0]) - seg.center[0]
    ty = float(body.position[1]) - seg.center[1]
    dist = math.sqrt(tx * tx + ty * ty)
    if dist < 1e-6:
        return None

    inv_dist = 1.0 / dist
    nx, ny = tx * inv_dist, ty * inv_dist

    outer_limit = seg.radius + half_width - HORSE_HALF_WIDTH
    if dist > outer_limit:
        return (_vec2(-nx, -ny), dist - outer_limit)

    inner_limit = seg.radius - half_width + HORSE_HALF_WIDTH
    if inner_limit > 0 and dist < inner_limit:
        return (_vec2(nx, ny), inner_limit - dist)

    return None


def _resolve_wall_impulse(body: HorseBody, normal: np.ndarray, depth: float) -> None:
    body.position += normal * depth
    vel_along_normal = float(np.dot(body.velocity, normal))
    body.velocity += normal * (-(1 + WALL_RESTITUTION) * vel_along_normal)


def resolve_wall_collisions(
    bodies: list[HorseBody],
    segments: list[TrackSegment],
    segment_indices: list[int],
    half_width: float = None,
) -> None:
    """Resolve wall collisions for all horses."""
    from horse_racing.core.types import TRACK_HALF_WIDTH
    if half_width is None:
        half_width = TRACK_HALF_WIDTH

    for idx, body in enumerate(bodies):
        seg = segments[segment_indices[idx]]
        if isinstance(seg, StraightSegment):
            hit = _detect_straight_wall(body, seg, half_width)
        else:
            hit = _detect_curve_wall(body, seg, half_width)
        if hit is not None:
            _resolve_wall_impulse(body, hit[0], hit[1])


# ---------------------------------------------------------------------------
# Orientation update
# ---------------------------------------------------------------------------


def update_orientation(body: HorseBody) -> None:
    """Set horse orientation to face velocity direction."""
    vx, vy = float(body.velocity[0]), float(body.velocity[1])
    speed = math.sqrt(vx * vx + vy * vy)
    if speed > 0.1:
        body.orientation = math.atan2(vy, vx)


# ---------------------------------------------------------------------------
# Full physics step for one horse
# ---------------------------------------------------------------------------


def step_horse(
    body: HorseBody,
    profile: HorseProfile,
    stamina: StaminaState,
    action: JockeyAction,
    frame: TrackFrame,
    dt: float,
    sim_time: float,
    draft_bonus: float = 0.0,
) -> tuple[float, bool]:
    """Execute one physics substep for a single horse.

    Returns:
        (cornering_excess, wall_collided): cornering excess force for stamina,
        and whether the horse hit a wall.
    """
    body.clear_force()

    # 1. Compute target speed from jockey effort + drafting bonus
    target_speed = compute_target_speed(action, profile)
    target_speed += draft_bonus * profile.top_speed

    # 2. Decompose current velocity into tangential and normal components
    tang_speed = float(np.dot(body.velocity, frame.tangential))
    norm_speed = float(np.dot(body.velocity, frame.normal))

    # 3. Compute raw forces
    raw_forward = compute_forward_force(tang_speed, target_speed, profile)
    raw_lateral = compute_lateral_force(action, profile, frame)

    # 4. Apply fatigue scaling
    raw_forward, raw_lateral = apply_fatigue(stamina, raw_forward, raw_lateral)

    # 5. Smooth forces (response lag)
    smooth_fwd, smooth_lat = smooth_force(body, raw_forward, raw_lateral, profile, dt)

    # 6. Apply forward force along tangential direction
    body.apply_force(frame.tangential * smooth_fwd)

    # 7. Apply lateral force along normal direction
    body.apply_force(frame.normal * smooth_lat)

    # 8. Add stride oscillation (cosmetic)
    stride = stride_oscillation(sim_time)
    body.apply_force(frame.tangential * stride * profile.weight)

    # 9. Add fatigue wobble
    wobble = fatigue_wobble(stamina, sim_time)
    body.apply_force(frame.normal * wobble * profile.weight)

    # 10. Centripetal force for curves
    body.apply_force(centripetal_force(body, frame, profile.weight))

    # 11. Slope force
    body.apply_force(frame.tangential * slope_force(profile, frame.slope))

    # 12. Lateral damping
    body.apply_force(frame.normal * (-NORMAL_DAMP * norm_speed * profile.weight))

    # 13. Integrate
    integrate(body, profile.weight, dt)

    # 14. Update orientation
    update_orientation(body)

    # 15. Compute cornering excess for stamina drain
    cornering_excess = 0.0
    if frame.turn_radius < float("inf"):
        required_centripetal = tang_speed * tang_speed / frame.turn_radius
        grip = (
            profile.cornering_grip_left
            if frame.turn_direction > 0
            else profile.cornering_grip_right
        )
        max_grip_force = grip * GRAVITY
        cornering_excess = max(0.0, required_centripetal - max_grip_force)

    return cornering_excess, False
