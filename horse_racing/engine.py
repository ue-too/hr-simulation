"""HorseRacingEngine — core simulation stepping the race forward."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from horse_racing.attributes import CoreAttributes, resolve_effective
from horse_racing.genome import (
    HorseGenome,
    default_genome,
    express_genome,
    modifier_is_present,
    modifier_strength,
)
from horse_racing.modifiers import (
    ActiveModifier, MODIFIER_IDS, MODIFIER_REGISTRY, ModifierContext,
    SKILL_MODIFIER_IDS,
)
from horse_racing.physics import integrate, resolve_all_collisions, resolve_horse_collisions, resolve_wall_collisions
from horse_racing.stamina import HorseRuntimeState, apply_exhaustion, update_stamina
from horse_racing.track import compute_rail_bboxes, compute_segment_length, load_track
from horse_racing.track_navigator import TrackNavigator
from horse_racing.types import (
    HORSE_COUNT,
    HORSE_SPACING,
    MAX_REL_HORSES,

    NORMAL_DAMP,
    PHYS_HZ,
    PHYS_SUBSTEPS,
    SKILL_IDS,
    TRACK_HALF_WIDTH,
    CurveSegment,
    HorseAction,
    HorseBody,
    TrackFrame,
    TrackSegment,
)


def _vec2(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=np.float64)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return _vec2(1.0, 0.0)
    return v / n


@dataclass
class HorseState:
    """Full per-horse state."""

    body: HorseBody = field(default_factory=HorseBody)
    navigator: TrackNavigator | None = None
    genome: HorseGenome = field(default_factory=default_genome)
    base_attrs: CoreAttributes = field(default_factory=CoreAttributes)
    effective_attrs: CoreAttributes = field(default_factory=CoreAttributes)
    runtime: HorseRuntimeState = field(default_factory=HorseRuntimeState)
    frame: TrackFrame | None = None
    track_progress: float = 0.0
    finished: bool = False
    collision_this_tick: bool = False


@dataclass
class EngineConfig:
    horse_count: int = HORSE_COUNT
    track_surface: str = "dry"


class HorseRacingEngine:
    """Core simulation engine."""

    def __init__(
        self,
        track_path: str | Path,
        config: EngineConfig | None = None,
        genomes: list[HorseGenome] | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.track_data = load_track(track_path)
        self.segments: list[TrackSegment] = self.track_data.segments
        self.inner_rails = self.track_data.inner_rails
        self.outer_rails = self.track_data.outer_rails
        self.inner_bboxes = compute_rail_bboxes(self.inner_rails)
        self.outer_bboxes = compute_rail_bboxes(self.outer_rails)
        self.dt = 1.0 / PHYS_HZ
        self.tick: int = 0
        self.horse_count = self.config.horse_count

        self.active_skills: set[str] = set()
        self._finish_order: list[int] = []
        self._finish_count: int = 0

        # Initialize horses
        self.horses: list[HorseState] = []
        for i in range(self.horse_count):
            hs = HorseState()
            hs.navigator = TrackNavigator(self.segments)
            if genomes and i < len(genomes):
                hs.genome = genomes[i]
            hs.base_attrs = express_genome(hs.genome)
            hs.effective_attrs = CoreAttributes(**hs.base_attrs.to_dict())
            hs.runtime = HorseRuntimeState(
                current_stamina=hs.base_attrs.stamina,
                base_attributes=hs.base_attrs,
            )
            self.horses.append(hs)

        self._finish_order = [0] * self.horse_count
        self._place_horses()

    def reset(self, genomes: list[HorseGenome] | None = None) -> None:
        self.tick = 0
        self._finish_order: list[int] = [0] * self.horse_count  # 0 = not finished
        self._finish_count: int = 0
        for i, hs in enumerate(self.horses):
            if genomes and i < len(genomes):
                hs.genome = genomes[i]
            hs.base_attrs = express_genome(hs.genome)
            hs.effective_attrs = CoreAttributes(**hs.base_attrs.to_dict())
            hs.runtime = HorseRuntimeState(
                current_stamina=hs.base_attrs.stamina,
                base_attributes=hs.base_attrs,
            )
            hs.body = HorseBody()
            hs.navigator = TrackNavigator(self.segments)
            hs.frame = None
            hs.track_progress = 0.0
            hs.finished = False
            hs.collision_this_tick = False
        self._place_horses()

    def _place_horses(self) -> None:
        """Place horses at the start of the first segment."""
        seg = self.segments[0]
        start = _vec2(*seg.start_point)

        if hasattr(seg, "end_point"):
            fwd = _normalize(_vec2(*seg.end_point) - _vec2(*seg.start_point))
        else:
            fwd = _vec2(1.0, 0.0)

        # Outward = forward rotated -90 degrees
        outward = _vec2(fwd[1], -fwd[0])

        # Spawn horses from the inner rail outward.
        # outward points toward the outer rail, so the innermost
        # position is at the most negative lateral offset.
        n = len(self.horses)
        # Start 1 spacing inward from the inner rail, then spread outward
        inner_edge = -(TRACK_HALF_WIDTH - HORSE_SPACING)
        for i, hs in enumerate(self.horses):
            lateral = inner_edge + HORSE_SPACING * i
            hs.body.position = start + outward * lateral
            hs.body.velocity[:] = 0.0
            hs.body.clear_force()

            # Orient each horse along the track tangent at its position
            frame = hs.navigator.update(hs.body.position)
            hs.body.orientation = math.atan2(frame.tangential[1], frame.tangential[0])

    def stagger_horses(self, offsets: list[float]) -> None:
        """Move horses forward along the track centerline by the given distances.

        Each offset (in meters) advances the corresponding horse from its
        current position. Used to create staggered starts for overtake training.
        """
        for i, (hs, offset) in enumerate(zip(self.horses, offsets)):
            if offset <= 0:
                continue
            # Compute current distance along track, then advance
            progress = hs.navigator.compute_progress(hs.body.position)
            current_dist = progress * hs.navigator.total_length
            new_pos = hs.navigator.position_at_distance(current_dist + offset)
            # Keep the same lateral offset from centerline
            old_frame = hs.navigator.compute_frame(hs.body.position)
            delta_to_center = hs.body.position - new_pos
            # Preserve lateral displacement
            hs.body.position = new_pos + old_frame.normal * float(
                np.dot(delta_to_center, old_frame.normal)
            )
            # Update navigator and orientation
            frame = hs.navigator.update(hs.body.position)
            hs.body.orientation = math.atan2(frame.tangential[1], frame.tangential[0])

    def step(self, actions: list[HorseAction]) -> None:
        """Advance the simulation by one game tick (PHYS_SUBSTEPS physics steps)."""
        self.tick += 1

        # Pad actions if fewer than horse_count
        while len(actions) < self.horse_count:
            actions.append(HorseAction())

        # 1. Resolve effective attributes (once per tick, before substeps)
        self._resolve_attributes()

        # 2. Update stamina (once per tick)
        for i, hs in enumerate(self.horses):
            if hs.frame is None:
                hs.frame = hs.navigator.update(hs.body.position)
            tangential_vel = float(np.dot(hs.body.velocity, hs.frame.tangential))
            normal_vel_val = float(np.dot(hs.body.velocity, hs.frame.normal))
            speed = float(np.linalg.norm(hs.body.velocity))
            extra_t = actions[i].extra_tangential * hs.effective_attrs.forward_accel
            extra_n = actions[i].extra_normal * hs.effective_attrs.turn_accel
            update_stamina(
                hs.runtime,
                hs.effective_attrs,
                extra_t,
                extra_n,
                speed,
                tangential_vel,
                normal_vel_val,
                hs.frame.turn_radius,
            )
            # Re-apply exhaustion after stamina update
            hs.effective_attrs = apply_exhaustion(
                hs.effective_attrs,
                hs.runtime.current_stamina,
                hs.base_attrs.stamina,
            )

        # 3. Physics substeps
        # JS engine order per substep: apply force → world.step()
        # world.step() does: collision resolution → integration → clear forces
        for _sub in range(PHYS_SUBSTEPS):
            # a. Recompute track frame and apply forces
            for i, hs in enumerate(self.horses):
                if hs.finished:
                    hs.body.clear_force()
                    continue
                hs.frame = hs.navigator.update(hs.body.position)
                hs.body.clear_force()
                force = self._compute_force(hs, actions[i])
                hs.body.apply_force(force)

            # b. Collision resolution (before integration, matching JS)
            bodies = [hs.body for hs in self.horses]
            masses = [hs.effective_attrs.weight for hs in self.horses]
            pushing_powers = [hs.effective_attrs.pushing_power for hs in self.horses]
            push_resistances = [hs.effective_attrs.push_resistance for hs in self.horses]
            collided = resolve_horse_collisions(bodies, masses, pushing_powers, push_resistances)
            resolve_wall_collisions(
                bodies,
                self.segments,
                [hs.navigator.segment_index for hs in self.horses],
                inner_rails=self.inner_rails,
                outer_rails=self.outer_rails,
                inner_bboxes=self.inner_bboxes,
                outer_bboxes=self.outer_bboxes,
            )

            for i, hs in enumerate(self.horses):
                hs.collision_this_tick = hs.collision_this_tick or collided[i]

            # c. Integration + clear forces
            for hs in self.horses:
                integrate(hs.body, hs.effective_attrs.weight, self.dt)
                hs.body.clear_force()
                # Clamp: horses cannot reverse
                if hs.frame is not None:
                    tang_vel = float(np.dot(hs.body.velocity, hs.frame.tangential))
                    if tang_vel < 0:
                        hs.body.velocity -= tang_vel * hs.frame.tangential

        # 4. Post-step: update navigators, progress, orientation
        for i, hs in enumerate(self.horses):
            hs.frame = hs.navigator.update(hs.body.position)
            hs.track_progress = hs.navigator.compute_progress(hs.body.position)
            hs.body.orientation = math.atan2(hs.frame.tangential[1], hs.frame.tangential[0])

            # Check finish — matches TS completedLap (segment exit geometry)
            if not hs.finished and hs.navigator.completed_lap:
                hs.finished = True
                hs.body.velocity[:] = 0.0
                self._finish_count += 1
                self._finish_order[i] = self._finish_count

    def _resolve_attributes(self) -> None:
        """Resolve modifiers and compute effective attributes for all horses."""
        positions = [
            (float(hs.body.position[0]), float(hs.body.position[1])) for hs in self.horses
        ]
        velocities = [
            (float(hs.body.velocity[0]), float(hs.body.velocity[1])) for hs in self.horses
        ]
        progresses = [hs.track_progress for hs in self.horses]

        for i, hs in enumerate(self.horses):
            ctx = ModifierContext(
                horse_index=i,
                positions=positions,
                velocities=velocities,
                track_progress=progresses,
                current_stamina=hs.runtime.current_stamina,
                max_stamina=hs.base_attrs.stamina,
                track_surface=self.config.track_surface,
                active_skills=self.active_skills if i == 0 else set(),
            )

            active: list[ActiveModifier] = []
            for mod_id, (pres_gene, str_gene) in hs.genome.modifiers.items():
                if mod_id not in MODIFIER_REGISTRY:
                    continue
                if not modifier_is_present(pres_gene):
                    continue
                defn = MODIFIER_REGISTRY[mod_id]
                if defn.condition(ctx):
                    strength = modifier_strength(str_gene)
                    active.append(ActiveModifier(id=mod_id, strength=strength))

            # Skill-based physics modifiers (horse 0 / trainee only)
            if i == 0 and self.active_skills:
                for skill_mod_id in SKILL_MODIFIER_IDS:
                    defn = MODIFIER_REGISTRY[skill_mod_id]
                    if defn.condition(ctx):
                        active.append(ActiveModifier(id=skill_mod_id, strength=1.0))

            hs.runtime.active_modifiers = active
            hs.effective_attrs = resolve_effective(hs.base_attrs, active)

    def _compute_force(self, hs: HorseState, action: HorseAction) -> np.ndarray:
        """Compute the total force on a horse given its state and action."""
        frame = hs.frame
        eff = hs.effective_attrs
        velocity = hs.body.velocity

        tangential_vel = float(np.dot(velocity, frame.tangential))
        normal_vel = float(np.dot(velocity, frame.normal))

        # Centripetal acceleration (only on curves)
        if frame.turn_radius < 1e6:
            centripetal = tangential_vel**2 / frame.turn_radius
        else:
            centripetal = 0.0

        # Auto-cruise spring: always active, pulls toward cruise speed.
        # Jockey input stacks on top — at steady state the spring balances
        # the push, giving vel = cruise + action * fwd_accel.
        speed_change = eff.cruise_speed - tangential_vel

        extra_tangential = action.extra_tangential * eff.forward_accel
        extra_normal = action.extra_normal * eff.turn_accel

        tangential_accel = speed_change + extra_tangential
        if tangential_vel >= eff.max_speed and tangential_accel > 0:
            tangential_accel = 0.0
        if tangential_vel <= 0.0 and tangential_accel < 0:
            tangential_accel = 0.0

        # Slope gravity: uphill decelerates, downhill accelerates
        if frame.slope != 0.0:
            slope_angle = math.atan(frame.slope)
            tangential_accel += -9.81 * math.sin(slope_angle)

        normal_accel = -centripetal - normal_vel * NORMAL_DAMP + extra_normal

        total_accel = tangential_accel * frame.tangential + normal_accel * frame.normal
        return total_accel * eff.weight  # F = ma

    def get_observations(self) -> list[dict]:
        """Return a list of observation dicts, one per horse."""
        placements = self.get_placements()
        num_horses = len(self.horses)

        obs_list = []
        for i, hs in enumerate(self.horses):
            frame = hs.frame
            if frame is None:
                frame = hs.navigator.compute_frame(hs.body.position)

            tangential_vel = float(np.dot(hs.body.velocity, frame.tangential))
            normal_vel = float(np.dot(hs.body.velocity, frame.normal))
            if frame.turn_radius < 1e6:
                displacement = frame.turn_radius - frame.target_radius
            else:
                seg = self.segments[hs.navigator.segment_index]
                start = _vec2(*seg.start_point)
                displacement = float(np.dot(hs.body.position - start, frame.normal))
            curvature = 1.0 / frame.turn_radius if frame.turn_radius < 1e6 else 0.0

            # Upcoming curvature: scan ahead for next curve segment
            next_curvature = 0.0
            dist_to_next_curve = 0.0
            lookahead = hs.navigator.lookahead_segments(5)
            accumulated_dist = 0.0
            for k, la_seg in enumerate(lookahead):
                if k == 0:
                    # Current segment — skip
                    continue
                if isinstance(la_seg, CurveSegment):
                    sign = 1.0 if la_seg.angle_span >= 0 else -1.0
                    next_curvature = sign / la_seg.radius
                    dist_to_next_curve = accumulated_dist
                    break
                accumulated_dist += compute_segment_length(la_seg)

            # Path efficiency: on curves, inner line covers less ground per
            # unit of angular progress.  seg.radius / actual_radius gives the
            # ratio of centerline arc to the horse's real arc (< 1 when wide).
            seg = self.segments[hs.navigator.segment_index]
            if isinstance(seg, CurveSegment) and frame.turn_radius < 1e6:
                path_efficiency = seg.radius / max(frame.turn_radius, 1e-6)
            else:
                path_efficiency = 1.0

            stamina_ratio = (
                hs.runtime.current_stamina / hs.base_attrs.stamina
                if hs.base_attrs.stamina > 1e-6
                else 0.0
            )

            # Cornering margin
            if frame.turn_radius < 1e6:
                required_force = tangential_vel**2 / frame.turn_radius
                tolerated_force = hs.effective_attrs.cornering_grip * 150.0
                cornering_margin = tolerated_force - required_force
            else:
                cornering_margin = float("inf")

            # Relative positions & velocities to other horses (sorted by track progress)
            relatives = []
            for j, other in enumerate(self.horses):
                if j == i:
                    continue
                delta = other.body.position - hs.body.position
                tang_off = float(np.dot(delta, frame.tangential))
                norm_off = float(np.dot(delta, frame.normal))
                vel_delta = other.body.velocity - hs.body.velocity
                rel_tang_vel = float(np.dot(vel_delta, frame.tangential))
                rel_norm_vel = float(np.dot(vel_delta, frame.normal))
                progress_diff = other.track_progress - hs.track_progress
                relatives.append((progress_diff, tang_off, norm_off, rel_tang_vel, rel_norm_vel))
            # Descending: horses ahead (positive progress_diff) come first
            relatives.sort(key=lambda r: -r[0])

            # Pad to MAX_REL_HORSES (19) entries
            while len(relatives) < MAX_REL_HORSES:
                relatives.append((0.0, 0.0, 0.0, 0.0, 0.0))

            obs_list.append(
                {
                    "tangential_vel": tangential_vel,
                    "normal_vel": normal_vel,
                    "displacement": displacement,
                    "track_progress": hs.track_progress,
                    "curvature": curvature,
                    "stamina_ratio": stamina_ratio,
                    "effective_cruise_speed": hs.effective_attrs.cruise_speed,
                    "effective_max_speed": hs.effective_attrs.max_speed,
                    "relatives": [
                        (r[1], r[2], r[3], r[4]) for r in relatives[:MAX_REL_HORSES]
                    ],
                    "cornering_margin": cornering_margin,
                    "path_efficiency": path_efficiency,
                    "next_curvature": next_curvature,
                    "distance_to_next_curve": dist_to_next_curve,
                    "track_length": hs.navigator._total_length,
                    "slope": frame.slope,
                    "pushing_power": hs.effective_attrs.pushing_power,
                    "push_resistance": hs.effective_attrs.push_resistance,
                    "forward_accel": hs.effective_attrs.forward_accel,
                    "turn_accel": hs.effective_attrs.turn_accel,
                    "cornering_grip": hs.effective_attrs.cornering_grip,
                    "drain_rate_mult": hs.effective_attrs.drain_rate_mult,
                    "placement_norm": (placements[i] - 1) / max(num_horses - 1, 1),
                    "num_horses": num_horses,
                    "active_modifiers": {
                        m.id for m in hs.runtime.active_modifiers
                    },
                    "collision": hs.collision_this_tick,
                    "finished": hs.finished,
                }
            )

        # Reset per-tick collision flags
        for hs in self.horses:
            hs.collision_this_tick = False

        return obs_list

    def get_placements(self) -> list[int]:
        """Return 1-indexed placement for each horse.

        Finished horses are ranked by actual finish order (who crossed
        the line first). Unfinished horses are ranked after all finishers,
        ordered by track progress (highest first).
        """
        n = len(self.horses)
        placements = [0] * n

        # Finished horses get their actual crossing order
        for i in range(n):
            if self._finish_order[i] > 0:
                placements[i] = self._finish_order[i]

        # Unfinished horses ranked after all finishers, by progress
        unfinished = [
            (i, self.horses[i].track_progress)
            for i in range(n)
            if self._finish_order[i] == 0
        ]
        unfinished.sort(key=lambda x: -x[1])
        for rank, (idx, _) in enumerate(unfinished):
            placements[idx] = self._finish_count + rank + 1

        return placements

    def obs_to_array(
        self, obs: dict, active_skills: set[str] | None = None,
    ) -> np.ndarray:
        """Convert observation dict to flat numpy array (111,).

        Layout: 8 ego + 76 relative (19×4) + 13 track/attr + 8 modifier flags + 6 skill flags.
        """
        active = obs.get("active_modifiers", set())
        modifier_flags = [1.0 if mid in active else 0.0 for mid in MODIFIER_IDS]
        # Scale skill flags to 5.0 so they have comparable magnitude to continuous features
        # (binary 0/1 flags get drowned out by features ranging 0-25)
        skill_flags = [5.0 if sid in (active_skills or set()) else 0.0 for sid in SKILL_IDS]

        # Flatten relatives: 19 horses × 4 features = 76 values
        rel_flat: list[float] = []
        for tang_off, norm_off, rel_tv, rel_nv in obs["relatives"]:
            rel_flat.extend([tang_off, norm_off, rel_tv, rel_nv])

        return np.array(
            [
                obs["tangential_vel"],
                obs["normal_vel"],
                obs["displacement"],
                obs["track_progress"],
                obs["curvature"],
                obs["stamina_ratio"],
                obs["effective_cruise_speed"],
                obs["effective_max_speed"],
                *rel_flat,
                min(obs["cornering_margin"], 1000.0),  # cap inf
                obs["slope"],
                obs["pushing_power"],
                obs["push_resistance"],
                obs["forward_accel"],
                obs["turn_accel"],
                obs["cornering_grip"],
                obs["drain_rate_mult"],
                obs["placement_norm"],
                obs["num_horses"] / 20.0,  # normalize to [0, 1]
                obs.get("next_curvature", 0.0),
                obs.get("distance_to_next_curve", 0.0) / 100.0,  # normalize: 100m scale
                obs.get("track_length", 900.0) / 2000.0,  # normalize: 2000m scale
                *modifier_flags,
                *skill_flags,
            ],
            dtype=np.float32,
        )
