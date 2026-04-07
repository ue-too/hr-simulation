#!/usr/bin/env python3
"""Evaluate a single-agent ONNX model (horse 0) vs BT scripted opponents.

Designed to diagnose acceleration/kick behavior by logging per-phase actions
and comparing the RL agent's performance against each BT archetype.

Usage:
    python scripts/eval_single_agent.py --model ~/Desktop/stage_4_jockey.onnx
    python scripts/eval_single_agent.py --model ~/Desktop/stage_4_jockey.onnx --tracks tracks/tokyo.json --episodes 10
    python scripts/eval_single_agent.py --model ~/Desktop/stage_4_jockey.onnx --horses 8 --verbose
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort

from horse_racing.bt_jockey import PERSONALITIES, make_bt_jockey
from horse_racing.engine import EngineConfig, HorseRacingEngine
from horse_racing.genome import random_genome
from horse_racing.reward import compute_reward
from horse_racing.types import HorseAction, OBS_SIZE


TRACKS = [
    ("Straight (400m)", "tracks/curriculum_1_straight.json"),
    ("Gentle Oval (1810m)", "tracks/curriculum_2_gentle_oval.json"),
    ("Tight Oval (1005m)", "tracks/curriculum_3_tight_oval.json"),
    ("Test Oval (905m)", "tracks/test_oval.json"),
    ("Tokyo (901m)", "tracks/tokyo.json"),
    ("Kokura (1415m)", "tracks/kokura.json"),
    ("Tokyo 2600 (1528m)", "tracks/tokyo_2600.json"),
    ("Hanshin (1008m)", "tracks/hanshin.json"),
    ("Kyoto (917m)", "tracks/kyoto.json"),
]

# Progress boundaries for phase analysis
PHASE_EARLY = 0.30    # 0 - 30%
PHASE_MID = 0.70      # 30 - 70%
PHASE_KICK = 1.0      # 70 - 100%


@dataclass
class PhaseStats:
    """Action & speed stats for a race phase."""
    tang_actions: list[float] = field(default_factory=list)
    norm_actions: list[float] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    stamina_values: list[float] = field(default_factory=list)

    @property
    def mean_tang(self) -> float:
        return np.mean(self.tang_actions) if self.tang_actions else 0.0

    @property
    def mean_norm(self) -> float:
        return np.mean(self.norm_actions) if self.norm_actions else 0.0

    @property
    def max_tang(self) -> float:
        return max(self.tang_actions) if self.tang_actions else 0.0

    @property
    def mean_speed(self) -> float:
        return np.mean(self.speeds) if self.speeds else 0.0

    @property
    def mean_stamina(self) -> float:
        return np.mean(self.stamina_values) if self.stamina_values else 0.0


@dataclass
class HorseResult:
    """Results for a single horse in a race."""
    horse_idx: int
    controller: str  # "onnx" or BT archetype name
    placement: int = 0
    progress: float = 0.0
    finished: bool = False
    steps: int = 0
    total_reward: float = 0.0
    avg_speed: float = 0.0
    max_speed: float = 0.0
    final_stamina: float = 1.0
    min_stamina: float = 1.0
    collisions: int = 0
    # Per-phase breakdown (only for ONNX agent)
    early: PhaseStats = field(default_factory=PhaseStats)
    mid: PhaseStats = field(default_factory=PhaseStats)
    kick: PhaseStats = field(default_factory=PhaseStats)


@dataclass
class RaceResult:
    """Full results of one race."""
    track_name: str
    horses: list[HorseResult] = field(default_factory=list)

    @property
    def onnx_horse(self) -> HorseResult:
        return self.horses[0]

    @property
    def bt_horses(self) -> list[HorseResult]:
        return self.horses[1:]


def run_race(
    session: ort.InferenceSession,
    track_name: str,
    track_path: str,
    horse_count: int,
    max_steps: int,
    verbose: bool = False,
) -> RaceResult:
    """Run one race: ONNX horse 0 vs BT opponents."""
    genomes = [random_genome() for _ in range(horse_count)]
    engine = HorseRacingEngine(track_path, EngineConfig(horse_count=horse_count), genomes=genomes)

    # Scale stamina for long tracks (same as training env)
    track_length = engine.horses[0].navigator._total_length
    REFERENCE_LENGTH = 900.0
    if track_length > REFERENCE_LENGTH:
        scale = track_length / REFERENCE_LENGTH
        for hs in engine.horses:
            hs.base_attrs.stamina *= scale
            hs.effective_attrs.stamina *= scale
            hs.runtime.current_stamina *= scale

    # Assign BT archetypes to opponents
    rng = np.random.default_rng()
    archetype_names = list(PERSONALITIES.keys())
    bt_jockeys = []
    bt_archetypes = []
    for _ in range(1, horse_count):
        arch = rng.choice(archetype_names)
        bt_jockeys.append(make_bt_jockey(arch))
        bt_archetypes.append(arch)

    results = RaceResult(track_name=track_name)
    results.horses.append(HorseResult(horse_idx=0, controller="onnx"))
    for i, arch in enumerate(bt_archetypes):
        results.horses.append(HorseResult(horse_idx=i + 1, controller=arch))

    all_speeds: list[list[float]] = [[] for _ in range(horse_count)]
    done = [False] * horse_count
    total_rewards = [0.0] * horse_count
    prev_obs = engine.get_observations()
    # Track actual finish order (engine.get_placements only ranks by progress snapshot)
    finish_order_counter = 0
    actual_finish_order = [0] * horse_count  # 0 = not finished yet

    for step in range(max_steps):
        all_obs = engine.get_observations()

        # ONNX action for horse 0
        obs_array = engine.obs_to_array(all_obs[0]).astype(np.float32)
        raw = session.run(None, {"obs": obs_array.reshape(1, -1)})[0][0]
        onnx_action = np.clip(raw, [-10.0, -5.0], [10.0, 5.0])

        actions = [HorseAction(float(onnx_action[0]), float(onnx_action[1]))]
        for j, bt in enumerate(bt_jockeys):
            actions.append(bt.compute_action(all_obs[j + 1]))

        engine.step(actions)
        new_obs = engine.get_observations()

        progress_0 = new_obs[0]["track_progress"]

        # Record phase stats for ONNX agent
        if not done[0]:
            if progress_0 < PHASE_EARLY:
                phase = results.horses[0].early
            elif progress_0 < PHASE_MID:
                phase = results.horses[0].mid
            else:
                phase = results.horses[0].kick
            phase.tang_actions.append(float(onnx_action[0]))
            phase.norm_actions.append(float(onnx_action[1]))
            phase.speeds.append(new_obs[0]["tangential_vel"])
            phase.stamina_values.append(new_obs[0]["stamina_ratio"])

        any_truncated = (step + 1) >= max_steps

        # Record actual finish order — who crosses the line first gets place 1
        for i in range(horse_count):
            if actual_finish_order[i] == 0 and new_obs[i].get("finished", False):
                finish_order_counter += 1
                actual_finish_order[i] = finish_order_counter

        for i in range(horse_count):
            if done[i]:
                continue

            o = new_obs[i]
            speed = o["tangential_vel"]
            all_speeds[i].append(speed)

            hr = results.horses[i]
            if speed > hr.max_speed:
                hr.max_speed = speed
            if o["stamina_ratio"] < hr.min_stamina:
                hr.min_stamina = o["stamina_ratio"]
            if o.get("collision", False):
                hr.collisions += 1

            finish_place = actual_finish_order[i] if actual_finish_order[i] > 0 else None
            r = compute_reward(
                prev_obs[i], o, o["collision"],
                placement=engine.get_placements()[i],
                num_horses=horse_count,
                finish_order=finish_place,
            )
            total_rewards[i] += r

            if o["finished"] or any_truncated:
                hr.progress = o["track_progress"]
                hr.final_stamina = o["stamina_ratio"]
                hr.finished = o.get("finished", False) or o["track_progress"] >= 0.99
                hr.steps = step + 1
                hr.total_reward = total_rewards[i]
                hr.avg_speed = np.mean(all_speeds[i]) if all_speeds[i] else 0.0
                # Use actual finish order, fall back to progress-based for DNFs
                if actual_finish_order[i] > 0:
                    hr.placement = actual_finish_order[i]
                else:
                    # DNF: place after all finishers, ranked by progress
                    hr.placement = horse_count
                done[i] = True

        prev_obs = new_obs
        if all(done):
            break

    if verbose:
        _print_race_detail(results)

    return results


def _print_race_detail(race: RaceResult):
    """Print detailed single-race breakdown."""
    rl = race.onnx_horse
    print(f"\n  --- {race.track_name} ---")
    print(f"  ONNX agent: placement={rl.placement}, progress={rl.progress:.1%}, "
          f"finished={rl.finished}, steps={rl.steps}")
    print(f"    Speed: avg={rl.avg_speed:.1f} max={rl.max_speed:.1f} m/s")
    print(f"    Stamina: final={rl.final_stamina:.0%} min={rl.min_stamina:.0%}")
    print(f"    Collisions: {rl.collisions}")
    print(f"    Phase actions (tangential) → early={rl.early.mean_tang:+.2f} "
          f"mid={rl.mid.mean_tang:+.2f} kick={rl.kick.mean_tang:+.2f}")
    print(f"    Phase actions (max tang)   → early={rl.early.max_tang:+.2f} "
          f"mid={rl.mid.max_tang:+.2f} kick={rl.kick.max_tang:+.2f}")
    print(f"    Phase speeds               → early={rl.early.mean_speed:.1f} "
          f"mid={rl.mid.mean_speed:.1f} kick={rl.kick.mean_speed:.1f}")
    print(f"    Phase stamina              → early={rl.early.mean_stamina:.0%} "
          f"mid={rl.mid.mean_stamina:.0%} kick={rl.kick.mean_stamina:.0%}")
    print()
    for bt in race.bt_horses:
        print(f"    BT {bt.controller:<14s}: place={bt.placement} progress={bt.progress:.1%} "
              f"avg_spd={bt.avg_speed:.1f} stamina={bt.final_stamina:.0%}")


def print_summary(all_races: list[RaceResult], model_name: str):
    print()
    print(f"{'=' * 110}")
    print(f"  SINGLE-AGENT EVAL: {model_name}")
    print(f"{'=' * 110}")

    # --- Per-track table ---
    header = (
        f"{'Track':<25s} {'Win%':>6s} {'AvgPlace':>9s} {'Finish%':>8s} "
        f"{'Progress':>10s} {'AvgSpd':>8s} {'MaxSpd':>8s} "
        f"{'Stamina':>8s} {'Reward':>10s}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    track_groups: dict[str, list[RaceResult]] = {}
    for r in all_races:
        track_groups.setdefault(r.track_name, []).append(r)

    total_wins = 0
    total_races = 0

    for track_name, races in track_groups.items():
        rl_results = [r.onnx_horse for r in races]
        n = len(rl_results)
        wins = sum(1 for r in rl_results if r.placement == 1)
        total_wins += wins
        total_races += n

        avg_place = np.mean([r.placement for r in rl_results])
        finish_pct = sum(1 for r in rl_results if r.finished) / n
        avg_prog = np.mean([r.progress for r in rl_results])
        avg_spd = np.mean([r.avg_speed for r in rl_results])
        max_spd = np.mean([r.max_speed for r in rl_results])
        avg_stam = np.mean([r.final_stamina for r in rl_results])
        avg_rew = np.mean([r.total_reward for r in rl_results])

        print(
            f"{track_name:<25s} {wins}/{n:>3d} {avg_place:>9.1f} {finish_pct:>8.0%} "
            f"{avg_prog:>10.1%} {avg_spd:>8.1f} {max_spd:>8.1f} "
            f"{avg_stam:>8.0%} {avg_rew:>10.1f}"
        )

    # --- Overall ---
    all_rl = [r.onnx_horse for r in all_races]
    all_bt = [h for r in all_races for h in r.bt_horses]

    print(f"\n{'─' * 80}")
    print(f"  OVERALL: {total_races} races, {total_wins} wins ({total_wins/total_races:.0%} win rate)")
    print(f"    ONNX avg placement: {np.mean([r.placement for r in all_rl]):.2f}")
    print(f"    ONNX avg speed:     {np.mean([r.avg_speed for r in all_rl]):.1f} m/s")
    print(f"    ONNX avg reward:    {np.mean([r.total_reward for r in all_rl]):.1f}")
    print(f"    BT   avg placement: {np.mean([h.placement for h in all_bt]):.2f}")
    print(f"    BT   avg speed:     {np.mean([h.avg_speed for h in all_bt]):.1f} m/s")

    # --- BT archetype breakdown ---
    print(f"\n  BT archetype head-to-head (ONNX wins / total matchups):")
    arch_stats: dict[str, dict] = {}
    for race in all_races:
        rl_place = race.onnx_horse.placement
        for bt in race.bt_horses:
            s = arch_stats.setdefault(bt.controller, {"wins": 0, "losses": 0, "ties": 0, "total": 0})
            s["total"] += 1
            if rl_place < bt.placement:
                s["wins"] += 1
            elif rl_place > bt.placement:
                s["losses"] += 1
            else:
                s["ties"] += 1

    for arch, s in sorted(arch_stats.items()):
        print(f"    vs {arch:<14s}: {s['wins']:>3d}W / {s['losses']:>3d}L / {s['ties']:>3d}T  "
              f"({s['total']} matchups, {s['wins']/s['total']:.0%} win rate)")

    # --- Kick/acceleration diagnosis ---
    print(f"\n  KICK / ACCELERATION DIAGNOSIS:")
    early_tangs = [r.onnx_horse.early.mean_tang for r in all_races if r.onnx_horse.early.tang_actions]
    mid_tangs = [r.onnx_horse.mid.mean_tang for r in all_races if r.onnx_horse.mid.tang_actions]
    kick_tangs = [r.onnx_horse.kick.mean_tang for r in all_races if r.onnx_horse.kick.tang_actions]
    kick_max = [r.onnx_horse.kick.max_tang for r in all_races if r.onnx_horse.kick.tang_actions]

    early_spd = [r.onnx_horse.early.mean_speed for r in all_races if r.onnx_horse.early.speeds]
    mid_spd = [r.onnx_horse.mid.mean_speed for r in all_races if r.onnx_horse.mid.speeds]
    kick_spd = [r.onnx_horse.kick.mean_speed for r in all_races if r.onnx_horse.kick.speeds]

    early_stam = [r.onnx_horse.early.mean_stamina for r in all_races if r.onnx_horse.early.stamina_values]
    mid_stam = [r.onnx_horse.mid.mean_stamina for r in all_races if r.onnx_horse.mid.stamina_values]
    kick_stam = [r.onnx_horse.kick.mean_stamina for r in all_races if r.onnx_horse.kick.stamina_values]

    def _fmt(vals):
        return f"{np.mean(vals):+.2f}" if vals else "n/a"

    def _fmt_spd(vals):
        return f"{np.mean(vals):.1f}" if vals else "n/a"

    def _fmt_pct(vals):
        return f"{np.mean(vals):.0%}" if vals else "n/a"

    print(f"    Mean tangential action → early={_fmt(early_tangs)}  mid={_fmt(mid_tangs)}  kick={_fmt(kick_tangs)}")
    print(f"    Max tangential (kick)  → {_fmt(kick_max)}")
    print(f"    Mean speed             → early={_fmt_spd(early_spd)}  mid={_fmt_spd(mid_spd)}  kick={_fmt_spd(kick_spd)}")
    print(f"    Mean stamina           → early={_fmt_pct(early_stam)}  mid={_fmt_pct(mid_stam)}  kick={_fmt_pct(kick_stam)}")

    if kick_tangs and mid_tangs:
        kick_boost = np.mean(kick_tangs) - np.mean(mid_tangs)
        print(f"    Kick boost (kick - mid tangential): {kick_boost:+.2f}")
        if kick_boost < 0.5:
            print(f"    ⚠  LOW KICK: model barely increases effort in final stretch")
        if kick_stam and np.mean(kick_stam) < 0.05:
            print(f"    ⚠  EMPTY TANK: stamina depleted before kick phase — nothing left to push with")
        elif kick_stam and np.mean(kick_stam) > 0.30:
            print(f"    ⚠  STAMINA HOARDING: model has stamina left but doesn't use it")

    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate single-agent ONNX model vs BT opponents")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per track")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--horses", type=int, default=4, help="Total horses (1 ONNX + N-1 BT)")
    parser.add_argument("--tracks", type=str, nargs="*", default=None, help="Specific track files")
    parser.add_argument("--verbose", action="store_true", help="Print per-race details")
    args = parser.parse_args()

    session = ort.InferenceSession(args.model)
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"Model: {args.model}")
    print(f"  Input:  {inp.name} {inp.shape}")
    print(f"  Output: {out.name} {out.shape}")
    print(f"  Horses: {args.horses} (1 ONNX + {args.horses - 1} BT opponents)")
    print(f"  Episodes/track: {args.episodes}, Max steps: {args.max_steps}")

    tracks = TRACKS
    if args.tracks:
        tracks = [(t.split("/")[-1], t) for t in args.tracks]

    all_races: list[RaceResult] = []

    for track_name, track_path in tracks:
        sys.stdout.write(f"  {track_name}: ")
        sys.stdout.flush()

        for _ in range(args.episodes):
            race = run_race(
                session, track_name, track_path,
                args.horses, args.max_steps, verbose=args.verbose,
            )
            all_races.append(race)
            rl = race.onnx_horse
            marker = str(rl.placement) if rl.finished else "x"
            sys.stdout.write(marker)
            sys.stdout.flush()

        # Quick track summary
        track_races = [r for r in all_races if r.track_name == track_name]
        wins = sum(1 for r in track_races if r.onnx_horse.placement == 1)
        finished = sum(1 for r in track_races if r.onnx_horse.finished)
        print(f"  [{wins}W/{finished}F out of {len(track_races)}]")

    print_summary(all_races, args.model)


if __name__ == "__main__":
    main()
