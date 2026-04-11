#!/usr/bin/env python3
"""Evaluate an ONNX-exported RLlib multi-agent model across tracks.

Usage:
    python scripts/eval_onnx.py --model path/to/horse_jockey.onnx
    python scripts/eval_onnx.py --model path/to/horse_jockey.onnx --tracks tracks/tokyo.json --episodes 10
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import random

import numpy as np
import onnxruntime as ort

from horse_racing.bt_jockey import PERSONALITIES, make_bt_jockey
from horse_racing.engine import EngineConfig, HorseRacingEngine
from horse_racing.genome import random_genome
from horse_racing.reward import compute_reward, ARCHETYPES
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
]


@dataclass
class EpisodeStats:
    progress: float = 0.0
    final_speed: float = 0.0
    max_speed: float = 0.0
    avg_speed: float = 0.0
    total_reward: float = 0.0
    steps: int = 0
    finished: bool = False
    collisions: int = 0
    min_stamina: float = 1.0
    final_stamina: float = 1.0
    placement: int = 0
    beat_bts: bool = False  # True if the policy finished ahead of all BT opponents


@dataclass
class TrackStats:
    track_name: str
    episodes: list[EpisodeStats] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.episodes)

    def mean(self, attr: str) -> float:
        vals = [getattr(e, attr) for e in self.episodes]
        return sum(vals) / len(vals) if vals else 0

    def std(self, attr: str) -> float:
        vals = [getattr(e, attr) for e in self.episodes]
        if len(vals) < 2:
            return 0
        m = sum(vals) / len(vals)
        return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    def completion_rate(self) -> float:
        if not self.episodes:
            return 0
        return sum(1 for e in self.episodes if e.finished) / len(self.episodes)


def run_episode(
    session: ort.InferenceSession,
    track_path: str,
    horse_count: int,
    max_steps: int,
    reward_phase: int = 1,
    bt_opponents: bool = True,
) -> EpisodeStats:
    """Run one episode in training conditions.

    Horse 0 is driven by the ONNX policy. When ``bt_opponents`` is True
    (default, matching ``HorseRacingSingleEnv``), horses 1..N are driven
    by randomly-chosen BT jockeys; otherwise they take zero actions.
    Only horse 0's stats are returned — this mirrors the single-agent
    training loop that produced ``history.json``.
    """
    genomes = [random_genome() for _ in range(horse_count)]
    engine_config = EngineConfig(horse_count=horse_count)
    engine = HorseRacingEngine(track_path, engine_config, genomes=genomes)

    bt_jockeys: list = []
    if bt_opponents and horse_count > 1:
        personality_choices = list(PERSONALITIES.keys())
        for _ in range(1, horse_count):
            bt_jockeys.append(make_bt_jockey(random.choice(personality_choices)))

    stats = EpisodeStats()
    speeds: list[float] = []
    prev_obs = engine.get_observations()
    prev_placement = engine.get_placements()[0]
    done0 = False
    total_reward = 0.0

    for step in range(max_steps):
        all_obs = engine.get_observations()

        # Horse 0: ONNX policy. The exported ONNX model has the
        # [-1,1] → physics remap baked into its graph (see the
        # SB3PolicyWrapper in notebooks/train_phased.ipynb), so the
        # raw output is already in tang [-3, +10], norm [-5, +5].
        # We only clip for safety against edge-case numerical drift.
        obs0 = engine.obs_to_array(all_obs[0]).astype(np.float32)
        raw = session.run(None, {"obs": obs0[None, :]})[0][0]
        tang = max(-10.0, min(10.0, float(raw[0])))
        norm = max(-5.0, min(5.0, float(raw[1])))

        action_list: list[HorseAction] = []
        if done0:
            action_list.append(HorseAction())
        else:
            action_list.append(HorseAction(tang, norm))

        # Horses 1..N: BT jockeys or zero actions
        for j in range(1, horse_count):
            if bt_jockeys:
                action_list.append(bt_jockeys[j - 1].compute_action(all_obs[j]))
            else:
                action_list.append(HorseAction())

        engine.step(action_list)

        new_obs = engine.get_observations()
        placements = engine.get_placements()
        any_truncated = (step + 1) >= max_steps

        if not done0:
            o = new_obs[0]
            speed = o["tangential_vel"]
            speeds.append(speed)

            if speed > stats.max_speed:
                stats.max_speed = speed
            if o["stamina_ratio"] < stats.min_stamina:
                stats.min_stamina = o["stamina_ratio"]
            if o.get("collision", False):
                stats.collisions += 1

            finish_order = placements[0] if o["finished"] else None
            r = compute_reward(
                prev_obs[0], o, o["collision"],
                placement=placements[0],
                num_horses=horse_count,
                finish_order=finish_order,
                prev_placement=prev_placement,
                reward_phase=reward_phase,
                raw_tang_action=tang,
            )
            total_reward += r
            prev_placement = placements[0]

            if o["finished"] or any_truncated:
                stats.progress = o["track_progress"]
                stats.final_speed = speed
                stats.steps = step + 1
                stats.finished = bool(o.get("finished", False)) or o["track_progress"] >= 0.99
                stats.total_reward = total_reward
                stats.avg_speed = sum(speeds) / len(speeds) if speeds else 0
                stats.final_stamina = o["stamina_ratio"]
                stats.placement = placements[0]
                # "Beat the BTs" = finished ahead of every BT opponent.
                # When BTs are off this always trivially true if horse 0
                # finished, so we only set it under competitive conditions.
                if bt_opponents and stats.finished:
                    stats.beat_bts = placements[0] == 1
                done0 = True

        prev_obs = new_obs

        if done0:
            break

    return stats


def print_report(all_stats: list[TrackStats], model_name: str, bt_opponents: bool):
    print()
    print(f"{'=' * 108}")
    print(f"  ONNX MODEL EVALUATION: {model_name}")
    print(f"  Mode: {'horse 0 vs BTs (training conditions)' if bt_opponents else 'zero-opponent (solo)'}")
    print(f"{'=' * 108}")
    print()

    header = (
        f"{'Track':<25s} {'Complete':>8s} {'BeatBTs':>8s} {'Place':>6s} "
        f"{'Progress':>12s} {'AvgSpeed':>10s} {'Reward':>12s} {'Steps':>10s} {'Stamina':>10s}"
    )
    print(header)
    print("-" * len(header))

    for ts in all_stats:
        comp = f"{ts.completion_rate():.0%}"
        won = f"{ts.mean('beat_bts'):.0%}" if bt_opponents else "  -  "
        place = f"{ts.mean('placement'):.1f}"
        prog = f"{ts.mean('progress'):.1%}±{ts.std('progress'):.1%}"
        avg_spd = f"{ts.mean('avg_speed'):.1f}±{ts.std('avg_speed'):.1f}"
        rew = f"{ts.mean('total_reward'):.0f}±{ts.std('total_reward'):.0f}"
        steps = f"{ts.mean('steps'):.0f}±{ts.std('steps'):.0f}"
        stam = f"{ts.mean('final_stamina'):.0%}±{ts.std('final_stamina'):.0%}"
        print(
            f"{ts.track_name:<25s} {comp:>8s} {won:>8s} {place:>6s} "
            f"{prog:>12s} {avg_spd:>10s} {rew:>12s} {steps:>10s} {stam:>10s}"
        )

    print()

    # Overall summary
    all_episodes = [e for ts in all_stats for e in ts.episodes]
    total_comp = sum(1 for e in all_episodes if e.finished) / len(all_episodes) if all_episodes else 0
    avg_progress = sum(e.progress for e in all_episodes) / len(all_episodes) if all_episodes else 0
    avg_reward = sum(e.total_reward for e in all_episodes) / len(all_episodes) if all_episodes else 0
    avg_speed = sum(e.avg_speed for e in all_episodes) / len(all_episodes) if all_episodes else 0
    worst_stamina = min(e.min_stamina for e in all_episodes) if all_episodes else 0

    print(f"  OVERALL ({len(all_episodes)} episodes across {len(all_stats)} tracks)")
    print(f"    Completion rate: {total_comp:.0%}")
    if bt_opponents:
        win_rate = sum(1 for e in all_episodes if e.beat_bts) / len(all_episodes) if all_episodes else 0
        avg_place = sum(e.placement for e in all_episodes) / len(all_episodes) if all_episodes else 0
        print(f"    Beat BTs:        {win_rate:.0%}")
        print(f"    Avg placement:   {avg_place:.2f}")
    print(f"    Avg progress:    {avg_progress:.1%}")
    print(f"    Avg reward:      {avg_reward:.1f}")
    print(f"    Avg speed:       {avg_speed:.1f} m/s")
    print(f"    Worst stamina:   {worst_stamina:.0%}")
    print()

    # Per-track ranking
    sorted_tracks = sorted(all_stats, key=lambda ts: ts.completion_rate(), reverse=True)
    print("  Track ranking (by completion):")
    for i, ts in enumerate(sorted_tracks, 1):
        print(f"    {i}. {ts.track_name}: {ts.completion_rate():.0%} complete, "
              f"avg progress {ts.mean('progress'):.1%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate ONNX horse racing model")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per track")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--horses", type=int, default=4, help="Number of horses per race")
    parser.add_argument("--tracks", type=str, nargs="*", default=None,
                        help="Track files (default: all)")
    parser.add_argument("--reward-phase", type=int, default=1, choices=[1, 2, 3],
                        help="Reward phase used for scoring (default: 1, matches stage-1 training)")
    parser.add_argument("--no-bt", action="store_true",
                        help="Disable BT opponents (horses 1..N take zero actions). "
                             "By default horse 0 races 3 BT jockeys, matching training.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for BT personality sampling and genome sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    bt_enabled = not args.no_bt

    # Load ONNX model
    session = ort.InferenceSession(args.model)
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"Model: {args.model}")
    print(f"  Input:  {input_info.name} {input_info.shape}")
    print(f"  Output: {output_info.name} {output_info.shape}")
    print(f"  Horses: {args.horses}, Episodes/track: {args.episodes}, Max steps: {args.max_steps}")
    print(f"  Reward phase: {args.reward_phase}  BT opponents: {'on' if bt_enabled else 'off'}")
    print()

    tracks = TRACKS
    if args.tracks:
        tracks = [(t.split("/")[-1], t) for t in args.tracks]

    all_stats: list[TrackStats] = []

    for track_name, track_path in tracks:
        ts = TrackStats(track_name=track_name)
        sys.stdout.write(f"  {track_name}: ")
        sys.stdout.flush()

        for ep in range(args.episodes):
            s = run_episode(
                session, track_path, args.horses, args.max_steps,
                reward_phase=args.reward_phase, bt_opponents=bt_enabled,
            )
            ts.episodes.append(s)

            # Progress glyph: W = beat all BTs, . = finished, x = DNF
            if bt_enabled and s.beat_bts:
                glyph = "W"
            elif s.finished:
                glyph = "."
            else:
                glyph = "x"
            sys.stdout.write(glyph)
            sys.stdout.flush()

        print(f" ({ts.completion_rate():.0%})")
        all_stats.append(ts)

    print_report(all_stats, args.model, bt_enabled)


if __name__ == "__main__":
    main()
