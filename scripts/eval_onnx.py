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

import numpy as np
import onnxruntime as ort

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
    ("Kyoto (917m)", "tracks/kyoto.json"),
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
    deterministic: bool = True,
) -> list[EpisodeStats]:
    """Run one episode with all horses controlled by the ONNX model."""
    genomes = [random_genome() for _ in range(horse_count)]
    engine_config = EngineConfig(horse_count=horse_count)
    engine = HorseRacingEngine(track_path, engine_config, genomes=genomes)

    stats = [EpisodeStats() for _ in range(horse_count)]
    speeds: list[list[float]] = [[] for _ in range(horse_count)]
    prev_obs = engine.get_observations()
    done = [False] * horse_count
    total_rewards = [0.0] * horse_count

    for step in range(max_steps):
        # Build observation batch for all non-done horses
        all_obs = engine.get_observations()
        obs_arrays = []
        for i in range(horse_count):
            obs_arrays.append(engine.obs_to_array(all_obs[i]))

        obs_batch = np.stack(obs_arrays, axis=0).astype(np.float32)
        raw_actions = session.run(None, {"obs": obs_batch})[0]
        # Clip to action space: tangential [-10, 10], normal [-5, 5]
        actions = np.clip(raw_actions, [-10.0, -5.0], [10.0, 5.0])

        # Build action list
        action_list = []
        for i in range(horse_count):
            if done[i]:
                action_list.append(HorseAction())
            else:
                action_list.append(HorseAction(float(actions[i, 0]), float(actions[i, 1])))

        engine.step(action_list)

        new_obs = engine.get_observations()
        placements = engine.get_placements()

        any_truncated = (step + 1) >= max_steps

        for i in range(horse_count):
            if done[i]:
                continue

            o = new_obs[i]
            speed = o["tangential_vel"]
            speeds[i].append(speed)

            if speed > stats[i].max_speed:
                stats[i].max_speed = speed
            if o["stamina_ratio"] < stats[i].min_stamina:
                stats[i].min_stamina = o["stamina_ratio"]
            if o.get("collision", False):
                stats[i].collisions += 1

            # Compute reward
            finish_order = placements[i] if o["finished"] else None
            r = compute_reward(
                prev_obs[i], o, o["collision"],
                placement=placements[i],
                num_horses=horse_count,
                finish_order=finish_order,
            )
            total_rewards[i] += r

            if o["finished"] or any_truncated:
                stats[i].progress = o["track_progress"]
                stats[i].final_speed = speed
                stats[i].steps = step + 1
                stats[i].finished = o.get("finished", False) or o["track_progress"] >= 0.99
                stats[i].total_reward = total_rewards[i]
                stats[i].avg_speed = sum(speeds[i]) / len(speeds[i]) if speeds[i] else 0
                stats[i].final_stamina = o["stamina_ratio"]
                stats[i].placement = placements[i]
                done[i] = True

        prev_obs = new_obs

        if all(done):
            break

    return stats


def print_report(all_stats: list[TrackStats], model_name: str):
    print()
    print(f"{'=' * 100}")
    print(f"  ONNX MODEL EVALUATION: {model_name}")
    print(f"{'=' * 100}")
    print()

    header = (
        f"{'Track':<25s} {'Complete':>8s} {'Progress':>12s} {'AvgSpeed':>10s} "
        f"{'MaxSpeed':>10s} {'Reward':>12s} {'Steps':>10s} {'Stamina':>10s}"
    )
    print(header)
    print("-" * len(header))

    for ts in all_stats:
        comp = f"{ts.completion_rate():.0%}"
        prog = f"{ts.mean('progress'):.1%}±{ts.std('progress'):.1%}"
        avg_spd = f"{ts.mean('avg_speed'):.1f}±{ts.std('avg_speed'):.1f}"
        max_spd = f"{ts.mean('max_speed'):.1f}±{ts.std('max_speed'):.1f}"
        rew = f"{ts.mean('total_reward'):.0f}±{ts.std('total_reward'):.0f}"
        steps = f"{ts.mean('steps'):.0f}±{ts.std('steps'):.0f}"
        stam = f"{ts.mean('final_stamina'):.0%}±{ts.std('final_stamina'):.0%}"
        print(
            f"{ts.track_name:<25s} {comp:>8s} {prog:>12s} {avg_spd:>10s} "
            f"{max_spd:>10s} {rew:>12s} {steps:>10s} {stam:>10s}"
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
    args = parser.parse_args()

    # Load ONNX model
    session = ort.InferenceSession(args.model)
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"Model: {args.model}")
    print(f"  Input:  {input_info.name} {input_info.shape}")
    print(f"  Output: {output_info.name} {output_info.shape}")
    print(f"  Horses: {args.horses}, Episodes/track: {args.episodes}, Max steps: {args.max_steps}")
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
            ep_stats = run_episode(session, track_path, args.horses, args.max_steps)
            # Use horse_0 as the primary agent for track-level stats
            # but record all horses for richer statistics
            for s in ep_stats:
                ts.episodes.append(s)

            # Show progress: check if horse_0 finished
            primary = ep_stats[0]
            sys.stdout.write("." if primary.finished else "x")
            sys.stdout.flush()

        print(f" ({ts.completion_rate():.0%})")
        all_stats.append(ts)

    print_report(all_stats, args.model)


if __name__ == "__main__":
    main()
