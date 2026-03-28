"""Evaluate a trained model across tracks with statistics over multiple episodes."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
from stable_baselines3 import PPO

from horse_racing.env import HorseRacingSingleEnv


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


@dataclass
class TrackStats:
    track_name: str
    episodes: list[EpisodeStats] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.episodes)

    def mean(self, field: str) -> float:
        vals = [getattr(e, field) for e in self.episodes]
        return sum(vals) / len(vals) if vals else 0

    def std(self, field: str) -> float:
        vals = [getattr(e, field) for e in self.episodes]
        if len(vals) < 2:
            return 0
        m = sum(vals) / len(vals)
        return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    def completion_rate(self) -> float:
        if not self.episodes:
            return 0
        return sum(1 for e in self.episodes if e.finished) / len(self.episodes)


TRACKS = [
    ("Straight (400m)", "tracks/curriculum_1_straight.json"),
    ("Gentle Oval (2885m)", "tracks/curriculum_2_gentle_oval.json"),
    ("Tight Oval (1542m)", "tracks/curriculum_3_tight_oval.json"),
    ("Complex (1821m)", "tracks/exp_track_8.json"),
]


def run_episode(
    model: PPO,
    track_path: str,
    max_steps: int,
    deterministic: bool = True,
) -> EpisodeStats:
    env = HorseRacingSingleEnv(track_path=track_path, max_steps=max_steps)
    obs, _ = env.reset()

    stats = EpisodeStats()
    speeds: list[float] = []

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        stats.total_reward += reward

        all_obs = env.engine.get_observations()
        o = all_obs[0]
        speed = o["tangential_vel"]
        speeds.append(speed)
        if speed > stats.max_speed:
            stats.max_speed = speed
        if o["stamina_ratio"] < stats.min_stamina:
            stats.min_stamina = o["stamina_ratio"]
        if o.get("collision", False):
            stats.collisions += 1

        if terminated or truncated:
            stats.progress = o["track_progress"]
            stats.final_speed = speed
            stats.steps = step + 1
            stats.finished = o.get("finished", False) or o["track_progress"] >= 0.99
            break

    stats.avg_speed = sum(speeds) / len(speeds) if speeds else 0
    env.close()
    return stats


def print_report(all_stats: list[TrackStats], model_name: str):
    print()
    print(f"{'=' * 90}")
    print(f"  EVALUATION REPORT: {model_name}")
    print(f"{'=' * 90}")
    print()

    header = f"{'Track':<25s} {'Complete':>8s} {'Progress':>10s} {'AvgSpeed':>10s} {'MaxSpeed':>10s} {'Reward':>10s} {'Steps':>8s} {'Collisions':>10s}"
    print(header)
    print("-" * len(header))

    for ts in all_stats:
        comp = f"{ts.completion_rate():.0%}"
        prog = f"{ts.mean('progress'):.1%}±{ts.std('progress'):.1%}"
        avg_spd = f"{ts.mean('avg_speed'):.1f}±{ts.std('avg_speed'):.1f}"
        max_spd = f"{ts.mean('max_speed'):.1f}±{ts.std('max_speed'):.1f}"
        rew = f"{ts.mean('total_reward'):.0f}±{ts.std('total_reward'):.0f}"
        steps = f"{ts.mean('steps'):.0f}±{ts.std('steps'):.0f}"
        colls = f"{ts.mean('collisions'):.0f}±{ts.std('collisions'):.0f}"
        print(f"{ts.track_name:<25s} {comp:>8s} {prog:>10s} {avg_spd:>10s} {max_spd:>10s} {rew:>10s} {steps:>8s} {colls:>10s}")

    print()

    # Detailed per-episode breakdown for the most interesting track
    best_track = max(all_stats, key=lambda ts: ts.mean("progress"))
    print(f"  Best track: {best_track.track_name}")
    print(f"  Completion rate: {best_track.completion_rate():.0%}")
    print(f"  Avg reward: {best_track.mean('total_reward'):.1f}")
    print(f"  Avg max speed: {best_track.mean('max_speed'):.1f} m/s")
    print(f"  Min stamina (worst): {min(e.min_stamina for e in best_track.episodes):.1%}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained horse racing model")
    parser.add_argument("--model", type=str, required=True, help="Path to SB3 model")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per track")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--tracks", type=str, nargs="*", default=None,
                        help="Track files (default: all curriculum + complex)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic actions (default: deterministic)")
    args = parser.parse_args()

    model = PPO.load(args.model)
    print(f"Loaded model from {args.model}")

    tracks = TRACKS
    if args.tracks:
        tracks = [(t.split("/")[-1], t) for t in args.tracks]

    all_stats: list[TrackStats] = []

    for track_name, track_path in tracks:
        ts = TrackStats(track_name=track_name)
        sys.stdout.write(f"  {track_name}: ")
        sys.stdout.flush()

        for ep in range(args.episodes):
            stats = run_episode(model, track_path, args.max_steps, not args.stochastic)
            ts.episodes.append(stats)
            sys.stdout.write("." if stats.finished else "x")
            sys.stdout.flush()

        print(f" ({ts.completion_rate():.0%})")
        all_stats.append(ts)

    print_report(all_stats, args.model)


if __name__ == "__main__":
    main()
