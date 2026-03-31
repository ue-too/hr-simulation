"""Profile the engine's hot paths to identify optimization targets.

Runs the simulation via HorseRacingSingleEnv.step() and measures time
spent in each component. Also runs cProfile for a full call tree.

Usage:
    python scripts/profile_engine.py
    # Then optionally: snakeviz logs/profile_engine.prof
"""

from __future__ import annotations

import cProfile
import pstats
import time
from pathlib import Path

import numpy as np

from horse_racing.engine import HorseRacingEngine
from horse_racing.env import HorseRacingSingleEnv
from horse_racing.reward import compute_reward
from horse_racing.types import HorseAction

TRACKS = [
    ("curriculum_1_straight.json", 1500),
    ("curriculum_3_tight_oval.json", 3000),
    ("tokyo.json", 3000),
]
N_STEPS = 3000


def profile_components(track_file: str, max_steps: int, n_steps: int) -> None:
    """Time each component of a single env step."""
    track_path = f"tracks/{track_file}"
    env = HorseRacingSingleEnv(track_path=track_path, max_steps=max_steps)
    obs_array, _ = env.reset()
    engine = env.engine

    # Accumulators (nanoseconds)
    t_env_step = 0
    t_engine_step = 0
    t_get_obs = 0
    t_obs_to_array = 0
    t_reward = 0
    t_placements = 0

    prev_obs = env._prev_obs
    prev_placement = 1
    action = np.array([1.0, 0.0], dtype=np.float32)

    for _ in range(n_steps):
        # --- Full env.step() ---
        t0 = time.perf_counter_ns()

        actions = [HorseAction(float(action[0]), float(action[1]))]
        for _ in range(1, engine.horse_count):
            actions.append(HorseAction())

        t1 = time.perf_counter_ns()
        engine.step(actions)
        t2 = time.perf_counter_ns()

        all_obs = engine.get_observations()
        t3 = time.perf_counter_ns()

        obs_curr = all_obs[0]
        obs_arr = engine.obs_to_array(obs_curr)
        t4 = time.perf_counter_ns()

        placements = engine.get_placements()
        t5 = time.perf_counter_ns()

        finish_order = placements[0] if obs_curr["finished"] else None
        reward = compute_reward(
            prev_obs, obs_curr, obs_curr["collision"],
            placement=placements[0],
            num_horses=engine.horse_count,
            finish_order=finish_order,
            prev_placement=prev_placement,
        )
        t6 = time.perf_counter_ns()

        prev_obs = obs_curr
        prev_placement = placements[0]

        t_env_step += t6 - t0
        t_engine_step += t2 - t1
        t_get_obs += t3 - t2
        t_obs_to_array += t4 - t3
        t_placements += t5 - t4
        t_reward += t6 - t5

        if obs_curr["finished"]:
            env.reset()
            prev_obs = env._prev_obs
            prev_placement = 1

    env.close()

    # Report
    total_us = t_env_step / 1000
    components = [
        ("engine.step()", t_engine_step),
        ("get_observations()", t_get_obs),
        ("obs_to_array()", t_obs_to_array),
        ("get_placements()", t_placements),
        ("compute_reward()", t_reward),
        ("env overhead", t_env_step - t_engine_step - t_get_obs - t_obs_to_array - t_placements - t_reward),
    ]

    print(f"\nTrack: {track_file} | {n_steps} steps | {engine.horse_count} horses")
    print(f"Total: {total_us / 1000:.1f} ms | {total_us / n_steps:.1f} µs/step | {n_steps / (t_env_step / 1e9):.0f} steps/sec")
    print(f"\n{'Component':<25} {'%':>6} {'µs/step':>10} {'total ms':>10}")
    print("-" * 55)
    for name, t_ns in components:
        pct = 100 * t_ns / t_env_step if t_env_step > 0 else 0
        us_per = t_ns / 1000 / n_steps
        ms_total = t_ns / 1e6
        print(f"  {name:<23} {pct:5.1f}% {us_per:9.1f} {ms_total:9.1f}")


def profile_cprofile(track_file: str, max_steps: int, n_steps: int) -> str:
    """Run cProfile on env stepping and return the .prof path."""
    track_path = f"tracks/{track_file}"
    env = HorseRacingSingleEnv(track_path=track_path, max_steps=max_steps)
    env.reset()
    action = np.array([1.0, 0.0], dtype=np.float32)

    def run_loop():
        for _ in range(n_steps):
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()

    prof_dir = Path("logs")
    prof_dir.mkdir(exist_ok=True)
    prof_path = str(prof_dir / "profile_engine.prof")

    profiler = cProfile.Profile()
    profiler.runcall(run_loop)
    profiler.dump_stats(prof_path)

    env.close()

    # Print top functions by cumulative time
    print(f"\n{'=' * 70}")
    print(f"cProfile top 25 by cumulative time ({track_file}, {n_steps} steps)")
    print(f"{'=' * 70}")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(25)

    return prof_path


def main() -> None:
    print("=" * 70)
    print("Engine Profiling — Component Breakdown")
    print("=" * 70)

    for track_file, max_steps in TRACKS:
        profile_components(track_file, max_steps, N_STEPS)

    # Full cProfile on tokyo (the most representative track)
    prof_path = profile_cprofile("tokyo.json", 3000, N_STEPS)
    print(f"\nProfile saved to: {prof_path}")
    print("View with: snakeviz logs/profile_engine.prof")


if __name__ == "__main__":
    main()
