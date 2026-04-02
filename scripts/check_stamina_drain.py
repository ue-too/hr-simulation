#!/usr/bin/env python3
"""Stamina drain sanity check — run a horse under different effort levels
and print stamina at progress checkpoints.

Usage:
    python scripts/check_stamina_drain.py [--track TRACK_JSON]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from horse_racing.engine import EngineConfig, HorseRacingEngine
from horse_racing.types import HorseAction


def run_scenario(
    track_path: str,
    label: str,
    action: HorseAction,
    max_ticks: int = 20_000,
) -> None:
    engine = HorseRacingEngine(track_path, EngineConfig(horse_count=1))

    next_checkpoint = 0.1
    hs = engine.horses[0]
    initial_stamina = hs.base_attrs.stamina

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  action: tangential={action.extra_tangential:.1f}, normal={action.extra_normal:.1f}")
    print(f"  stamina pool: {initial_stamina:.1f}")
    print(f"{'=' * 60}")
    print(f"  {'Progress':>10}  {'Stamina':>10}  {'Ratio':>8}  {'Speed':>8}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 8}")

    for tick in range(max_ticks):
        engine.step([action])

        progress = hs.track_progress
        stamina = hs.runtime.current_stamina
        ratio = stamina / initial_stamina if initial_stamina > 0 else 0
        speed = float((hs.body.velocity[0] ** 2 + hs.body.velocity[1] ** 2) ** 0.5)

        if progress >= next_checkpoint or hs.finished:
            print(f"  {progress:>9.1%}  {stamina:>10.2f}  {ratio:>7.1%}  {speed:>7.2f}")
            next_checkpoint += 0.1

        if hs.finished:
            print(f"  Finished at tick {tick + 1} ({(tick + 1) / 240:.1f}s sim time)")
            break

        if stamina <= 0:
            print(f"  EXHAUSTED at tick {tick + 1}, progress={progress:.1%}")
            break
    else:
        print(f"  Did not finish in {max_ticks} ticks, progress={hs.track_progress:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stamina drain sanity check")
    parser.add_argument(
        "--track",
        default="tracks/curriculum_1_straight.json",
        help="Track JSON file (default: straight)",
    )
    args = parser.parse_args()

    track = args.track
    if not Path(track).exists():
        print(f"Track not found: {track}")
        return

    scenarios = [
        ("Cruise (zero action)", HorseAction(0.0, 0.0)),
        ("Moderate push", HorseAction(3.0, 0.0)),
        ("Hard push", HorseAction(5.0, 0.0)),
        ("Hard push + steering", HorseAction(5.0, 3.0)),
        ("Max effort", HorseAction(10.0, 5.0)),
    ]

    for label, action in scenarios:
        run_scenario(track, label, action)


if __name__ == "__main__":
    main()
