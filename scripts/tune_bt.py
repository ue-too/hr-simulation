#!/usr/bin/env python3
"""Batch-run utility-BT races and aggregate stats for systematic tuning.

Runs headless races where every horse uses `BehaviorTreeStrategy` with a chosen
archetype, then prints win rates, mean finish position, mean race length, and
optional stamina-at-finish summaries.

Example::

    uv run python scripts/tune_bt.py --races 200 --seed 42
    uv run python scripts/tune_bt.py --track tracks/test_oval.json --races 500 \\
        --randomize-attrs --horse-count 6 --csv artifacts/bt_tune.csv

A tqdm progress bar runs on stderr unless ``--no-progress``, ``--verbose``,
or stderr is not a TTY (e.g. when piping).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# Repo root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from horse_racing.core.attributes import (  # noqa: E402
    create_default_attributes,
    create_randomized_attributes,
)
from horse_racing.core.race import Race
from horse_racing.core.track import load_track_json
from horse_racing.core.types import InputState
from horse_racing.opponents.behavior_tree import (  # noqa: E402
    ARCHETYPES,
    BehaviorTreeStrategy,
)


MAX_TICKS_SAFETY = 15000


def _default_track() -> Path:
    p = ROOT / "tracks" / "test_oval.json"
    if p.exists():
        return p
    return ROOT / "tracks" / "simple_oval.json"


def _parse_archetypes(arg: str | None, horse_count: int) -> list[str]:
    names = sorted(ARCHETYPES.keys())
    if arg is None or arg.strip().lower() == "all":
        # Round-robin assign archetypes in stable order
        order = ["front-runner", "closer", "stalker", "speedball", "steady"]
        return [order[i % len(order)] for i in range(horse_count)]
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    for p in parts:
        if p not in ARCHETYPES:
            raise SystemExit(
                f"Unknown archetype {p!r}. Valid: {', '.join(sorted(ARCHETYPES))}"
            )
    return [parts[i % len(parts)] for i in range(horse_count)]


def run_one_race(
    segments,
    horse_count: int,
    archetypes: list[str],
    *,
    randomize_attrs: bool,
    rng: random.Random,
) -> dict:
    """Run a single race; return summary dict."""
    if randomize_attrs:
        # create_randomized_attributes uses the global `random` module.
        # One seed per race gives reproducible attribute rolls across horses.
        random.seed(rng.randrange(1 << 30))
        attr_factories = {i: create_randomized_attributes for i in range(horse_count)}
    else:
        attr_factories = {i: create_default_attributes for i in range(horse_count)}

    race = Race(segments, horse_count, attr_factories=attr_factories)
    race.start(player_horse_id=None)

    strategies: dict[int, BehaviorTreeStrategy] = {}
    for hid in range(horse_count):
        name = archetypes[hid]
        cfg = ARCHETYPES[name]()
        strategies[hid] = BehaviorTreeStrategy(race, hid, cfg)

    guard = 0
    while race.state.phase == "running" and guard < MAX_TICKS_SAFETY:
        inputs: dict[int, InputState] = {}
        for hid, strat in strategies.items():
            h = race.state.horses[hid]
            if h.finished:
                continue
            inp = strat.act_continuous(h)
            if inp is not None:
                inputs[hid] = inp
        race.tick(inputs)
        guard += 1

    if race.state.phase != "finished":
        raise RuntimeError(
            f"Race did not finish in {MAX_TICKS_SAFETY} ticks "
            f"(phase={race.state.phase!r})"
        )

    rows = []
    for hid in race.state.finish_order:
        h = race.state.horses[hid]
        name = archetypes[hid]
        rows.append(
            {
                "horse_id": hid,
                "archetype": name,
                "place": h.finish_order,
                "stamina_frac": h.current_stamina / h.base_attributes.max_stamina
                if h.base_attributes.max_stamina > 0
                else 0.0,
                "ticks": race.state.tick,
            }
        )

    winner_arch = archetypes[race.state.finish_order[0]]
    return {
        "winner": winner_arch,
        "finish_order": [archetypes[hid] for hid in race.state.finish_order],
        "ticks": race.state.tick,
        "per_horse": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch BT races for win-rate / finish-position statistics."
    )
    parser.add_argument(
        "--track",
        type=Path,
        default=None,
        help=f"Track JSON (default: {_default_track().relative_to(ROOT)})",
    )
    parser.add_argument("--races", type=int, default=100, help="Number of races to run")
    parser.add_argument(
        "--horse-count",
        type=int,
        default=5,
        help="Number of horses (each gets a BT archetype, round-robin)",
    )
    parser.add_argument(
        "--archetypes",
        type=str,
        default=None,
        help='Comma-separated archetype names, or omit / "all" for default '
        f"round-robin. Valid: {', '.join(sorted(ARCHETYPES))}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master RNG seed for race order and per-race attr jitter (default: time-based)",
    )
    parser.add_argument(
        "--randomize-attrs",
        action="store_true",
        help="±10%% attribute jitter per horse (like training env opponents)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="If set, append one row per race (winner, ticks, finish_order)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each race finish order",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar (TTY: on by default)",
    )
    args = parser.parse_args()

    track_path = args.track or _default_track()
    if not track_path.is_absolute():
        track_path = ROOT / track_path
    if not track_path.exists():
        raise SystemExit(f"Track not found: {track_path}")

    segments = load_track_json(track_path)
    archetypes_per_lane = _parse_archetypes(args.archetypes, args.horse_count)

    seed = args.seed if args.seed is not None else int(time.time_ns() % (1 << 31))
    rng = random.Random(seed)
    print(f"Master seed: {seed}")

    wins: dict[str, int] = defaultdict(int)
    sum_place: dict[str, list[float]] = defaultdict(list)
    sum_ticks: list[int] = []

    csv_file = None
    csv_writer = None
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        new_file = not args.csv.exists()
        csv_file = open(args.csv, "a", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["race", "ticks", "winner", "finish_order"],
        )
        if new_file:
            csv_writer.writeheader()

    use_progress = (
        not args.no_progress
        and not args.verbose
        and sys.stderr.isatty()
    )

    try:
        race_iter = range(args.races)
        if use_progress:
            race_iter = tqdm(
                race_iter,
                desc="BT races",
                unit="race",
                file=sys.stderr,
                leave=True,
            )
        for r in race_iter:
            summary = run_one_race(
                segments,
                args.horse_count,
                archetypes_per_lane,
                randomize_attrs=args.randomize_attrs,
                rng=rng,
            )
            w = summary["winner"]
            wins[w] += 1
            sum_ticks.append(summary["ticks"])
            for row in summary["per_horse"]:
                sum_place[row["archetype"]].append(float(row["place"]))

            if use_progress:
                race_iter.set_postfix(
                    last_win=w[:12],
                    ticks=summary["ticks"],
                    refresh=False,
                )
            elif args.verbose:
                tqdm.write(
                    f"race {r + 1:4d}  ticks={summary['ticks']:4d}  "
                    f"winner={w:12s}  order={summary['finish_order']}"
                )

            if csv_writer is not None:
                csv_writer.writerow(
                    {
                        "race": r + 1,
                        "ticks": summary["ticks"],
                        "winner": w,
                        "finish_order": ";".join(summary["finish_order"]),
                    }
                )
    finally:
        if csv_file is not None:
            csv_file.close()

    n = args.races
    print(f"Track: {track_path.relative_to(ROOT)}")
    print(f"Races: {n}  horses: {args.horse_count}  randomize_attrs: {args.randomize_attrs}")
    print(f"Archetypes (slot 0..{args.horse_count - 1}): {archetypes_per_lane}")
    print(f"Mean ticks / race: {sum(sum_ticks) / len(sum_ticks):.1f}")
    print()
    print(f"{'Archetype':<14} {'Win%':>8} {'Mean place':>12} {'n (slots)':>10}")
    print("-" * 48)
    # Each archetype appears horse_count times per race as different slots —
    # reporting wins is by archetype name only (which archetype won).
    for name in sorted(ARCHETYPES.keys()):
        wr = 100.0 * wins[name] / n if n else 0.0
        places = sum_place.get(name, [])
        mp = sum(places) / len(places) if places else float("nan")
        print(f"{name:<14} {wr:8.1f} {mp:12.2f} {len(places):10d}")
    print()
    print(
        "Win% = fraction of races where that archetype (any horse slot) won. "
        "Mean place = average finishing position over all appearances."
    )
    if args.csv:
        print(f"Appended {n} rows to {args.csv}")


if __name__ == "__main__":
    main()
