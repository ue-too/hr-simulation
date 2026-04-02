#!/usr/bin/env python3
"""Simulate v9 baseline on all real tracks and plot race lines colored by acceleration state.

Two rows per track: inner start (default) and outer start.

Colors:
  - Green:  accelerating (extra_tangential > threshold)
  - Red:    decelerating (extra_tangential < -threshold)
  - Blue:   cruising (in between)

Outputs: artifacts/race_lines_v9.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import onnxruntime as ort

from horse_racing.engine import HorseRacingEngine
from horse_racing.types import (
    HorseAction,
    HORSE_COUNT,
    HORSE_SPACING,
    TRACK_HALF_WIDTH,
)

# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH = "models/v9/baseline.onnx"
ACCEL_THRESHOLD = 0.15  # action magnitude below this = cruising
MAX_TICKS = 7000

REAL_TRACKS = [
    ("Tokyo", "tracks/tokyo.json"),
    ("Kokura", "tracks/kokura.json"),
    ("Tokyo 2600", "tracks/tokyo_2600.json"),
    ("Hanshin", "tracks/hanshin.json"),
    ("Kyoto", "tracks/kyoto.json"),
    ("Niigata", "tracks/niigata.json"),
]


# ── Track centerline extraction ───────────────────────────────────────
def get_track_centerline(track_path: str) -> np.ndarray:
    with open(track_path) as f:
        data = json.load(f)
    segments = data["segments"] if isinstance(data, dict) else data

    points: list[tuple[float, float]] = []
    for seg in segments:
        if seg["tracktype"] == "STRAIGHT":
            sx, sy = seg["startPoint"]["x"], seg["startPoint"]["y"]
            ex, ey = seg["endPoint"]["x"], seg["endPoint"]["y"]
            length = math.hypot(ex - sx, ey - sy)
            n = max(10, int(length / 2))
            for t in np.linspace(0, 1, n, endpoint=False):
                points.append((sx + t * (ex - sx), sy + t * (ey - sy)))
        elif seg["tracktype"] == "CURVE":
            cx, cy = seg["center"]["x"], seg["center"]["y"]
            r = seg["radius"]
            sx, sy = seg["startPoint"]["x"], seg["startPoint"]["y"]
            start_angle = math.atan2(sy - cy, sx - cx)
            span = seg["angleSpan"]
            arc_len = abs(r * span)
            n = max(10, int(arc_len / 2))
            for t in np.linspace(0, 1, n, endpoint=False):
                a = start_angle + t * span
                points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    if points:
        points.append(points[0])
    return np.array(points)


# ── Move horse 0 to outer rail ───────────────────────────────────────
def _move_horse_to_outer(engine: HorseRacingEngine) -> None:
    """Reposition horse 0 from inner rail to outer rail."""
    hs = engine.horses[0]
    frame = hs.navigator.update(hs.body.position)
    # Inner edge is at -(TRACK_HALF_WIDTH - HORSE_SPACING)
    # Outer edge is at +(TRACK_HALF_WIDTH - HORSE_SPACING)
    outer_edge = TRACK_HALF_WIDTH - HORSE_SPACING
    inner_edge = -(TRACK_HALF_WIDTH - HORSE_SPACING)
    # Current lateral offset = inner_edge (horse 0 is innermost)
    shift = outer_edge - inner_edge
    hs.body.position = hs.body.position + frame.normal * shift
    # Re-orient along track at new position
    frame = hs.navigator.update(hs.body.position)
    hs.body.orientation = math.atan2(frame.tangential[1], frame.tangential[0])


# ── Simulate one race ─────────────────────────────────────────────────
def simulate_race(
    sess: ort.InferenceSession,
    track_path: str,
    outer_start: bool = False,
):
    engine = HorseRacingEngine(track_path)
    if outer_start:
        _move_horse_to_outer(engine)

    positions = []
    accel_actions = []

    for tick in range(MAX_TICKS):
        obs = engine.get_observations()
        arr = engine.obs_to_array(obs[0]).reshape(1, -1)
        action = sess.run(["action"], {"obs": arr})[0][0]

        extra_tang = float(action[0])
        accel_actions.append(extra_tang)

        horse = engine.horses[0]
        positions.append((float(horse.body.position[0]), float(horse.body.position[1])))

        actions = [HorseAction(float(action[0]), float(action[1]))]
        for _ in range(1, HORSE_COUNT):
            actions.append(HorseAction())
        engine.step(actions)

        if engine.horses[0].finished:
            hf = engine.horses[0]
            positions.append((float(hf.body.position[0]), float(hf.body.position[1])))
            accel_actions.append(extra_tang)
            break

    finished = engine.horses[0].finished
    return np.array(positions), np.array(accel_actions), finished


# ── Colored line segments ─────────────────────────────────────────────
def make_colored_segments(positions, accel_actions, threshold):
    points = positions.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = []
    for a in accel_actions[:-1]:
        if a > threshold:
            colors.append((0.2, 0.8, 0.2, 0.9))
        elif a < -threshold:
            colors.append((0.9, 0.2, 0.2, 0.9))
        else:
            colors.append((0.3, 0.5, 0.9, 0.9))

    return segments, colors


def draw_race(ax, track_path, track_name, sess, outer_start, label):
    centerline = get_track_centerline(track_path)
    ax.plot(centerline[:, 0], centerline[:, 1], color="#ddd", linewidth=8,
            solid_capstyle="round", zorder=1)
    ax.plot(centerline[:, 0], centerline[:, 1], color="#bbb", linewidth=1,
            linestyle="--", zorder=2)

    positions, accel_actions, finished = simulate_race(sess, track_path, outer_start)
    segments, colors = make_colored_segments(positions, accel_actions, ACCEL_THRESHOLD)

    lc = mcoll.LineCollection(segments, colors=colors, linewidths=2.0, zorder=3)
    ax.add_collection(lc)

    ax.plot(positions[0, 0], positions[0, 1], "o", color="white",
            markersize=8, markeredgecolor="black", markeredgewidth=1.5, zorder=5)
    marker_color = "gold" if finished else "red"
    ax.plot(positions[-1, 0], positions[-1, 1], "s", color=marker_color,
            markersize=8, markeredgecolor="black", markeredgewidth=1.5, zorder=5)

    ax.set_title(f"{track_name} — {label}", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f0")
    ax.grid(True, alpha=0.3)

    n_accel = np.sum(accel_actions > ACCEL_THRESHOLD)
    n_decel = np.sum(accel_actions < -ACCEL_THRESHOLD)
    n_cruise = len(accel_actions) - n_accel - n_decel
    total = len(accel_actions)
    status = "Finished" if finished else "DNF"
    ax.text(0.02, 0.98,
            f"{status} | Ticks: {total}\n"
            f"Accel: {n_accel/total:.0%}  Cruise: {n_cruise/total:.0%}  Brake: {n_decel/total:.0%}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


# ── Main ──────────────────────────────────────────────────────────────
def main():
    sess = ort.InferenceSession(MODEL_PATH)
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    tracks = [(name, path) for name, path in REAL_TRACKS if Path(path).exists()]
    n_tracks = len(tracks)

    # 2 columns per track: inner start, outer start
    # Layout: n_tracks rows x 2 columns
    fig, axes = plt.subplots(n_tracks, 2, figsize=(14, 5.5 * n_tracks))
    if n_tracks == 1:
        axes = axes.reshape(1, -1)

    for row, (track_name, track_path) in enumerate(tracks):
        print(f"Simulating {track_name} (inner start)...")
        draw_race(axes[row, 0], track_path, track_name, sess,
                  outer_start=False, label="Inner Start")

        print(f"Simulating {track_name} (outer start)...")
        draw_race(axes[row, 1], track_path, track_name, sess,
                  outer_start=True, label="Outer Start")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=(0.2, 0.8, 0.2), linewidth=3, label="Accelerating"),
        Line2D([0], [0], color=(0.3, 0.5, 0.9), linewidth=3, label="Cruising"),
        Line2D([0], [0], color=(0.9, 0.2, 0.2), linewidth=3, label="Decelerating"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=8, label="Start"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gold",
               markeredgecolor="black", markersize=8, label="Finish"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=8, label="DNF"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6,
               fontsize=11, frameon=True, fancybox=True)

    fig.suptitle("v9 Baseline — Race Lines: Inner vs Outer Start",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path = output_dir / "race_lines_v9.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
