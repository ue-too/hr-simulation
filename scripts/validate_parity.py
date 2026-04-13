"""Compare Python v2 simulation against TS Phase 3 server for parity."""

import argparse
import sys

import requests

from horse_racing.core.race import Race
from horse_racing.core.track import load_track_json
from horse_racing.core.types import InputState

EPSILON = 1e-6


def run_python(track_path: str, horse_count: int, actions: list[dict], num_ticks: int):
    """Run simulation in Python and collect per-tick state."""
    segments = load_track_json(track_path)
    race = Race(segments, horse_count)
    race.start(None)

    history = []
    for t in range(num_ticks):
        inputs: dict[int, InputState] = {}
        for a in actions:
            inputs[a["horse_id"]] = InputState(a["tangential"], a["normal"])
        race.tick(inputs)

        tick_state = []
        for h in race.state.horses:
            tick_state.append({
                "id": h.id,
                "x": float(h.pos[0]),
                "y": float(h.pos[1]),
                "tangential_vel": h.tangential_vel,
                "normal_vel": h.normal_vel,
                "stamina": h.current_stamina,
                "progress": h.track_progress,
            })
        history.append(tick_state)
    return history


def run_ts_server(
    server_url: str, track_path: str, horse_count: int,
    actions: list[dict], num_ticks: int,
):
    """Run simulation via TS Phase 3 server and collect per-tick state."""
    # Reset
    resp = requests.post(f"{server_url}/reset", json={
        "trackPath": track_path,
        "horseCount": horse_count,
    })
    resp.raise_for_status()

    history = []
    for t in range(num_ticks):
        action_map = {}
        for a in actions:
            action_map[str(a["horse_id"])] = {
                "tangential": a["tangential"],
                "normal": a["normal"],
            }
        resp = requests.post(f"{server_url}/step", json={"actions": action_map})
        resp.raise_for_status()
        data = resp.json()
        history.append(data["horses"])
    return history


def compare(py_history, ts_history, epsilon: float) -> list[str]:
    """Compare two histories tick by tick. Return list of mismatch descriptions."""
    errors = []
    for t, (py_tick, ts_tick) in enumerate(zip(py_history, ts_history)):
        for py_h, ts_h in zip(py_tick, ts_tick):
            hid = py_h["id"]
            for key in ["x", "y", "tangential_vel", "normal_vel", "stamina", "progress"]:
                py_val = py_h[key]
                ts_key = key
                # TS server may use camelCase
                if ts_key == "tangential_vel":
                    ts_key = "tangentialVel"
                elif ts_key == "normal_vel":
                    ts_key = "normalVel"
                ts_val = ts_h.get(ts_key, ts_h.get(key))
                if ts_val is None:
                    errors.append(f"tick {t}, horse {hid}: missing key '{key}' in TS response")
                    continue
                if abs(py_val - ts_val) > epsilon:
                    errors.append(
                        f"tick {t}, horse {hid}, {key}: "
                        f"Python={py_val:.10f} TS={ts_val:.10f} "
                        f"diff={abs(py_val - ts_val):.2e}"
                    )
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate Python-TS parity")
    parser.add_argument("--track", type=str, required=True, help="Path to track JSON")
    parser.add_argument("--horses", type=int, default=2, help="Number of horses")
    parser.add_argument("--ticks", type=int, default=60, help="Number of ticks to compare")
    parser.add_argument("--server", type=str, default="http://localhost:3001",
                        help="TS server URL")
    parser.add_argument("--epsilon", type=float, default=EPSILON, help="Tolerance")
    args = parser.parse_args()

    # Fixed actions: all horses cruise (no input)
    actions = [{"horse_id": i, "tangential": 0.0, "normal": 0.0} for i in range(args.horses)]

    print(f"Running Python simulation ({args.ticks} ticks, {args.horses} horses)...")
    py_history = run_python(args.track, args.horses, actions, args.ticks)

    print(f"Running TS server simulation ({args.server})...")
    try:
        ts_history = run_ts_server(args.server, args.track, args.horses, actions, args.ticks)
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to TS server at {args.server}")
        print("Start the server with: bun run dev:horse-racing (in the TS monorepo)")
        sys.exit(1)

    print("Comparing...")
    errors = compare(py_history, ts_history, args.epsilon)

    if errors:
        print(f"\nFAILED: {len(errors)} mismatches found:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        sys.exit(1)
    else:
        print(f"\nPASSED: All {args.ticks} ticks match within ε={args.epsilon}")


if __name__ == "__main__":
    main()
