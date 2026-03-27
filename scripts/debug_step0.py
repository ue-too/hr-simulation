"""Debug script: compare step 0 positions between Python and JS."""

import httpx
import numpy as np

from horse_racing.engine import HorseRacingEngine
from horse_racing.types import HORSE_COUNT, HORSE_RADIUS, HorseAction, PHYS_SUBSTEPS, PHYS_HZ
from horse_racing.physics import resolve_horse_collisions, resolve_wall_collisions, integrate


TRACK = "tracks/exp_track_8.json"
JS_SERVER = "http://localhost:3456"


def main():
    engine = HorseRacingEngine(TRACK)

    print("=== Initial Positions ===")
    for i, hs in enumerate(engine.horses):
        print(f"  horse_{i}: pos=({hs.body.position[0]:.6f}, {hs.body.position[1]:.6f})")

    # Check initial overlaps
    print("\n=== Initial Horse Distances ===")
    for i in range(HORSE_COUNT):
        for j in range(i + 1, HORSE_COUNT):
            d = np.linalg.norm(engine.horses[j].body.position - engine.horses[i].body.position)
            overlap = 2 * HORSE_RADIUS - d
            status = f"OVERLAP by {overlap:.3f}" if overlap > 0 else "OK"
            print(f"  ({i},{j}): dist={d:.3f} {status}")

    # Manual substep trace for first 2 substeps
    actions = [HorseAction() for _ in range(HORSE_COUNT)]
    engine._resolve_attributes()

    # Stamina update
    for i, hs in enumerate(engine.horses):
        hs.frame = hs.navigator.update(hs.body.position)

    dt = 1.0 / PHYS_HZ

    for sub in range(min(3, PHYS_SUBSTEPS)):
        print(f"\n=== Substep {sub} ===")

        # Track frame + forces
        for i, hs in enumerate(engine.horses):
            hs.frame = hs.navigator.update(hs.body.position)
            hs.body.clear_force()
            force = engine._compute_force(hs, actions[i])
            hs.body.apply_force(force)

        print("  After force application:")
        for i, hs in enumerate(engine.horses):
            print(f"    horse_{i}: force=({hs.body.force[0]:.4f}, {hs.body.force[1]:.4f})")

        # Collision resolution
        bodies = [hs.body for hs in engine.horses]
        masses = [hs.effective_attrs.weight for hs in engine.horses]
        collided = resolve_horse_collisions(bodies, masses)

        print("  After horse-horse collision:")
        for i, hs in enumerate(engine.horses):
            print(f"    horse_{i}: pos=({hs.body.position[0]:.6f}, {hs.body.position[1]:.6f})"
                  f"  vel=({hs.body.velocity[0]:.6f}, {hs.body.velocity[1]:.6f})"
                  f"  collided={collided[i]}")

        resolve_wall_collisions(
            bodies, engine.segments,
            [hs.navigator.segment_index for hs in engine.horses],
        )

        print("  After wall collision:")
        for i, hs in enumerate(engine.horses):
            print(f"    horse_{i}: pos=({hs.body.position[0]:.6f}, {hs.body.position[1]:.6f})")

        # Integration
        for hs in engine.horses:
            integrate(hs.body, hs.effective_attrs.weight, dt)
            hs.body.clear_force()

        print("  After integration:")
        for i, hs in enumerate(engine.horses):
            print(f"    horse_{i}: pos=({hs.body.position[0]:.6f}, {hs.body.position[1]:.6f})"
                  f"  vel=({hs.body.velocity[0]:.6f}, {hs.body.velocity[1]:.6f})")

    # Now do a clean run to compare with JS
    engine2 = HorseRacingEngine(TRACK)
    engine2.step(actions)

    print("\n=== Python After Full Step 0 ===")
    for i, hs in enumerate(engine2.horses):
        print(f"  horse_{i}: pos=({hs.body.position[0]:.6f}, {hs.body.position[1]:.6f})"
              f"  vel=({hs.body.velocity[0]:.6f}, {hs.body.velocity[1]:.6f})")

    # JS
    payload = {
        "track": "exp_track_8",
        "actions": [[{"extraTangential": 0, "extraNormal": 0}] * HORSE_COUNT],
        "steps": 1,
    }
    resp = httpx.post(f"{JS_SERVER}/simulate", json=payload, timeout=10.0)
    resp.raise_for_status()
    js_data = resp.json()["trajectories"]

    print("\n=== JS After Step 0 ===")
    for i, h in enumerate(js_data[0]):
        print(f"  horse_{i}: pos=({h['x']:.6f}, {h['y']:.6f})"
              f"  vel=({h['vx']:.6f}, {h['vy']:.6f})")

    print("\n=== Divergence ===")
    for i, hs in enumerate(engine2.horses):
        js = js_data[0][i]
        dx = hs.body.position[0] - js["x"]
        dy = hs.body.position[1] - js["y"]
        dvx = hs.body.velocity[0] - js["vx"]
        dvy = hs.body.velocity[1] - js["vy"]
        print(f"  horse_{i}: dpos=({dx:.6f}, {dy:.6f}) dvel=({dvx:.6f}, {dvy:.6f})")


if __name__ == "__main__":
    main()
