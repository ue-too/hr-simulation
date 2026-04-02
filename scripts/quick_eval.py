#!/usr/bin/env python3
"""Quick evaluation of a trained checkpoint on a single track.

Usage:
    python scripts/quick_eval.py --model checkpoints/baseline/stage_1.zip --track tracks/curriculum_1_straight.json
    python scripts/quick_eval.py --model checkpoints/baseline/stage_4.zip --track tracks/tokyo.json --stamina-profile
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from stable_baselines3 import PPO

from horse_racing.env import HorseRacingSingleEnv


def run_episode(model: PPO, track_path: str, max_steps: int, stamina_profile: bool):
    env = HorseRacingSingleEnv(track_path=track_path, max_steps=max_steps)
    obs, _ = env.reset()

    total_reward = 0.0
    speeds = []
    min_stamina = 1.0
    stamina_at_progress = {}
    checkpoints = {0.25, 0.50, 0.75, 1.00}

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        o = env.engine.get_observations()[0]
        speed = o["tangential_vel"]
        speeds.append(speed)
        stamina = o["stamina_ratio"]
        if stamina < min_stamina:
            min_stamina = stamina

        if stamina_profile:
            progress = o["track_progress"]
            for cp in list(checkpoints):
                if progress >= cp:
                    stamina_at_progress[cp] = stamina
                    checkpoints.discard(cp)

        if terminated or truncated:
            finished = o.get("finished", False)
            progress = o["track_progress"]
            break
    else:
        finished = False
        progress = env.engine.horses[0].track_progress

    env.close()
    return {
        "finished": finished,
        "progress": progress,
        "reward": total_reward,
        "avg_speed": np.mean(speeds) if speeds else 0,
        "min_stamina": min_stamina,
        "final_stamina": o["stamina_ratio"],
        "steps": step + 1,
        "stamina_profile": stamina_at_progress,
    }


def main():
    parser = argparse.ArgumentParser(description="Quick eval of a checkpoint")
    parser.add_argument("--model", type=str, required=True, help="SB3 checkpoint path")
    parser.add_argument("--track", type=str, required=True, help="Track JSON path")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--stamina-profile", action="store_true",
                        help="Print stamina at 25/50/75/100%% progress")
    args = parser.parse_args()

    model = PPO.load(args.model)
    print(f"Model: {args.model}")
    print(f"Track: {args.track}")
    print()

    results = []
    for ep in range(args.episodes):
        r = run_episode(model, args.track, args.max_steps, args.stamina_profile)
        results.append(r)
        status = "OK" if r["finished"] else f"{r['progress']:.0%}"
        sys.stdout.write(f"  ep {ep + 1}: {status} | reward={r['reward']:.0f} | "
                         f"speed={r['avg_speed']:.1f} | stamina={r['final_stamina']:.0%}\n")

    print()
    comp = sum(1 for r in results if r["finished"]) / len(results)
    print(f"  Completion: {comp:.0%}")
    print(f"  Mean reward: {np.mean([r['reward'] for r in results]):.1f}")
    print(f"  Mean speed: {np.mean([r['avg_speed'] for r in results]):.1f} m/s")
    print(f"  Mean final stamina: {np.mean([r['final_stamina'] for r in results]):.0%}")
    print(f"  Worst stamina: {min(r['min_stamina'] for r in results):.0%}")

    if args.stamina_profile:
        print()
        print("  Stamina profile (mean across episodes):")
        for cp in [0.25, 0.50, 0.75, 1.00]:
            vals = [r["stamina_profile"].get(cp) for r in results if cp in r["stamina_profile"]]
            if vals:
                print(f"    {cp:5.0%} progress → {np.mean(vals):.0%} stamina")


if __name__ == "__main__":
    main()
