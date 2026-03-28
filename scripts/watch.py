"""Watch a trained model race — prints live progress bar in terminal."""

from __future__ import annotations

import argparse
import time
import sys

import numpy as np
from stable_baselines3 import PPO

from horse_racing.env import HorseRacingSingleEnv


HORSE_SYMBOLS = ["🟡", "🟤", "🔵", "⚪"]
BAR_WIDTH = 60


def render_race(progresses: list[float], speeds: list[float], step: int, reward: float):
    """Clear screen and draw race state."""
    sys.stdout.write("\033[H\033[J")  # clear screen
    print(f"Step: {step:>5d}  |  Total Reward: {reward:>8.2f}")
    print(f"{'─' * (BAR_WIDTH + 20)}")

    for i, (prog, spd) in enumerate(zip(progresses, speeds)):
        pos = int(prog * BAR_WIDTH)
        pos = min(pos, BAR_WIDTH - 1)
        bar = "░" * pos + HORSE_SYMBOLS[i % len(HORSE_SYMBOLS)] + "░" * (BAR_WIDTH - pos - 1)
        label = "YOU " if i == 0 else f"H{i}  "
        print(f"  {label} |{bar}| {prog:>5.1%}  {spd:>5.1f} m/s")

    print(f"{'─' * (BAR_WIDTH + 20)}")
    print("Horse 0 = trained agent, others = auto-cruise")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Watch trained model race")
    parser.add_argument("--model", type=str, default="checkpoints/straight_sb3")
    parser.add_argument("--track", type=str, default="tracks/curriculum_1_straight.json")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = HorseRacingSingleEnv(track_path=args.track, max_steps=args.max_steps)
    obs, _ = env.reset()

    total_reward = 0.0
    tick_delay = (1.0 / 30) / args.speed  # ~30 FPS at 1x speed

    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Get all horses' progress and speed from engine
        all_obs = env.engine.get_observations()
        progresses = [o["track_progress"] for o in all_obs]
        speeds = [o["tangential_vel"] for o in all_obs]

        render_race(progresses, speeds, step, total_reward)

        if terminated:
            print("\n  🏁 FINISHED!")
            break
        if truncated:
            print("\n  ⏱️  Time's up!")
            break

        time.sleep(tick_delay)

    print(f"\nFinal: progress={progresses[0]:.1%}, speed={speeds[0]:.1f} m/s, reward={total_reward:.2f}")


if __name__ == "__main__":
    main()
