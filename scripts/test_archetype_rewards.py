"""Test that archetype reward bonuses push behavior in the right direction.

Simulates a race with 4 horses (each archetype) and logs per-tick archetype
bonuses at different race phases to verify the reward signal makes sense.
"""

from __future__ import annotations

import numpy as np

from horse_racing.engine import HorseRacingEngine
from horse_racing.reward import ARCHETYPES, _archetype_bonus, compute_reward
from horse_racing.types import HorseAction


def main():
    engine = HorseRacingEngine("tracks/hanshin.json")
    prev_obs = engine.get_observations()
    prev_placements = engine.get_placements()

    # Accumulate archetype bonuses by race phase
    phase_bonuses = {arch: {"early": [], "mid": [], "late": []} for arch in ARCHETYPES}
    phase_rewards = {arch: {"early": [], "mid": [], "late": []} for arch in ARCHETYPES}

    for tick in range(3000):
        all_obs = engine.get_observations()
        placements = engine.get_placements()

        # Give each horse slightly different actions to create position differences
        actions = []
        for i, arch in enumerate(ARCHETYPES):
            o = all_obs[i]
            # Front-runner pushes hard, closer conserves, etc.
            if arch == "front_runner":
                tang = 5.0
            elif arch == "stalker":
                tang = 2.0
            elif arch == "closer":
                tang = -2.0
            else:  # presser
                tang = 3.0
            actions.append(HorseAction(tang, 0.0))

        engine.step(actions)
        curr_obs = engine.get_observations()
        curr_placements = engine.get_placements()

        for i, arch in enumerate(ARCHETYPES):
            progress = curr_obs[i]["track_progress"]
            if curr_obs[i]["finished"]:
                continue

            bonus = _archetype_bonus(
                arch, prev_obs[i], curr_obs[i],
                curr_placements[i], 4, progress,
            )
            reward = compute_reward(
                prev_obs[i], curr_obs[i], curr_obs[i]["collision"],
                placement=curr_placements[i], num_horses=4,
                archetype=arch,
                prev_placement=prev_placements[i],
            )

            if progress < 0.5:
                phase = "early"
            elif progress < 0.75:
                phase = "mid"
            else:
                phase = "late"
            phase_bonuses[arch][phase].append(bonus)
            phase_rewards[arch][phase].append(reward)

        prev_obs = curr_obs
        prev_placements = curr_placements

    # Print summary
    print("=" * 70)
    print("Archetype Reward Analysis — Hanshin Track")
    print("=" * 70)
    print()
    print(f"{'Archetype':>15s} | {'Phase':>6s} | {'Arch Bonus':>12s} | {'Total Reward':>12s} | {'Ticks':>6s}")
    print("-" * 70)

    for arch in ARCHETYPES:
        for phase in ["early", "mid", "late"]:
            bonuses = phase_bonuses[arch][phase]
            rewards = phase_rewards[arch][phase]
            if not bonuses:
                continue
            mean_b = np.mean(bonuses)
            mean_r = np.mean(rewards)
            print(f"{arch:>15s} | {phase:>6s} | {mean_b:>+12.4f} | {mean_r:>+12.4f} | {len(bonuses):>6d}")
        print()

    # Verify expected patterns
    print("Expected patterns:")
    print("  front_runner: high early bonus (leading), lower late")
    print("  stalker:      moderate early (2nd-3rd), high late (takes lead)")
    print("  closer:       positive early (stamina conservation), high late (kick)")
    print("  presser:      consistent bonus throughout (above cruise)")


if __name__ == "__main__":
    main()
