"""Full training pipeline: curriculum → archetype specialization.

Phase 1 (Curriculum): Shared policy learns base racing on progressively
harder tracks. All 4 horses share one network.

Phase 2 (Archetypes): Each horse gets its own policy, initialized from
the shared weights, then fine-tuned with archetype-specific reward shaping.
Exports per-archetype ONNX models for browser inference.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from horse_racing.rllib_env import HorseRacingRLlibEnv
from horse_racing.reward import ARCHETYPES


# ---------------------------------------------------------------------------
# Curriculum stages
# ---------------------------------------------------------------------------

CURRICULUM = [
    {
        "track": "tracks/curriculum_1_straight.json",
        "iterations": 50,
        "max_steps": 1500,
        "name": "Stage 1: Straight",
    },
    {
        "track": "tracks/curriculum_2_gentle_oval.json",
        "iterations": 100,
        "max_steps": 3000,
        "name": "Stage 2: Gentle oval",
    },
    {
        "track": "tracks/curriculum_3_tight_oval.json",
        "iterations": 100,
        "max_steps": 3000,
        "name": "Stage 3: Tight oval",
    },
    {
        "track": "tracks/exp_track_8.json",
        "iterations": 150,
        "max_steps": 5000,
        "name": "Stage 4: Complex track",
    },
]


def build_shared_config(
    track_path: str,
    horse_count: int,
    max_steps: int,
    lr: float = 3e-4,
) -> PPOConfig:
    return (
        PPOConfig()
        .environment(
            env=HorseRacingRLlibEnv,
            env_config={
                "track_path": track_path,
                "horse_count": horse_count,
                "max_steps": max_steps,
            },
        )
        .env_runners(num_env_runners=0)
        .training(
            train_batch_size=4000,
            minibatch_size=256,
            num_epochs=10,
            lr=lr,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
        )
        .framework("torch")
        .multi_agent(
            policies={"shared_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *a, **kw: "shared_policy",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full training: curriculum + archetype specialization"
    )
    parser.add_argument("--horse-count", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--skip-curriculum", action="store_true",
                        help="Skip curriculum, go straight to archetypes")
    parser.add_argument("--resume-curriculum", type=str, default=None,
                        help="Resume curriculum from this checkpoint path")
    parser.add_argument("--resume-stage", type=int, default=1,
                        help="Resume from this curriculum stage (1-indexed)")
    parser.add_argument("--archetype-iterations", type=int, default=200)
    parser.add_argument("--archetype-track", type=str,
                        default="tracks/exp_track_8.json")
    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", "full_pipeline")
    else:
        args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ray.init(runtime_env={"working_dir": None})

    # ===================================================================
    # Phase 1: Curriculum — shared policy, progressive tracks
    # ===================================================================
    shared_state = None

    if not args.skip_curriculum:
        print("\n" + "=" * 60)
        print("PHASE 1: CURRICULUM TRAINING (shared policy)")
        print("=" * 60)

        algo = None

        for stage_idx, stage in enumerate(CURRICULUM):
            stage_num = stage_idx + 1
            if stage_num < args.resume_stage:
                continue

            print(f"\n{'─'*60}")
            print(f"{stage['name']}")
            print(f"Track: {stage['track']}  |  Iterations: {stage['iterations']}")
            print(f"{'─'*60}\n")

            config = build_shared_config(
                track_path=stage["track"],
                horse_count=args.horse_count,
                max_steps=stage["max_steps"],
            )

            if algo is None:
                if args.resume_curriculum:
                    print(f"Resuming from {args.resume_curriculum}")
                    algo = config.build_algo()
                    algo.restore(args.resume_curriculum)
                else:
                    algo = config.build_algo()
            else:
                # Transfer: rebuild with new env config, restore weights
                state = algo.get_module("shared_policy").state_dict()
                algo.stop()
                algo = config.build_algo()
                algo.get_module("shared_policy").load_state_dict(state, strict=False)

            best_reward = float("-inf")
            for i in range(stage["iterations"]):
                result = algo.train()
                env_runners = result.get("env_runners", {})
                mean_reward = env_runners.get("episode_reward_mean", 0.0)

                if (i + 1) % 10 == 0 or i == 0:
                    print(
                        f"  [{stage['name']}] Iter {i+1:4d} | "
                        f"reward: {mean_reward:8.2f}"
                    )

                if mean_reward > best_reward:
                    best_reward = mean_reward

            save_path = str(checkpoint_dir / f"curriculum_stage_{stage_num}")
            algo.save(save_path)
            print(f"  Saved → {save_path}  (best reward: {best_reward:.2f})")

        shared_state = algo.get_module("shared_policy").state_dict()
        algo.stop()
        print("\nCurriculum complete.")

    # ===================================================================
    # Phase 2: Archetype specialization — per-horse policies
    # ===================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: ARCHETYPE TRAINING (per-horse policies)")
    print("=" * 60)

    archetype_map = {
        f"horse_{i}": ARCHETYPES[i % len(ARCHETYPES)]
        for i in range(args.horse_count)
    }
    for agent, arch in archetype_map.items():
        print(f"  {agent} → {arch}")

    policy_ids = [f"horse_{i}_policy" for i in range(args.horse_count)]

    config_arch = (
        PPOConfig()
        .environment(
            env=HorseRacingRLlibEnv,
            env_config={
                "track_path": args.archetype_track,
                "horse_count": args.horse_count,
                "max_steps": 5000,
                "archetypes": archetype_map,
            },
        )
        .env_runners(num_env_runners=0)
        .training(
            train_batch_size=4000,
            minibatch_size=256,
            num_epochs=10,
            lr=1e-4,  # lower LR for fine-tuning
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
        )
        .framework("torch")
        .multi_agent(
            policies={pid: PolicySpec() for pid in policy_ids},
            policy_mapping_fn=lambda agent_id, *a, **kw: f"{agent_id}_policy",
        )
    )

    algo_arch = config_arch.build_algo()

    # Initialize from curriculum shared weights
    if shared_state is not None:
        for pid in policy_ids:
            module = algo_arch.get_module(pid)
            module.load_state_dict(shared_state, strict=False)
        print("\nInitialized all policies from curriculum weights")
    else:
        print("\nNo curriculum weights — training archetypes from scratch")

    print(f"\nTraining {args.archetype_iterations} iterations on {args.archetype_track}\n")

    best_reward = float("-inf")
    for i in range(args.archetype_iterations):
        result = algo_arch.train()
        env_runners = result.get("env_runners", {})
        mean_reward = env_runners.get("episode_reward_mean", 0.0)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Archetype Iter {i+1:4d} | reward: {mean_reward:8.2f}")

        if mean_reward > best_reward:
            best_reward = mean_reward

    arch_path = str(checkpoint_dir / "archetypes_final")
    algo_arch.save(arch_path)
    print(f"\nArchetype training saved → {arch_path}")

    # ===================================================================
    # Export per-archetype ONNX models
    # ===================================================================
    print("\nExporting per-archetype ONNX models...")
    import torch

    for idx, pid in enumerate(policy_ids):
        module = algo_arch.get_module(pid)
        module.eval()
        arch_name = archetype_map[f"horse_{idx}"]

        class PolicyWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, obs):
                result = self.m.forward_inference({"obs": obs})
                return result["action_dist_inputs"][:, :2]

        wrapper = PolicyWrapper(module)
        wrapper.eval()
        dummy = torch.zeros(1, 26, dtype=torch.float32)
        onnx_path = str(checkpoint_dir / f"jockey_{arch_name}.onnx")

        torch.onnx.export(
            wrapper, dummy, onnx_path,
            input_names=["obs"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
        print(f"  {arch_name} → {onnx_path}")

    algo_arch.stop()
    ray.shutdown()

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"ONNX models: {checkpoint_dir}/jockey_*.onnx")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
