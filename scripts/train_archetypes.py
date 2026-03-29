"""Train per-archetype jockey policies using real HKJC pace archetypes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from horse_racing.rllib_env import HorseRacingRLlibEnv
from horse_racing.reward import ARCHETYPES


# Default archetype assignment: one per horse
DEFAULT_ARCHETYPE_MAP = {
    "horse_0": "front_runner",
    "horse_1": "stalker",
    "horse_2": "closer",
    "horse_3": "presser",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train jockey archetype policies")
    parser.add_argument("--horse-count", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--track", type=str, default="tracks/tokyo.json")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--shared-warmup", type=int, default=50,
                        help="Shared policy warmup iterations before splitting")
    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", "archetypes")
    else:
        args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ray.init(runtime_env={"working_dir": None})

    archetype_map = {f"horse_{i}": ARCHETYPES[i % len(ARCHETYPES)]
                     for i in range(args.horse_count)}

    print(f"Archetype assignments:")
    for agent, arch in archetype_map.items():
        print(f"  {agent} → {arch}")

    # ===== Phase 1: Shared warmup =====
    if args.shared_warmup > 0:
        print(f"\n{'='*60}")
        print(f"Phase 1: Shared policy warmup ({args.shared_warmup} iterations)")
        print(f"Track: {args.track}")
        print(f"{'='*60}\n")

        config_warmup = (
            PPOConfig()
            .environment(
                env=HorseRacingRLlibEnv,
                env_config={
                    "track_path": args.track,
                    "horse_count": args.horse_count,
                    "max_steps": 5000,
                    # No archetypes during warmup — learn base racing first
                },
            )
            .env_runners(num_env_runners=0)
            .training(
                train_batch_size=4000,
                minibatch_size=256,
                num_epochs=10,
                lr=3e-4,
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

        algo_warmup = config_warmup.build_algo()

        for i in range(args.shared_warmup):
            result = algo_warmup.train()
            env_runners = result.get("env_runners", {})
            mean_reward = env_runners.get("episode_reward_mean", 0.0)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Warmup Iter {i+1:4d} | reward: {mean_reward:8.2f}")

        shared_state = algo_warmup.get_module("shared_policy").state_dict()
        warmup_path = str(checkpoint_dir / "warmup_shared")
        algo_warmup.save(warmup_path)
        print(f"\nWarmup saved to {warmup_path}")
        algo_warmup.stop()
    else:
        shared_state = None

    # ===== Phase 2: Per-archetype training =====
    print(f"\n{'='*60}")
    print(f"Phase 2: Per-archetype training ({args.iterations} iterations)")
    print(f"Track: {args.track}")
    for agent, arch in archetype_map.items():
        print(f"  {agent} → {arch}")
    print(f"{'='*60}\n")

    policy_ids = [f"horse_{i}_policy" for i in range(args.horse_count)]

    config_arch = (
        PPOConfig()
        .environment(
            env=HorseRacingRLlibEnv,
            env_config={
                "track_path": args.track,
                "horse_count": args.horse_count,
                "max_steps": 5000,
                "archetypes": archetype_map,  # each agent gets its archetype reward
            },
        )
        .env_runners(num_env_runners=0)
        .training(
            train_batch_size=4000,
            minibatch_size=256,
            num_epochs=10,
            lr=1e-4,  # lower LR for archetype fine-tuning
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

    # Initialize from shared warmup weights
    if shared_state is not None:
        for pid in policy_ids:
            module = algo_arch.get_module(pid)
            module.load_state_dict(shared_state, strict=False)
        print("Initialized all policies from shared warmup weights")

    for i in range(args.iterations):
        result = algo_arch.train()
        env_runners = result.get("env_runners", {})
        mean_reward = env_runners.get("episode_reward_mean", 0.0)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Archetype Iter {i+1:4d} | reward: {mean_reward:8.2f}")

    arch_path = str(checkpoint_dir / "archetypes_final")
    algo_arch.save(arch_path)
    print(f"\nArchetype training saved to {arch_path}")

    # Export per-archetype ONNX models
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
    print("Archetype training complete!")
    print(f"Models: {checkpoint_dir}/jockey_*.onnx")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
