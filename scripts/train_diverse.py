"""Per-agent policy training — shared-then-specialize pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from horse_racing.rllib_env import HorseRacingRLlibEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train diverse per-agent horse racing policies")
    parser.add_argument("--horse-count", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--phase-a-track", type=str, default="tracks/curriculum_2_gentle_oval.json")
    parser.add_argument("--phase-a-iters", type=int, default=50, help="Shared policy iterations")
    parser.add_argument("--phase-b-track", type=str, default="tracks/exp_track_8.json")
    parser.add_argument("--phase-b-iters", type=int, default=100, help="Per-agent policy iterations")
    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", "diverse")
    else:
        args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ray.init(runtime_env={"working_dir": None})

    # ===== Phase A: Shared policy =====
    print(f"\n{'='*60}")
    print(f"Phase A: Shared policy training ({args.phase_a_iters} iterations)")
    print(f"Track: {args.phase_a_track}")
    print(f"{'='*60}\n")

    config_a = (
        PPOConfig()
        .environment(
            env=HorseRacingRLlibEnv,
            env_config={
                "track_path": args.phase_a_track,
                "horse_count": args.horse_count,
                "max_steps": 3000,
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

    algo_a = config_a.build_algo()

    for i in range(args.phase_a_iters):
        result = algo_a.train()
        env_runners = result.get("env_runners", {})
        mean_reward = env_runners.get("episode_reward_mean", 0.0)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Phase A Iter {i+1:4d} | reward: {mean_reward:8.2f}")

    phase_a_path = str(checkpoint_dir / "phase_a_shared")
    algo_a.save(phase_a_path)
    print(f"\nPhase A saved to {phase_a_path}")

    # Get shared policy weights for initializing per-agent policies
    shared_module = algo_a.get_module("shared_policy")
    shared_state = shared_module.state_dict()
    algo_a.stop()

    # ===== Phase B: Per-agent policies =====
    print(f"\n{'='*60}")
    print(f"Phase B: Per-agent policy training ({args.phase_b_iters} iterations)")
    print(f"Track: {args.phase_b_track}")
    print(f"{'='*60}\n")

    policy_ids = [f"horse_{i}_policy" for i in range(args.horse_count)]

    config_b = (
        PPOConfig()
        .environment(
            env=HorseRacingRLlibEnv,
            env_config={
                "track_path": args.phase_b_track,
                "horse_count": args.horse_count,
                "max_steps": 5000,
            },
        )
        .env_runners(num_env_runners=0)
        .training(
            train_batch_size=4000,
            minibatch_size=256,
            num_epochs=10,
            lr=1e-4,  # Lower LR for fine-tuning
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

    algo_b = config_b.build_algo()

    # Initialize all per-agent policies with the shared policy weights
    for pid in policy_ids:
        module = algo_b.get_module(pid)
        module.load_state_dict(shared_state, strict=False)
    print("Initialized per-agent policies from shared weights")

    for i in range(args.phase_b_iters):
        result = algo_b.train()
        env_runners = result.get("env_runners", {})
        mean_reward = env_runners.get("episode_reward_mean", 0.0)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Phase B Iter {i+1:4d} | reward: {mean_reward:8.2f}")

    phase_b_path = str(checkpoint_dir / "phase_b_diverse")
    algo_b.save(phase_b_path)
    print(f"\nPhase B saved to {phase_b_path}")

    # Export each policy to ONNX
    print("\nExporting per-agent ONNX models...")
    import torch

    for idx, pid in enumerate(policy_ids):
        module = algo_b.get_module(pid)
        module.eval()

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
        onnx_path = str(checkpoint_dir / f"horse_{idx}.onnx")

        torch.onnx.export(
            wrapper, dummy, onnx_path,
            input_names=["obs"], output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
        print(f"  Exported {pid} → {onnx_path}")

    algo_b.stop()
    ray.shutdown()

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Per-agent models: {checkpoint_dir}/horse_0.onnx ... horse_{args.horse_count-1}.onnx")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
