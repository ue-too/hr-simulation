"""Run validation against JS server."""

from __future__ import annotations

import argparse

from horse_racing.validation import validate_zero_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Python engine against JS server")
    parser.add_argument(
        "--track",
        type=str,
        default="tracks/exp_track_8.json",
        help="Path to track JSON file",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--server", type=str, default="http://localhost:3456")
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    print(f"Validating zero-action trajectory on {args.track}")
    print(f"JS server: {args.server}")
    print(f"Steps: {args.steps}, tolerance: {args.tolerance}")
    print()

    result = validate_zero_actions(
        track_path=args.track,
        steps=args.steps,
        js_server_url=args.server,
        tolerance=args.tolerance,
    )

    print(f"Max divergence:  {result.max_divergence:.6f} (step {result.max_divergence_step}, horse {result.max_divergence_horse})")
    print(f"Mean divergence: {result.mean_divergence:.6f}")
    print(f"Result:          {'PASS' if result.passed else 'FAIL'}")


if __name__ == "__main__":
    main()
