"""Trajectory comparison against JS server for physics validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from horse_racing.engine import HorseRacingEngine
from horse_racing.types import HORSE_COUNT, HorseAction


def validate_obs_schema(schema_path: str | Path = "obs_schema.json") -> None:
    """Verify Python obs_to_array matches the shared observation schema.

    Raises AssertionError on mismatch. Call from CI or pre-commit to catch
    drift between Python and browser observation vectors.
    """
    from horse_racing.modifiers import MODIFIER_IDS

    with open(schema_path) as f:
        schema = json.load(f)

    engine = HorseRacingEngine("tracks/test_oval.json")
    engine.step([HorseAction() for _ in range(HORSE_COUNT)])
    obs = engine.get_observations()
    arr = engine.obs_to_array(obs[0])

    assert arr.shape[0] == schema["size"], (
        f"Python obs size {arr.shape[0]} != schema size {schema['size']}"
    )

    # Verify modifier IDs in schema match Python MODIFIER_IDS order
    schema_mod_fields = [
        f["name"].removeprefix("mod_")
        for f in schema["fields"]
        if f["name"].startswith("mod_")
    ]
    assert schema_mod_fields == MODIFIER_IDS, (
        f"Schema modifier order {schema_mod_fields} != Python {MODIFIER_IDS}"
    )


@dataclass
class ValidationResult:
    max_divergence: float
    max_divergence_step: int
    max_divergence_horse: int
    mean_divergence: float
    max_stamina_divergence: float
    max_stamina_step: int
    max_stamina_horse: int
    passed: bool  # within tolerance


def run_python_engine(
    track_path: str | Path,
    actions: list[list[HorseAction]],
) -> list[list[dict]]:
    """Run actions through the Python engine and return per-step, per-horse state."""
    engine = HorseRacingEngine(track_path)
    trajectories: list[list[dict]] = []

    for step_actions in actions:
        engine.step(step_actions)
        obs = engine.get_observations()
        trajectories.append(
            [
                {
                    "x": float(hs.body.position[0]),
                    "y": float(hs.body.position[1]),
                    "vx": float(hs.body.velocity[0]),
                    "vy": float(hs.body.velocity[1]),
                    "currentStamina": hs.runtime.current_stamina,
                    "effectiveCruiseSpeed": hs.effective_attrs.cruise_speed,
                    "effectiveMaxSpeed": hs.effective_attrs.max_speed,
                    "trackProgress": hs.track_progress,
                    "activeModifierIds": sorted(
                        m.id for m in hs.runtime.active_modifiers
                    ),
                }
                for hs, o in zip(engine.horses, obs)
            ]
        )

    return trajectories


def query_js_server(
    js_server_url: str,
    track_name: str,
    actions: list[list[dict]],
    steps: int,
) -> list[list[dict]]:
    """Query the JS validation server and return trajectories."""
    payload = {
        "track": track_name,
        "actions": actions,
        "steps": steps,
    }
    resp = httpx.post(f"{js_server_url}/simulate", json=payload, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    return data["trajectories"]


def validate_trajectory(
    track_path: str | Path,
    actions: list[list[HorseAction]],
    js_server_url: str = "http://localhost:3456",
    tolerance: float = 0.01,
) -> ValidationResult:
    """Run the same action sequence through Python and JS, compare trajectories."""
    # Run Python engine
    py_trajectories = run_python_engine(track_path, actions)

    # Convert actions to JS format
    track_name = Path(track_path).stem
    js_actions = [
        [
            {"extraTangential": a.extra_tangential, "extraNormal": a.extra_normal}
            for a in step_actions
        ]
        for step_actions in actions
    ]

    # Query JS server
    js_trajectories = query_js_server(js_server_url, track_name, js_actions, len(actions))

    # Compare position and stamina
    max_divergence = 0.0
    max_step = 0
    max_horse = 0
    total_divergence = 0.0
    max_stamina_div = 0.0
    max_stamina_step = 0
    max_stamina_horse = 0
    count = 0

    num_steps = min(len(py_trajectories), len(js_trajectories))
    for step in range(num_steps):
        num_horses = min(len(py_trajectories[step]), len(js_trajectories[step]))
        for horse in range(num_horses):
            py = py_trajectories[step][horse]
            js = js_trajectories[step][horse]

            # Position divergence
            dx = py["x"] - js["x"]
            dy = py["y"] - js["y"]
            dist = (dx**2 + dy**2) ** 0.5

            total_divergence += dist
            count += 1

            if dist > max_divergence:
                max_divergence = dist
                max_step = step
                max_horse = horse

            # Stamina divergence
            stamina_diff = abs(
                py.get("currentStamina", 0) - js.get("currentStamina", 0)
            )
            if stamina_diff > max_stamina_div:
                max_stamina_div = stamina_diff
                max_stamina_step = step
                max_stamina_horse = horse

    mean_divergence = total_divergence / count if count > 0 else 0.0

    return ValidationResult(
        max_divergence=max_divergence,
        max_divergence_step=max_step,
        max_divergence_horse=max_horse,
        mean_divergence=mean_divergence,
        max_stamina_divergence=max_stamina_div,
        max_stamina_step=max_stamina_step,
        max_stamina_horse=max_stamina_horse,
        passed=max_divergence <= tolerance and max_stamina_div <= tolerance,
    )


def validate_zero_actions(
    track_path: str | Path,
    steps: int = 1000,
    js_server_url: str = "http://localhost:3456",
    tolerance: float = 0.01,
) -> ValidationResult:
    """Validate with zero-action trajectories (pure auto-cruise)."""
    actions = [[HorseAction() for _ in range(HORSE_COUNT)] for _ in range(steps)]
    return validate_trajectory(track_path, actions, js_server_url, tolerance)
