# Horse Racing RL

Reinforcement learning multi-agents for horse racing simulation. This Python environment reimplements a TypeScript simulation that runs in the browser, with physics that must match exactly for validation.

## Setup

```bash
uv sync
uv sync --extra dev   # include test dependencies
```

## Running Tests

```bash
uv run pytest
```

## Training

```bash
uv run python scripts/train.py --track tracks/simple_oval.json --total-timesteps 1000000
```

## Validation Against JS Server

Start the JS validation server at `http://localhost:3456`, then:

```bash
uv run python scripts/validate.py --track tracks/simple_oval.json --steps 1000
```

## Observation Vector Sync (Python ↔ Browser)

The Python training code and browser ONNX inference must build identical observation
vectors. A shared schema at `obs_schema.json` defines the canonical layout.

### Workflow for changing the observation vector

1. **Update `obs_schema.json` first** — add/remove/reorder fields, bump the `version`.
2. **Update Python** — edit `engine.py:obs_to_array()` to match the new schema.
   Update `env.py`, `multi_agent_env.py`, `rllib_env.py` observation space shapes.
3. **Update browser** — edit `ai-jockey.ts:observationToArray()` to match.
   Update tensor shapes in `AIJockey` and `AIJockeyManager`.
   If new fields were added to `HorseObservation`, update the type and the
   observation build in `horse-racing-engine.ts`.
4. **Update ONNX export scripts** — `scripts/export_onnx*.py` dummy input dimensions.
5. **Run tests** — `uv run pytest` verifies `test_obs_array_matches_schema` passes.

Key files:
- `obs_schema.json` — single source of truth for field order and count
- `horse_racing/engine.py:obs_to_array()` — Python observation builder
- Browser: `src/simulation/ai-jockey.ts:observationToArray()` — browser observation builder

## Project Structure

- `horse_racing/` — Core simulation package
  - `types.py` — Track segments, physics constants, shared data structures
  - `attributes.py` — CoreAttributes, trait ranges, modifier resolution
  - `genome.py` — HorseGenome, gene expression, breeding
  - `modifiers.py` — 8 built-in modifier definitions and conditions
  - `stamina.py` — Stamina depletion/recovery with cornering threshold
  - `track.py` — Track JSON parsing
  - `track_navigator.py` — Per-horse segment tracking and local frame computation
  - `physics.py` — 2D physics integration and collision resolution
  - `engine.py` — HorseRacingEngine core simulation loop
  - `env.py` — Single-agent Gymnasium environment
  - `multi_agent_env.py` — PettingZoo ParallelEnv for multi-agent training
  - `reward.py` — Modular reward function
  - `validation.py` — Trajectory comparison against JS server
- `tracks/` — Track JSON files
- `scripts/` — Training and validation entry points
- `tests/` — Test suite
