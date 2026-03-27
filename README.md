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
