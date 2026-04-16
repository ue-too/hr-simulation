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

## Tuning the utility-based BT (batch statistics)

Run headless races where every horse uses `BehaviorTreeStrategy` with a fixed
archetype lineup, then inspect win rates and mean finish position. Use this to
see whether an archetype dominates, never wins, or looks balanced after you
change weights in `horse_racing/opponents/behavior_tree.py`.

```bash
# 200 races on the default test oval, reproducible seed, default archetypes per lane
uv run python scripts/tune_bt.py --races 200 --seed 42

# Same physics variety as training env opponents (±10% attributes)
uv run python scripts/tune_bt.py --races 500 --randomize-attrs --seed 42

# Custom track and six horses (archetypes repeat in round-robin order)
uv run python scripts/tune_bt.py --track tracks/simple_oval.json --horse-count 6 --races 300

# Explicit archetype per slot: repeats if fewer names than horses
uv run python scripts/tune_bt.py --archetypes closer,speedball,stalker,front-runner --races 100

# Append results for spreadsheet analysis
uv run python scripts/tune_bt.py --races 1000 --csv artifacts/bt_tune_runs.csv

# Plain output (no progress bar — useful when piping or in CI)
uv run python scripts/tune_bt.py --races 200 --no-progress
```

While running, a **progress bar** appears on stderr (race count, last winner,
ticks). It is off automatically when stderr is not a TTY, when `--verbose` is
set, or when you pass `--no-progress`.

Interpretation: **Win%** is how often that archetype finished first (any slot).
**Mean place** is the average finishing position across all appearances of that
archetype in the batch. Tune one archetype at a time, re-run the script, and
compare before/after. See `horse_racing/opponents/JOCKEY.md` for what each
parameter does.

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

## Visualizing Race Lines

Simulate races on all real tracks and produce a race-line plot colored by acceleration state (green = accelerating, blue = cruising, red = decelerating):

```bash
uv run python scripts/plot_race_lines.py                # uses latest model version
uv run python scripts/plot_race_lines.py --version v9    # specify a version
```

Output is saved to `artifacts/race_lines_{version}.png`. Each track is shown with both inner-start and outer-start positions. The plot includes tick counts and acceleration/cruise/brake breakdowns.

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
