# Utility-Scored Jockey (behavior_tree.py)

The scripted opponent AI uses a **utility-scored selector** with **committed
maneuvers** and a **reactive defensive overlay**. It reads only the same
observation vector the RL agent sees — no privileged access to race state.

This implementation mirrors the TypeScript version in the web sim
(`apps/horse-racing/src/ai/bt-jockey.ts`) so trained RL policies face
identical opponent behavior in both environments.

## Architecture at a glance

```
                   ┌──────────────────────┐
                   │   Observation Vector  │
                   │ obs[0]  progress      │
                   │ obs[1]  speed ratio   │
                   │ obs[3]  stamina frac  │
                   │ obs[15] lateral norm  │
                   │ obs[26+] opp slots    │
                   └──────────┬───────────┘
                              │
                   ┌──────────▼───────────┐
                   │   Utility Selector    │
                   │                       │
                   │  _score_cruise(obs)   │
                   │  _score_pass(obs)     │
                   │  _score_kick(progress,│
                   │              stamina) │
                   │                       │
                   │  highest score wins   │
                   └──────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         ┌─────────┐   ┌──────────┐   ┌──────────┐
         │  CRUISE  │──▶│ PASSING  │──▶│ SETTLING │
         │          │   │(committed│   │(lane lerp│
         │          │   │ min ticks│   │ back)    │
         └────┬─────┘   └──────────┘   └──────────┘
              │
              ▼
         ┌─────────┐
         │  KICK   │  (absorbing — never exits)
         └─────────┘

         ── Defensive overlay applied to ALL outputs ──
```

## States

| State | Purpose | Duration |
|---|---|---|
| **CRUISE** | Hold speed band, steer toward archetype target lane. | Until utility selects another action. |
| **PASSING** | Swing wide + accelerate to overtake a slower blocker. | Committed for `pass_min_ticks` (default 40). |
| **SETTLING** | After a pass, interpolate lateral position back toward archetype lane. | `settle_ticks` (default 40). |
| **KICK** | Final sprint at max tangential. | Absorbing — once entered, permanent. |

KICK can be entered from any state. PASSING and SETTLING can also be
interrupted by the forced kick late-cap (`kick_late_cap`).

## Utility scoring

Each tick, when the horse is in CRUISE and not committed to a maneuver, three
scores are computed:

### _score_cruise

```
score = 1.0  (baseline)
if drafting:
    score += (0.2 + (1 - stamina_frac) * 0.3) * w_draft
```

The draft bonus makes cruising behind another horse more attractive, especially
when tired. Archetypes with high `w_draft` (closers, stalkers) are more
inclined to stay tucked in.

### _score_pass

Scans opponent slots for a **blocker**: an opponent that is ahead
(`0 < progress_delta < block_progress_max`), in the same lane
(`|normal_offset| < block_lateral_tol`), and moving meaningfully slower
(`tvel_delta < -block_min_slowness`).

```
score = (0.3 + severity * 5.0 - lateral_cost * 2.0) * w_pass
```

Returns `-10` (never pass) when no blocker is detected. The pass cooldown and
transition budget further gate when passing is allowed.

### _score_kick

Replaces the old hard threshold `progress >= kick_phase` with a
stamina-aware window:

```
early_phase = kick_phase - kick_early_margin
late_phase  = min(kick_phase + kick_early_margin, kick_late_cap)

if progress < early_phase  →  -10  (too early, never kick)
if progress >= late_phase  →  10   (forced kick regardless of stamina)

otherwise:
    sustainability = stamina_frac - remaining * 1.5
    if sustainability <= 0  →  -1  (can't sustain, don't kick yet)
    score = (0.5 + sustainability * 3.0) * w_kick
```

This gives closers (high `kick_phase`, high `w_kick`) a late explosive finish
while front-runners (low `kick_phase`) kick earlier. The `kick_late_cap`
ensures every horse kicks eventually, even if stamina is depleted.

### Selection rule

```
if kick_u >= cruise_u AND kick_u >= pass_u AND kick_u > 0 → KICK
if pass_u > cruise_u AND pass_u > 0 AND can_transition → PASSING
otherwise → stay in CRUISE
```

## Defensive overlay

The defensive overlay is **not** a state — it's a post-process applied to
whatever action the current state produces. It prevents opponents from passing
by contesting their line.

**Detection** — `_compute_threat_score` scans opponents for:
- Slightly behind: `-0.03 <= progress_delta < 0`
- Gaining speed: `tvel_delta > 0.03`
- Swinging wide: `normal_offset > 0.05`

**Hysteresis** prevents flickering:
- Activate defense when threat score exceeds `defend_on_score` (default 0.6)
- Deactivate when it drops below `defend_off_score` (default 0.3)

**Effect** when defending (and `stamina_frac >= 0.30`):
- Tangential is raised to at least `defend_tang_min` (default 0.5)
- Normal is nudged outward by `defend_drift` (default 0.15)

## Oscillation guardrails

| Mechanism | What it prevents |
|---|---|
| **Pass commitment** (`pass_min_ticks`) | Aborting a pass mid-maneuver. |
| **Settle period** (`settle_ticks`) | Rail-snap after completing a pass. |
| **Pass cooldown** (`pass_cooldown_ticks`) | Repeated passing attempts. |
| **Transition budget** (`transition_min_ticks`) | Rapid CRUISE / PASSING flipping. |
| **Defense hysteresis** (`defend_on_score` / `defend_off_score`) | Frame-by-frame defense toggling. |
| **KICK is absorbing** | No exit from the final sprint. |

## Drafting stamina bonus

Separate from the jockey AI, the simulation itself (`race.py` / `stamina.py`)
provides a physics-level drafting bonus:

- **Detection** (`Race._compute_draft_bonus`): another horse is 0.5%-5% ahead
  on track progress, within 1 meter laterally, and at similar or higher speed.
- **Effect**: all stamina drain components are reduced by 15% for that tick.
- **Cancelled** when steering hard (`|input.normal| >= 0.3`), so weaving
  through the field doesn't get free draft.

The jockey's `_score_cruise` independently detects drafting via the observation
vector and boosts the cruise score, making the horse *choose* to stay tucked in.
These are complementary: the jockey decides to draft, the physics rewards it.

## Archetype profiles

Each archetype is a factory function returning a `BTConfig` with specific
overrides. The key differentiators:

| Archetype | Cruise band | Target lane | Kick timing | Personality |
|---|---|---|---|---|
| **Stalker** | 0.55 - 0.70 | -0.60 (just off rail) | 0.75 | Balanced; high draft value. Sits mid-pack, conserves energy. |
| **Front-runner** | 0.72 - 0.85 | -0.80 (inside rail) | 0.65 | Pushes early, defends position, kicks early but may fade. |
| **Closer** | 0.40 - 0.52 | -0.30 (2-3 wide) | 0.85 | Very conservative cruise; sits wide; explosive late kick. |
| **Speedball** | 0.60 - 0.75 | -0.20 (sits wide) | 0.70 | Aggressive passer (high w_pass), moves through the field constantly. |
| **Steady** | 0.58 - 0.68 | -0.70 (near rail) | 0.80 | Narrow band, rarely passes, rarely defends. Mid-pack finisher. |

### Lane preferences explained

`target_lane` is a normalized lateral offset where -0.95 is the inside rail
and 0 is the track center. The `_steer_to_lane` helper uses a bang-bang
controller with a 0.05 dead zone, scaled by `lateral_aggression`.

A closer sitting at -0.30 (2-3 wide) is immediately recognizable to anyone who
watches racing. After a pass, the SETTLING state smoothly interpolates back
toward the archetype lane over `settle_ticks` instead of snapping.

## Configuration reference

| Field | Default | Description |
|---|---|---|
| `cruise_low` / `cruise_high` | 0.55 / 0.70 | Speed ratio band (tvel / max_speed). |
| `target_lane` | -0.80 | Preferred lateral position (normalized). |
| `lateral_aggression` | 0.6 | Steering strength toward target lane (0-1). |
| `kick_phase` | 0.75 | Center of the kick timing window. |
| `kick_early_margin` | 0.10 | How much earlier than `kick_phase` kick can happen. |
| `kick_late_cap` | 0.92 | Forced kick regardless of stamina. |
| `block_progress_max` | 0.03 | Max progress delta to consider an opponent a blocker. |
| `block_lateral_tol` | 0.15 | Max lateral offset for blocker detection. |
| `block_min_slowness` | 0.03 | Blocker must be at least this much slower. |
| `conserve_threshold` | 0.30 | Stamina below this caps tangential at 0.25. |
| `pass_min_ticks` | 40 | Minimum committed ticks in PASSING. |
| `pass_clear_lateral` | 0.25 | Lateral offset threshold for "clear of blocker". |
| `pass_cooldown_ticks` | 80 | Ticks after a pass before another can start. |
| `settle_ticks` | 40 | Ticks to interpolate back to archetype lane. |
| `transition_min_ticks` | 30 | Minimum ticks between non-absorbing transitions. |
| `defend_on_score` | 0.6 | Threat score to activate defense. |
| `defend_off_score` | 0.3 | Threat score to deactivate defense. |
| `defend_tang_min` | 0.5 | Minimum tangential when defending. |
| `defend_drift` | 0.15 | Outward lateral nudge when defending. |
| `w_pass` | 1.0 | Weight multiplier on pass utility score. |
| `w_kick` | 1.0 | Weight multiplier on kick utility score. |
| `w_draft` | 1.0 | Weight multiplier on cruise-while-drafting bonus. |

## Systematic tuning (Python batch runner)

Use `scripts/tune_bt.py` to run many headless races and print aggregate **win
rates** and **mean finish position** per archetype. Pass `--randomize-attrs` to
match training-env opponent variety (±10% stats). Example:

```bash
uv run python scripts/tune_bt.py --races 200 --seed 42
uv run python scripts/tune_bt.py --races 500 --randomize-attrs --csv artifacts/bt_tune.csv
```

After changing weights in `BTConfig` or an archetype factory, re-run the same
command and compare tables. See the project `README.md` for all CLI flags.

## TS / Python parity

This file (`behavior_tree.py`) and the web sim version (`bt-jockey.ts`) are
kept in sync. They share:
- Identical observation indices and opponent slot layout.
- Identical utility formulas and threshold constants.
- Identical state machine transitions and commitment rules.

The only structural difference is that the Python version is **per-horse**
(one `BehaviorTreeStrategy` instance per horse), while the TS version manages
all horses from a single `BTJockey` instance via an internal state map.
