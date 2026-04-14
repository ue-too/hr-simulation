# Phase 1: Stamina Intelligence

**Goal:** Replace the binary exhaustion cliff with a gradual degradation model so the RL agent can learn realistic pacing — conserve mid-race, sprint at the end — across any track length.

**Problem:** v15 either burns stamina early and hits the binary exhaustion wall, or doesn't understand the cliff at all. The binary model (full stats above 0 stamina, sudden floor at 0) provides no gradient for the agent to learn from. A previous attempt at smooth linear degradation was too lenient — the agent could push hard the whole race without meaningful penalty.

## 1. Gradual Exhaustion Model

**File:** `horse_racing/core/exhaustion.py`

Replace the binary exhaustion model with a sigmoid-knee curve that provides smooth gradient everywhere but punishes low stamina steeply.

### Formula

```python
def effective_ratio(stamina_pct: float, knee: float, floor: float, k: float) -> float:
    """Map stamina percentage (0-1) to stat multiplier (floor-1.0).

    Uses a sigmoid curve with a tunable knee point.
    - Above the knee: stats degrade gently
    - Below the knee: stats drop steeply
    - At 0 stamina: stats bottom out at `floor`
    """
    # Sigmoid centered on knee, scaled to [floor, 1.0]
    raw = 1.0 / (1.0 + math.exp(-k * (stamina_pct - knee)))
    # Normalize so that sigmoid(1.0) maps to 1.0 and sigmoid(0.0) maps to floor
    sig_at_1 = 1.0 / (1.0 + math.exp(-k * (1.0 - knee)))
    sig_at_0 = 1.0 / (1.0 + math.exp(-k * (0.0 - knee)))
    normalized = (raw - sig_at_0) / (sig_at_1 - sig_at_0)
    return floor + (1.0 - floor) * normalized
```

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `knee` | 0.28 | Inflection point — degradation accelerates below ~28% stamina |
| `floor` | 0.45 | Minimum stat ratio at 0 stamina — enough for a weak kick |
| `k` | 10 | Steepness — makes the transition sharp around the knee |

### Expected Stat Curve

| Stamina % | Effective Ratio | Meaning |
|-----------|----------------|---------|
| 100% | 1.00 | Full performance |
| 80% | ~0.997 | Barely noticeable |
| 50% | ~0.94 | Starting to feel it |
| 30% | ~0.74 | Steep drop, danger zone |
| 10% | ~0.50 | Almost at floor |
| 0% | 0.45 | Floor — can still move, just slow |

### Affected Stats

The `effective_ratio` multiplier applies to four stats:
- `cruise_speed`
- `max_speed`
- `forward_accel`
- `turn_accel`

### Implementation

Replace `apply_exhaustion()`. Instead of checking `if stamina > 0` (binary), compute effective attributes as:

```python
def apply_exhaustion(horse: Horse) -> CoreAttributes:
    base = horse.base_attributes
    stamina_pct = horse.current_stamina / base.max_stamina
    ratio = effective_ratio(stamina_pct, KNEE, FLOOR, K)
    return dataclasses.replace(
        base,
        cruise_speed=base.cruise_speed * ratio,
        max_speed=base.max_speed * ratio,
        forward_accel=base.forward_accel * ratio,
        turn_accel=base.turn_accel * ratio,
    )
```

This removes the `EXHAUSTION_DECAY` exponential decay logic and the per-stat floor ratios. The sigmoid curve handles both the degradation shape and the floor.

## 2. Track-Length-Normalized Drain

**File:** `horse_racing/core/stamina.py`

Scale all drain rates by a normalization factor so that a cruise-effort race drains roughly the same total stamina percentage regardless of track length.

### Approach

Compute a `drain_scale` factor at race init:

```python
REFERENCE_CRUISE_TICKS = 2000  # calibrated from test_oval at cruise speed

def compute_drain_scale(track_total_length: float, cruise_speed: float) -> float:
    """Scale drain so cruise-effort stamina usage is consistent across tracks."""
    estimated_ticks = track_total_length / (cruise_speed * FIXED_DT * PHYS_SUBSTEPS)
    return REFERENCE_CRUISE_TICKS / estimated_ticks
```

The `drain_scale` is multiplied into the final drain value in `drain_stamina()`, after all drain terms are summed but before applying to stamina:

```python
drain *= attrs.drain_rate_mult
drain *= drain_scale  # NEW: track-length normalization
horse.current_stamina = max(0.0, horse.current_stamina - drain)
```

### Threading

- `Race.__init__` computes `drain_scale` from track total length and default cruise speed
- `Race.tick()` passes `drain_scale` to `drain_stamina()`
- `drain_stamina()` accepts `drain_scale` as a new parameter (default 1.0 for backward compat)

**File changes:** `horse_racing/core/race.py` — compute and pass `drain_scale`

## 3. Reward Shaping

**File:** `horse_racing/reward.py`

### Remove

- `EXHAUSTION_PENALTY = -0.002` — the gradual model now handles this through stat degradation, making the flat per-tick penalty redundant

### Add

- **Stamina efficiency bonus at finish:** When the horse finishes, add a small bonus based on how efficiently it used stamina. This encourages the agent to use its stamina budget rather than hoarding or wasting it.

```python
STAMINA_EFFICIENCY_BONUS = 1.0

def compute_reward(
    prev_progress: float,
    curr_progress: float,
    finish_order: int | None,
    current_stamina: float = 1.0,
    max_stamina: float = 100.0,
) -> float:
    reward = curr_progress - prev_progress

    if finish_order is not None:
        reward += _FINISH_BONUS.get(finish_order, 0.0)
        # Reward efficient stamina use: best reward near 0% remaining
        stamina_pct = current_stamina / max_stamina
        reward += STAMINA_EFFICIENCY_BONUS * (1.0 - stamina_pct)

    return reward
```

The efficiency bonus rewards finishing with low stamina (you used what you had) but only triggers at finish — it doesn't discourage mid-race conservation. Combined with the gradual exhaustion model, the agent must balance: burn too early and your stats tank in the final stretch, but hoard too much and you leave reward on the table.

### Keep

- Delta-progress reward (`curr_progress - prev_progress`)
- Finish-order bonus (`_FINISH_BONUS`)

## 4. Observation Enhancement

**File:** `horse_racing/core/observation.py`

Add one new observation: the stamina drain rate from the previous tick.

### New field

| Index | Field | Description |
|-------|-------|-------------|
| 139 | `stamina_drain_rate` | Drain from the previous tick, normalized by `max_stamina`. Gives the agent direct signal about how current effort maps to stamina cost. |

`OBS_SIZE` changes from 139 to 140.

### Implementation

The drain rate needs to be tracked on the `Horse` dataclass or passed through the environment. Simplest approach: add a `last_drain` field to `Horse` (default 0.0), set it in `drain_stamina()`, and read it in `build_observations()`.

```python
# In observation.py, after opponent slots:
obs[139] = self_horse.last_drain / base.max_stamina
```

**File changes:**
- `horse_racing/core/types.py` — add `last_drain: float = 0.0` to `Horse`
- `horse_racing/core/stamina.py` — set `horse.last_drain = drain` before applying
- `horse_racing/core/observation.py` — read `last_drain`, bump `OBS_SIZE`

## 5. Environment Changes

**File:** `horse_racing/env/single_env.py`

- Update `observation_space` shape to `(140,)` (follows from OBS_SIZE change)
- Pass `max_stamina` to `compute_reward()` for the efficiency bonus
- No other env changes needed — drain scale is handled internally by `Race`

## 6. What Stays the Same

- Action space (9x9, 81 discrete actions)
- Physics model (`physics.py`)
- Opponent strategies (`scripted.py`)
- Track geometry and collision system
- Training setup (PPO, hyperparams, notebook structure)
- All other observation fields (0-138)

## 7. Files Changed Summary

| File | Change |
|------|--------|
| `horse_racing/core/exhaustion.py` | Replace binary model with sigmoid-knee curve |
| `horse_racing/core/stamina.py` | Add `drain_scale` param, set `horse.last_drain` |
| `horse_racing/core/types.py` | Add `last_drain: float = 0.0` to `Horse` |
| `horse_racing/core/race.py` | Compute `drain_scale` from track length, pass to `drain_stamina()` |
| `horse_racing/core/observation.py` | Add `stamina_drain_rate` at index 139, `OBS_SIZE` 139 -> 140 |
| `horse_racing/reward.py` | Remove exhaustion penalty, add stamina efficiency bonus at finish |
| `horse_racing/env/single_env.py` | Pass `max_stamina` to reward, update obs shape |

## 8. Tuning Knobs

After implementation, these are the primary values to tune across training iterations:

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `knee` | `exhaustion.py` | 0.28 | Where degradation accelerates |
| `floor` | `exhaustion.py` | 0.45 | Minimum stat ratio at 0 stamina |
| `k` | `exhaustion.py` | 10 | Steepness of transition |
| `STAMINA_EFFICIENCY_BONUS` | `reward.py` | 1.0 | How much finishing with low stamina matters |
| `REFERENCE_CRUISE_TICKS` | `stamina.py` | 2000 | Baseline for drain normalization |
