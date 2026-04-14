# Stamina Intelligence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the binary exhaustion cliff with a sigmoid-knee degradation curve, normalize stamina drain by track length, reshape rewards for pacing, and add drain-rate observation — so the RL agent can learn realistic pacing.

**Architecture:** The exhaustion model becomes a pure function of stamina percentage (sigmoid curve with tunable knee/floor/steepness). Stamina drain gets a track-length normalization factor computed at race init. The reward drops the flat exhaustion penalty in favor of a finish-time efficiency bonus. One new observation field (drain rate) is appended.

**Tech Stack:** Python, NumPy, Gymnasium, pytest

**Spec:** `docs/superpowers/specs/2026-04-14-stamina-intelligence-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `horse_racing/core/exhaustion.py` | Rewrite | Sigmoid-knee curve replaces binary model |
| `tests/test_exhaustion.py` | Rewrite | Tests for new curve behavior |
| `horse_racing/core/types.py` | Modify | Add `last_drain` field to `Horse` |
| `horse_racing/core/stamina.py` | Modify | Add `drain_scale` param, record `last_drain` |
| `tests/test_stamina.py` | Modify | Add tests for `drain_scale` and `last_drain` |
| `horse_racing/core/race.py` | Modify | Compute `drain_scale`, pass to `drain_stamina()` |
| `tests/test_race.py` | Modify | Verify drain normalization in integration |
| `horse_racing/reward.py` | Rewrite | Remove exhaustion penalty, add efficiency bonus |
| `tests/test_reward.py` | Rewrite | Tests for new reward structure |
| `horse_racing/core/observation.py` | Modify | Add `last_drain` obs at index 139, bump OBS_SIZE |
| `tests/test_observation.py` | Modify | Update size assertions, test new field |
| `horse_racing/env/single_env.py` | Modify | Pass `max_stamina` to reward |
| `tests/test_env.py` | Modify | Update OBS_SIZE references |

---

### Task 1: Sigmoid-knee exhaustion model

**Files:**
- Rewrite: `horse_racing/core/exhaustion.py`
- Rewrite: `tests/test_exhaustion.py`

- [ ] **Step 1: Write failing tests for the new exhaustion model**

Replace the contents of `tests/test_exhaustion.py` with:

```python
import math

import numpy as np
import pytest

from horse_racing.core.attributes import create_default_attributes
from horse_racing.core.exhaustion import apply_exhaustion, effective_ratio
from horse_racing.core.types import Horse


def make_horse(**kwargs) -> Horse:
    attrs = create_default_attributes()
    defaults = dict(
        id=0, color=0, pos=np.array([0.0, 0.0]),
        tangential_vel=0.0, normal_vel=0.0, track_progress=0.0,
        navigator=None, finished=False, finish_order=None,
        base_attributes=attrs, current_stamina=attrs.max_stamina,
        effective_attributes=attrs,
    )
    defaults.update(kwargs)
    return Horse(**defaults)


class TestEffectiveRatio:
    def test_full_stamina_returns_one(self):
        assert effective_ratio(1.0) == pytest.approx(1.0, abs=1e-3)

    def test_zero_stamina_returns_floor(self):
        assert effective_ratio(0.0) == pytest.approx(0.45, abs=1e-3)

    def test_monotonically_increasing(self):
        values = [effective_ratio(p / 100) for p in range(101)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

    def test_knee_region_has_steep_drop(self):
        """Between 40% and 20% stamina, ratio should drop significantly."""
        high = effective_ratio(0.4)
        low = effective_ratio(0.2)
        assert high - low > 0.15

    def test_above_knee_gentle(self):
        """Between 80% and 60% stamina, ratio barely changes."""
        high = effective_ratio(0.8)
        low = effective_ratio(0.6)
        assert high - low < 0.05

    def test_floor_is_respected(self):
        """Even at 0 stamina, ratio never goes below floor."""
        assert effective_ratio(0.0) >= 0.45
        assert effective_ratio(0.01) >= 0.45

    def test_custom_parameters(self):
        r = effective_ratio(0.5, knee=0.5, floor=0.3, k=15)
        assert 0.3 < r < 1.0


class TestApplyExhaustion:
    def test_full_stamina_no_degradation(self):
        horse = make_horse(current_stamina=100.0)
        result = apply_exhaustion(horse)
        base = horse.base_attributes
        assert result.cruise_speed == pytest.approx(base.cruise_speed, abs=0.1)
        assert result.max_speed == pytest.approx(base.max_speed, abs=0.1)
        assert result.forward_accel == pytest.approx(base.forward_accel, abs=0.01)
        assert result.turn_accel == pytest.approx(base.turn_accel, abs=0.01)

    def test_zero_stamina_hits_floor(self):
        horse = make_horse(current_stamina=0.0)
        result = apply_exhaustion(horse)
        base = horse.base_attributes
        assert result.cruise_speed == pytest.approx(base.cruise_speed * 0.45, abs=0.1)
        assert result.max_speed == pytest.approx(base.max_speed * 0.45, abs=0.1)

    def test_half_stamina_mild_degradation(self):
        horse = make_horse(current_stamina=50.0)
        result = apply_exhaustion(horse)
        base = horse.base_attributes
        # At 50% stamina, ratio ~0.94 — still mostly full
        assert result.cruise_speed > base.cruise_speed * 0.9
        assert result.cruise_speed < base.cruise_speed

    def test_low_stamina_significant_degradation(self):
        horse = make_horse(current_stamina=10.0)
        result = apply_exhaustion(horse)
        base = horse.base_attributes
        # At 10% stamina, ratio ~0.50 — heavily degraded
        assert result.cruise_speed < base.cruise_speed * 0.55

    def test_no_iterative_decay(self):
        """New model is stateless — same stamina always gives same result."""
        horse = make_horse(current_stamina=30.0)
        r1 = apply_exhaustion(horse)
        horse.effective_attributes = r1
        r2 = apply_exhaustion(horse)
        assert r1.cruise_speed == pytest.approx(r2.cruise_speed)

    def test_cornering_grip_unchanged(self):
        """cornering_grip is not affected by exhaustion."""
        horse = make_horse(current_stamina=0.0)
        result = apply_exhaustion(horse)
        assert result.cornering_grip == horse.base_attributes.cornering_grip

    def test_weight_unchanged(self):
        horse = make_horse(current_stamina=0.0)
        result = apply_exhaustion(horse)
        assert result.weight == horse.base_attributes.weight
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_exhaustion.py -v`
Expected: FAIL — `effective_ratio` does not exist yet, old `apply_exhaustion` has wrong behavior

- [ ] **Step 3: Implement the sigmoid-knee exhaustion model**

Replace the contents of `horse_racing/core/exhaustion.py` with:

```python
"""Gradual exhaustion model — sigmoid-knee degradation curve."""

import dataclasses
import math

from .attributes import CoreAttributes
from .types import Horse

KNEE = 0.28
FLOOR = 0.45
K = 10


def effective_ratio(
    stamina_pct: float,
    knee: float = KNEE,
    floor: float = FLOOR,
    k: float = K,
) -> float:
    """Map stamina percentage (0-1) to stat multiplier (floor-1.0).

    Uses a sigmoid curve centered on `knee`:
    - Above knee: stats degrade gently
    - Below knee: stats drop steeply
    - At 0 stamina: stats bottom out at `floor`
    """
    raw = 1.0 / (1.0 + math.exp(-k * (stamina_pct - knee)))
    sig_at_1 = 1.0 / (1.0 + math.exp(-k * (1.0 - knee)))
    sig_at_0 = 1.0 / (1.0 + math.exp(-k * (0.0 - knee)))
    normalized = (raw - sig_at_0) / (sig_at_1 - sig_at_0)
    return floor + (1.0 - floor) * normalized


def apply_exhaustion(horse: Horse) -> CoreAttributes:
    """Compute effective attributes based on current stamina level."""
    base = horse.base_attributes
    stamina_pct = horse.current_stamina / base.max_stamina if base.max_stamina > 0 else 0.0
    stamina_pct = max(0.0, min(1.0, stamina_pct))
    ratio = effective_ratio(stamina_pct)
    return dataclasses.replace(
        base,
        cruise_speed=base.cruise_speed * ratio,
        max_speed=base.max_speed * ratio,
        forward_accel=base.forward_accel * ratio,
        turn_accel=base.turn_accel * ratio,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_exhaustion.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add horse_racing/core/exhaustion.py tests/test_exhaustion.py
git commit -m "feat(exhaustion): replace binary cliff with sigmoid-knee curve"
```

---

### Task 2: Add `last_drain` field to Horse

**Files:**
- Modify: `horse_racing/core/types.py`
- Modify: `horse_racing/core/stamina.py`
- Modify: `tests/test_stamina.py`

- [ ] **Step 1: Write failing test for `last_drain` tracking**

Add to the end of `tests/test_stamina.py`:

```python
class TestLastDrain:
    def test_last_drain_recorded(self):
        horse = make_horse(tangential_vel=18.0)
        attrs = horse.base_attributes
        drain_stamina(horse, attrs, InputState(1.0, 0), flat_straight())
        assert horse.last_drain > 0.0

    def test_last_drain_matches_stamina_change(self):
        horse = make_horse(tangential_vel=13.0)
        attrs = horse.base_attributes
        before = horse.current_stamina
        drain_stamina(horse, attrs, InputState(0.5, 0), flat_straight())
        actual_drain = before - horse.current_stamina
        assert horse.last_drain == pytest.approx(actual_drain)

    def test_last_drain_zero_default(self):
        horse = make_horse()
        assert horse.last_drain == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stamina.py::TestLastDrain -v`
Expected: FAIL — `Horse` has no `last_drain` attribute

- [ ] **Step 3: Add `last_drain` to Horse dataclass**

In `horse_racing/core/types.py`, add after the `effective_attributes` field:

```python
    last_drain: float = 0.0
```

- [ ] **Step 4: Record `last_drain` in `drain_stamina()`**

In `horse_racing/core/stamina.py`, change the last two lines of `drain_stamina()` from:

```python
    drain *= attrs.drain_rate_mult
    horse.current_stamina = max(0.0, horse.current_stamina - drain)
```

to:

```python
    drain *= attrs.drain_rate_mult
    horse.last_drain = drain
    horse.current_stamina = max(0.0, horse.current_stamina - drain)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_stamina.py -v`
Expected: All PASS (both old and new tests)

- [ ] **Step 6: Commit**

```bash
git add horse_racing/core/types.py horse_racing/core/stamina.py tests/test_stamina.py
git commit -m "feat(stamina): track last_drain on Horse for observation"
```

---

### Task 3: Track-length-normalized drain

**Files:**
- Modify: `horse_racing/core/stamina.py`
- Modify: `horse_racing/core/race.py`
- Modify: `tests/test_stamina.py`

- [ ] **Step 1: Write failing tests for drain scale**

Add to `tests/test_stamina.py`:

```python
from horse_racing.core.stamina import compute_drain_scale

class TestDrainScale:
    def test_reference_track_returns_one(self):
        """A track that takes exactly REFERENCE_CRUISE_TICKS at cruise speed."""
        from horse_racing.core.stamina import REFERENCE_CRUISE_TICKS
        from horse_racing.core.types import FIXED_DT, PHYS_SUBSTEPS
        cruise_speed = 13.0
        # length = cruise_speed * dt_per_tick * reference_ticks
        dt_per_tick = FIXED_DT * PHYS_SUBSTEPS
        length = cruise_speed * dt_per_tick * REFERENCE_CRUISE_TICKS
        scale = compute_drain_scale(length, cruise_speed)
        assert scale == pytest.approx(1.0, abs=0.01)

    def test_longer_track_reduces_drain(self):
        """A track twice as long should drain at half the rate."""
        from horse_racing.core.stamina import REFERENCE_CRUISE_TICKS
        from horse_racing.core.types import FIXED_DT, PHYS_SUBSTEPS
        cruise_speed = 13.0
        dt_per_tick = FIXED_DT * PHYS_SUBSTEPS
        ref_length = cruise_speed * dt_per_tick * REFERENCE_CRUISE_TICKS
        scale = compute_drain_scale(ref_length * 2, cruise_speed)
        assert scale == pytest.approx(0.5, abs=0.01)

    def test_shorter_track_increases_drain(self):
        """A track half as long should drain at double the rate."""
        from horse_racing.core.stamina import REFERENCE_CRUISE_TICKS
        from horse_racing.core.types import FIXED_DT, PHYS_SUBSTEPS
        cruise_speed = 13.0
        dt_per_tick = FIXED_DT * PHYS_SUBSTEPS
        ref_length = cruise_speed * dt_per_tick * REFERENCE_CRUISE_TICKS
        scale = compute_drain_scale(ref_length / 2, cruise_speed)
        assert scale == pytest.approx(2.0, abs=0.01)

    def test_drain_scale_applied(self):
        """drain_scale=0.5 should halve the drain."""
        h1 = make_horse(tangential_vel=18.0)
        h2 = make_horse(tangential_vel=18.0)
        attrs = h1.base_attributes
        frame = flat_straight()
        drain_stamina(h1, attrs, InputState(1.0, 0), frame, drain_scale=1.0)
        drain_stamina(h2, attrs, InputState(1.0, 0), frame, drain_scale=0.5)
        drain_1x = 100.0 - h1.current_stamina
        drain_half = 100.0 - h2.current_stamina
        assert drain_half == pytest.approx(drain_1x * 0.5, abs=0.001)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stamina.py::TestDrainScale -v`
Expected: FAIL — `compute_drain_scale` does not exist, `drain_stamina` doesn't accept `drain_scale`

- [ ] **Step 3: Implement `compute_drain_scale` and update `drain_stamina`**

In `horse_racing/core/stamina.py`, add at the top (after existing constants):

```python
from .types import FIXED_DT, PHYS_SUBSTEPS

REFERENCE_CRUISE_TICKS = 2000


def compute_drain_scale(track_total_length: float, cruise_speed: float) -> float:
    """Scale drain so cruise-effort stamina usage is consistent across tracks."""
    dt_per_tick = FIXED_DT * PHYS_SUBSTEPS
    estimated_ticks = track_total_length / (cruise_speed * dt_per_tick)
    if estimated_ticks < 1e-6:
        return 1.0
    return REFERENCE_CRUISE_TICKS / estimated_ticks
```

Update the `drain_stamina` signature and final lines:

```python
def drain_stamina(
    horse: Horse,
    attrs: CoreAttributes,
    inp: InputState,
    frame: TrackFrame,
    drain_scale: float = 1.0,
) -> None:
```

And change the end of the function from:

```python
    drain *= attrs.drain_rate_mult
    horse.last_drain = drain
    horse.current_stamina = max(0.0, horse.current_stamina - drain)
```

to:

```python
    drain *= attrs.drain_rate_mult
    drain *= drain_scale
    horse.last_drain = drain
    horse.current_stamina = max(0.0, horse.current_stamina - drain)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_stamina.py -v`
Expected: All PASS

- [ ] **Step 5: Wire drain_scale into Race**

In `horse_racing/core/race.py`, add to `Race.__init__` after `self._add_horse_bodies()`:

```python
        # Compute drain normalization from track length
        from .stamina import compute_drain_scale
        from .attributes import create_default_attributes
        navigator = self.state.horses[0].navigator
        default_attrs = create_default_attributes()
        self._drain_scale = compute_drain_scale(
            navigator.total_length, default_attrs.cruise_speed
        )
```

Update the `drain_stamina` call in `Race.tick()` from:

```python
                drain_stamina(h, h.effective_attributes, horse_input, frame)
```

to:

```python
                drain_stamina(h, h.effective_attributes, horse_input, frame, self._drain_scale)
```

Also update `Race.reset()` — add the same drain_scale computation after `self._add_horse_bodies()`:

```python
        from .stamina import compute_drain_scale
        from .attributes import create_default_attributes
        navigator = self.state.horses[0].navigator
        default_attrs = create_default_attributes()
        self._drain_scale = compute_drain_scale(
            navigator.total_length, default_attrs.cruise_speed
        )
```

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -v -m "not slow"`
Expected: All PASS (existing drain tests still pass because `drain_scale` defaults to 1.0)

- [ ] **Step 7: Commit**

```bash
git add horse_racing/core/stamina.py horse_racing/core/race.py tests/test_stamina.py
git commit -m "feat(stamina): normalize drain rate by track length"
```

---

### Task 4: Reward shaping

**Files:**
- Rewrite: `horse_racing/reward.py`
- Rewrite: `tests/test_reward.py`
- Modify: `horse_racing/env/single_env.py`

- [ ] **Step 1: Write failing tests for new reward**

Replace the contents of `tests/test_reward.py` with:

```python
import pytest

from horse_racing.reward import STAMINA_EFFICIENCY_BONUS, compute_reward


def test_positive_progress():
    reward = compute_reward(
        prev_progress=0.1, curr_progress=0.15, finish_order=None
    )
    assert reward == pytest.approx(0.05)


def test_no_progress():
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.5, finish_order=None
    )
    assert reward == pytest.approx(0.0)


def test_first_place_bonus():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        current_stamina=0.0, max_stamina=100.0,
    )
    # progress + placement + efficiency (used all stamina)
    assert reward == pytest.approx(0.01 + 10.0 + STAMINA_EFFICIENCY_BONUS)


def test_second_place_bonus():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=2,
        current_stamina=0.0, max_stamina=100.0,
    )
    assert reward == pytest.approx(0.01 + 5.0 + STAMINA_EFFICIENCY_BONUS)


def test_third_place_bonus():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=3,
        current_stamina=0.0, max_stamina=100.0,
    )
    assert reward == pytest.approx(0.01 + 2.0 + STAMINA_EFFICIENCY_BONUS)


def test_no_bonus_for_fourth():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=4,
        current_stamina=0.0, max_stamina=100.0,
    )
    assert reward == pytest.approx(0.01 + STAMINA_EFFICIENCY_BONUS)


def test_efficiency_bonus_scales_with_stamina_used():
    # Finish with 50% stamina = half the efficiency bonus
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        current_stamina=50.0, max_stamina=100.0,
    )
    expected = 0.01 + 10.0 + STAMINA_EFFICIENCY_BONUS * 0.5
    assert reward == pytest.approx(expected)


def test_efficiency_bonus_zero_when_full_stamina():
    reward = compute_reward(
        prev_progress=0.99, curr_progress=1.0, finish_order=1,
        current_stamina=100.0, max_stamina=100.0,
    )
    assert reward == pytest.approx(0.01 + 10.0)


def test_no_efficiency_bonus_mid_race():
    """Efficiency bonus only triggers at finish (finish_order is not None)."""
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.51, finish_order=None,
        current_stamina=10.0, max_stamina=100.0,
    )
    assert reward == pytest.approx(0.01)


def test_no_exhaustion_penalty():
    """Old exhaustion penalty is removed — zero stamina mid-race has no direct penalty."""
    reward = compute_reward(
        prev_progress=0.5, curr_progress=0.51, finish_order=None,
        current_stamina=0.0, max_stamina=100.0,
    )
    assert reward == pytest.approx(0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_reward.py -v`
Expected: FAIL — old `compute_reward` doesn't have `max_stamina` param, `STAMINA_EFFICIENCY_BONUS` doesn't exist

- [ ] **Step 3: Implement new reward function**

Replace the contents of `horse_racing/reward.py` with:

```python
"""Reward function — delta-progress + finish bonus + stamina efficiency."""

from __future__ import annotations

_FINISH_BONUS = {1: 10.0, 2: 5.0, 3: 2.0}
STAMINA_EFFICIENCY_BONUS = 1.0


def compute_reward(
    prev_progress: float,
    curr_progress: float,
    finish_order: int | None,
    current_stamina: float = 1.0,
    max_stamina: float = 100.0,
) -> float:
    """Compute reward for one step.

    Args:
        prev_progress: Track progress at previous tick [0, 1].
        curr_progress: Track progress at current tick [0, 1].
        finish_order: Finishing position (1-based) if horse finished, else None.
        current_stamina: Horse's current stamina.
        max_stamina: Horse's maximum stamina.
    """
    reward = curr_progress - prev_progress
    if finish_order is not None:
        reward += _FINISH_BONUS.get(finish_order, 0.0)
        stamina_pct = current_stamina / max_stamina if max_stamina > 0 else 0.0
        reward += STAMINA_EFFICIENCY_BONUS * (1.0 - stamina_pct)
    return reward
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reward.py -v`
Expected: All PASS

- [ ] **Step 5: Update env to pass `max_stamina` to reward**

In `horse_racing/env/single_env.py`, change the `compute_reward` call (around line 96) from:

```python
        reward = compute_reward(
            self._prev_progress, curr_progress, agent_horse.finish_order,
            agent_horse.current_stamina,
        )
```

to:

```python
        reward = compute_reward(
            self._prev_progress, curr_progress, agent_horse.finish_order,
            agent_horse.current_stamina,
            agent_horse.base_attributes.max_stamina,
        )
```

- [ ] **Step 6: Run env tests**

Run: `python -m pytest tests/test_env.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add horse_racing/reward.py tests/test_reward.py horse_racing/env/single_env.py
git commit -m "feat(reward): replace exhaustion penalty with stamina efficiency bonus"
```

---

### Task 5: Observation — add stamina drain rate

**Files:**
- Modify: `horse_racing/core/observation.py`
- Modify: `tests/test_observation.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Write failing tests for new observation field**

In `tests/test_observation.py`, change `test_obs_size_is_139`:

```python
def test_obs_size_is_140():
    assert OBS_SIZE == 140
```

Add a new test at the end:

```python
def test_drain_rate_observation():
    segments = load_track_json(TRACKS_DIR / "test_oval.json")
    race = Race(segments, horse_count=2)
    race.start(None)
    # Tick once so drain is recorded
    from horse_racing.core.types import InputState
    race.tick({0: InputState(1.0, 0), 1: InputState(0, 0)})
    obs = build_observations(race)
    # Horse 0 pushed hard — should have nonzero drain
    assert obs[0][139] > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_observation.py::test_obs_size_is_140 tests/test_observation.py::test_drain_rate_observation -v`
Expected: FAIL — OBS_SIZE is still 139, index 139 doesn't exist or is in opponent slots

- [ ] **Step 3: Update observation builder**

In `horse_racing/core/observation.py`, change:

```python
SELF_STATE_SIZE = 14
```

to:

```python
SELF_STATE_SIZE = 15
```

In the `build_observations` function, after line `obs[13] = normalize_trait(base.weight, "weight")`, add:

```python
        obs[14] = self_horse.last_drain / base.max_stamina
```

And update the track context block — shift all indices up by 1. Change:

```python
        obs[14] = curvature(frame.turn_radius)
        obs[15] = frame.slope

        for i, dist in enumerate(LOOKAHEAD_DISTANCES):
            lookahead = self_horse.navigator.sample_track_ahead(self_horse.pos, dist)
            obs[16 + i * 2] = curvature(lookahead.turn_radius)
            obs[16 + i * 2 + 1] = lookahead.slope
```

to:

```python
        obs[15] = curvature(frame.turn_radius)
        obs[16] = frame.slope

        for i, dist in enumerate(LOOKAHEAD_DISTANCES):
            lookahead = self_horse.navigator.sample_track_ahead(self_horse.pos, dist)
            obs[17 + i * 2] = curvature(lookahead.turn_radius)
            obs[17 + i * 2 + 1] = lookahead.slope
```

And update the opponent base offset:

```python
        opponent_base = SELF_STATE_SIZE + TRACK_CONTEXT_SIZE
```

This line already uses the constants, so it auto-adjusts. Verify that `OBS_SIZE` recomputes correctly:
- `SELF_STATE_SIZE = 15`
- `TRACK_CONTEXT_SIZE = 2 + 4 * 2 = 10` (unchanged)
- `OPPONENT_SLOTS * OPPONENT_SLOT_SIZE = 23 * 5 = 115` (unchanged)
- `OBS_SIZE = 15 + 10 + 115 = 140` ✓

- [ ] **Step 4: Run observation tests**

Run: `python -m pytest tests/test_observation.py -v`
Expected: PASS — but `test_obs_size_is_139` should have been renamed in step 1. Also `test_self_state_size` will fail.

Update `tests/test_observation.py` — change:

```python
def test_self_state_size():
    assert SELF_STATE_SIZE == 14
```

to:

```python
def test_self_state_size():
    assert SELF_STATE_SIZE == 15
```

Run: `python -m pytest tests/test_observation.py -v`
Expected: All PASS

- [ ] **Step 5: Update env tests for new OBS_SIZE**

In `tests/test_env.py`, the tests reference `OBS_SIZE` dynamically via import, so they should auto-adjust. Verify:

Run: `python -m pytest tests/test_env.py -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -v -m "not slow"`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add horse_racing/core/observation.py tests/test_observation.py tests/test_env.py
git commit -m "feat(obs): add stamina drain rate observation, OBS_SIZE 139 → 140"
```

---

### Task 6: Full integration verification

**Files:**
- None created — verification only

- [ ] **Step 1: Run full test suite including slow tests**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Smoke test the environment loop**

Run:

```bash
python -c "
from horse_racing.env import HorseRacingSingleEnv
env = HorseRacingSingleEnv(track_path='tracks/test_oval.json', horse_count=4)
obs, info = env.reset()
print(f'Obs shape: {obs.shape}')
print(f'Initial stamina: {info[\"stamina\"]:.1f}')
total_r = 0
for i in range(200):
    obs, r, term, trunc, info = env.step(76)  # push hard
    total_r += r
    if term or trunc:
        break
print(f'After 200 steps pushing hard:')
print(f'  Stamina: {info[\"stamina\"]:.1f}')
print(f'  Progress: {info[\"progress\"]:.3f}')
print(f'  Reward: {total_r:.3f}')
print(f'  Drain obs (obs[14]): {obs[14]:.6f}')
env.close()
print('Integration OK')
"
```

Expected: Obs shape `(140,)`, stamina draining gradually, drain obs nonzero, no errors.

- [ ] **Step 3: Verify sigmoid curve produces sensible behavior**

Run:

```bash
python -c "
from horse_racing.core.exhaustion import effective_ratio
print('Stamina%  Ratio')
for pct in [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 0]:
    r = effective_ratio(pct / 100)
    print(f'  {pct:3d}%     {r:.3f}')
"
```

Expected: Smooth curve from 1.0 down to 0.45, steep drop around 20-30%.

- [ ] **Step 4: Commit (if any fixups were needed)**

```bash
git add -A
git commit -m "fix: integration fixups for stamina intelligence"
```

Only commit if changes were needed. If all passed clean, skip this step.
