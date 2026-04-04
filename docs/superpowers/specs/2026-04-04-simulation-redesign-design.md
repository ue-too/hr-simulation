# Horse Racing Simulation Redesign

## Context

The current simulation has grown organically across 17 versions into a tangled system of overlapping mechanisms (10 genome traits, 8 conditional modifiers, 6 skills with both reward and physics effects). Despite this complexity, RL agents converge to a single universal strategy regardless of horse attributes. Movement feels robotic, and horses don't feel meaningfully different.

This redesign rebuilds the simulation engine and RL pipeline from scratch with three goals:
1. **Realistic race simulation** with organic movement and diverse strategies
2. **Competitive AI** that adapts its racing style based on horse capabilities
3. **Game/product foundation** that is clean, extensible, and maintainable

Key design philosophy: **Horse = body, Jockey = brain.** The horse is purely physical capabilities. The jockey (RL model or human player) observes the horse's attributes and makes tactical decisions. Strategic diversity emerges from jockeys intelligently adapting to different horses, not from the physics forcing specific behaviors.

## Architecture

```
horse_racing/
  core/
    horse.py              # Horse physical profile (8 traits)
    track.py              # Track loading (existing JSON format)
    track_navigator.py    # Per-horse track position tracking
    physics.py            # Force integration, collision, movement smoothing
    stamina.py            # Stamina drain model (speed-efficiency curve)
  env/
    racing_env.py         # Gymnasium environment (single-agent + self-play)
    observation.py        # Observation vector building (~63 features)
    reward.py             # Minimal outcome-focused reward (5 components)
  training/
    curriculum.py         # 5-stage training definitions
    self_play.py          # Self-play with frozen ONNX opponents
    export_onnx.py        # ONNX export (handles GRU recurrent state)
```

**Removed from current system:** `genome.py`, `modifiers.py`, `skills.py`, `attributes.py` (complex resolution), `multi_agent_env.py`. No layered modifier/skill/skill-physics system.

## Horse Model (8 Traits)

The horse is a data class of physical capabilities. No strategy, no personality, no skills.

| Trait | Range | Purpose |
|---|---|---|
| `top_speed` | 16-20 m/s | Absolute speed ceiling |
| `acceleration` | 0.5-1.5 | How quickly it reaches target speed |
| `stamina_pool` | 60-150 | Total energy budget (drain only, no recovery) |
| `stamina_efficiency` | 0.7-1.3 | How cheaply it sustains speed |
| `cornering_grip_left` | 0.5-1.5 | Lateral grip on left-hand turns |
| `cornering_grip_right` | 0.5-1.5 | Lateral grip on right-hand turns |
| `climbing_power` | 0.5-1.5 | Uphill/downhill performance |
| `weight` | 430-550 kg | Mass for collisions and inertia |

### Speed-Stamina Relationship

Stamina drain is a continuous function of speed relative to an efficiency threshold:

```
efficiency_speed = top_speed * 0.75
drain_per_tick = base_drain + excess_drain * max(0, current_speed - efficiency_speed)^2
```

- High `top_speed` + low `stamina_efficiency` = narrow efficient zone, pays dearly for speed
- Moderate `top_speed` + high `stamina_efficiency` = sustains speed longer
- This naturally creates different optimal pacing strategies without any modifier system

### Horse Generation

Simple random sampling within trait ranges. No genome/breeding/alleles in the core sim (can be added as a game feature later).

## Jockey Model

The jockey is the RL policy (or human player). It observes the horse, the race, and its own style parameters.

### Jockey Style Parameters (in observation)

| Param | Range | Meaning |
|---|---|---|
| `risk_tolerance` | 0-1 | 0 = conservative pacing, 1 = aggressive gambler |
| `tactical_bias` | -1 to 1 | -1 = front-runner, 0 = stalker, 1 = closer |
| `skill_level` | 0-1 | How well the jockey reads and adapts |

One trained policy conditions on these parameters. At race time, assign any jockey profile to any horse — one ONNX model, infinite combinations.

## Action Space

2D continuous: **Effort + Lane**

| Action | Range | Meaning |
|---|---|---|
| `effort` | -1 to 1 | -1 = ease up, 0 = cruise at efficiency speed, +1 = maximum push |
| `lane` | -1 to 1 | -1 = move toward inside rail, 0 = hold line, +1 = move outside |

The horse's attributes translate jockey intent to physics:
- `effort=0.8` on a fast horse = strong force toward 19 m/s
- `effort=0.8` on a slow horse = weaker force toward 17 m/s
- Same decision, different physical outcome

**Keyboard mapping (web sim):** W/S = effort up/down, A/D = lane inside/outside.

## Observation Space (~63 features)

```
Ego state (7):
  speed, lateral_vel, displacement, progress, curvature, curvature_direction, stamina_ratio

Horse profile (6):
  top_speed, acceleration, stamina_efficiency,
  cornering_grip_left, cornering_grip_right, climbing_power

Jockey style (3):
  risk_tolerance, tactical_bias, skill_level

Race context (3):
  placement_norm, num_horses_norm, race_progress_elapsed

Track lookahead - next 3 segments (12):
  [curvature, turn_direction, length, slope] x 3

Relative horses - top 8 by proximity (32):
  [tang_offset, norm_offset, rel_speed, stamina_estimate] x 8
```

Key design decisions:
- 8 relative horses (not 19) — only nearby horses matter tactically
- Track lookahead lets the recurrent jockey plan ahead for curves and slopes
- Horse profile included so jockey knows its mount
- Opponent stamina is estimated (not exact) for realistic information asymmetry
- Curvature direction needed for left/right grip differentiation

## Reward Function (5 components)

| Component | Magnitude | Purpose |
|---|---|---|
| `placement_reward` | 0-0.5/tick | `0.5 * (N - placement) / (N - 1)` |
| `finish_bonus` | [50, 30, 15, 5] | One-time bonus for 1st-4th |
| `progress_nudge` | ~30 * delta_progress | Small forward incentive (prevents stalling) |
| `alive_penalty` | -0.1/tick | Time pressure |
| `collision_penalty` | -1.0 | Discourage bumping |

**Jockey style influence (light nudges):**
- Front-runner bias: +0.1 bonus for leading at 25% and 50% progress marks
- Closer bias: +0.1 bonus for positions gained in final 25%
- High risk: no stamina penalty
- Low risk: -0.05 if stamina < 20% before 80% progress

**Removed:** speed bonus, cornering line bonus, stamina budget shaping, all skill bonuses.

## Physics & Movement

### Core Physics
- 240 Hz tick rate, 8 substeps (1920 Hz effective)
- Semi-implicit Euler integration (unchanged)
- Track format: existing JSON with straights + curves (unchanged)

### Action-to-Force Translation

```python
# Effort -> forward force
target_speed = efficiency_speed + effort * (top_speed - efficiency_speed)  # effort > 0
forward_force = acceleration * (target_speed - current_speed) * mass

# Lane -> lateral force
lateral_force = lane * active_cornering_grip * sign_correction
# active_cornering_grip = cornering_grip_left or cornering_grip_right based on turn direction

# Stamina drain
drain = base_rate + excess_rate * max(0, current_speed - efficiency_speed)^2
```

### Movement Smoothing (organic feel)

1. **Response lag:** Exponential moving average on force output, tau=0.3s. High-acceleration horses respond faster. Eliminates robotic instant acceleration.

2. **Stride oscillation:** Sinusoidal forward speed variation (+/-0.2 m/s at 2.5 Hz) simulating gallop rhythm. Cosmetic only, not controllable or observable by agent.

3. **Fatigue effects (below 30% stamina):**
   - Forward force scales down linearly
   - Lateral response lag doubles
   - Low-frequency lateral drift (Perlin noise, +/-0.1 m/s) — tired horses waver

4. **Cornering:** Centripetal force auto-applied on curves. Excess speed beyond grip causes outward drift. Left/right grip asymmetry creates turn-dependent handling.

5. **Slope:** Uphill reduces forward force by `slope * gravity * (1 / climbing_power)`. Downhill adds force.

6. **Drafting:** Physics effect when within 15m behind another horse — reduced air resistance, slight speed boost. No observation flag; jockey feels it through speed changes.

7. **Collision:** OBB (oriented bounding box). Impulse scales with weight. Heavy horses push light horses.

## Training Pipeline

### Algorithm
- **SAC (Soft Actor-Critic)** with GRU recurrent policy
- **Library:** Ray RLlib (supports SAC + LSTM/GRU natively)
- **Runtime:** Google Colab Pro+ (`ray.init(num_cpus=2, num_gpus=1)`)

### Why SAC + GRU
- **SAC:** Off-policy (sample efficient), entropy maximization (maintains behavioral diversity), better at conditional policies than PPO
- **GRU:** Gives jockey memory of race phases. Can plan: "I've been conserving for 600m, now push." MLP cannot do this.

### Policy Network
```
Observation (63) -> Linear(128) -> GRU(128) -> Linear(64) -> Action (2)
                                             -> Linear(64) -> Q-value
```

### Training Curriculum (5 stages)

| Stage | Setup | Focus | Gate |
|---|---|---|---|
| 1 | Solo, straight track | Basic effort control, finish race | Finishes 90% of episodes |
| 2 | Solo, curved tracks (mix) | Cornering, lane positioning, L/R grip | Finishes 80%, reasonable times |
| 3 | 2-4 horses, real tracks | Collision avoidance, drafting, awareness | Placement <= 2nd in 60% |
| 4 | 4-10 horses, self-play | Competitive tactics, position battles | Wins 30%+ vs frozen opponents |
| 5 | 6-20 horses, diverse self-play | Full field, jockey style conditioning | Style params visibly affect behavior |

### Jockey Style Schedule
- Stages 1-3: Fixed neutral style (risk=0.5, tactic=0, skill=1.0)
- Stage 4: Randomize style params per episode
- Stage 5: Full randomization, opponents get random styles too

### Self-Play
- Freeze checkpoints every N episodes as opponent pool
- Each episode: random track, random field size, random opponent models, random jockey styles
- Opponents get randomized horse attributes

## ONNX Export & Web Integration

Export GRU policy to ONNX. The web sim (`ai-jockey.ts`) must maintain GRU hidden state across ticks — a change from current stateless inference.

```typescript
// Web sim change: carry hidden state
class AIJockey {
  private gruState: Float32Array;  // Persistent across ticks
  
  infer(observation: Float32Array): [effort: number, lane: number] {
    const [action, newState] = this.session.run(observation, this.gruState);
    this.gruState = newState;
    return action;
  }
  
  reset() { this.gruState = new Float32Array(128).fill(0); }
}
```

The web sim action-to-physics translation must match the Python implementation exactly.

## What's Kept From Current System
- Track JSON format and all track files
- Pixi.js web visualization
- OBB collision detection approach
- 240 Hz physics tick with substeps
- ONNX model deployment pattern
- Per-horse model selection in toolbar

## Verification Plan

1. **Unit tests:** Horse trait generation, physics force calculations, stamina drain curves, observation building
2. **Integration test:** Run a complete solo race, verify horse finishes, stamina drains correctly, observation dimensions match schema
3. **Training smoke test:** Stage 1 (straight track) reaches 90% finish gate within 200K steps
4. **Diversity test:** After stage 5, compare race trajectories of same horse with different jockey styles — front-runner bias should lead early, closer bias should gain late
5. **Web integration:** Export ONNX, load in web sim, verify GRU state management works, verify action translation matches Python
6. **Human play test:** Keyboard control (W/S/A/D) in web sim feels intuitive and responsive
