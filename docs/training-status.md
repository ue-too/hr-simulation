# Training Status — v20 Single Agent (2026-04-07)

## Model
- **File**: `stage_4_jockey.onnx` (111-dim obs, 2-dim action)
- **Training**: SB3 PPO, single agent vs BT scripted opponents
- **Curriculum**: Straight → Oval → All tracks → All tracks 8 horses

## Evaluation Results (4 horses, 5 episodes per track)

| Track | Complete | AvgSpeed | MaxSpeed | Reward | Stamina | Notes |
|---|---|---|---|---|---|---|
| Straight (400m) | 100% | 13.9 | 14.9 | 714 | 82% | Too conservative |
| Test Oval (905m) | 100% | 14.5 | 22.0 | 1786 | 36% | Good |
| Tokyo (901m) | 100% | 14.4 | 16.3 | 1218 | 42% | Good |
| Tight Oval (1005m) | 100% | 15.2 | 21.9 | 1538 | 30% | Good |
| Hanshin (1008m) | 100% | 14.5 | 18.4 | 2109 | 36% | Best reward |
| Kyoto (917m) | 100% | 15.0 | 18.0 | 1241 | 40% | Good |
| Kokura (1415m) | 100% | 14.6 | 17.4 | 256 | 16% | Draining too much |
| Tokyo 2600 (1528m) | 100% | 14.6 | 19.8 | -559 | 10% | Draining too much |
| Gentle Oval (1810m) | 70% | 12.2 | 17.3 | -1237 | 13% | Failing |

**Overall**: 97% completion, avg reward +785, avg speed 14.3 m/s

## What's Working
- **Kicking**: MaxSpeed 21-22 on ovals shows the agent learned to sprint late
- **Short/medium tracks**: Solid pacing with 30-42% stamina and positive rewards
- **Cornering**: Pre-positioning on straights before curves (signed curvature obs working)
- **BT opponent handling**: Trained against 7 personality types including degenerate ones

## What Needs Improvement
- **Long tracks (1400m+)**: Kicks too aggressively, drains to 10-16% stamina, negative rewards
- **Gentle Oval (1810m)**: 70% completion — some horses fail to finish
- **Short tracks**: Too conservative — 82% stamina on 400m means barely pushing
- **Track-length conditioning**: Agent has track_length in obs but doesn't modulate kick intensity well

## Suggested Next Steps (priority order)

### 1. Gate kick bonus on stamina
The kick reward fires at 75%+ progress regardless of stamina. On long tracks the agent kicks into exhaustion. Fix: only reward kicking when stamina > 30%.
```python
# Current:
elif progress > 0.75 and vel > cruise_spd and stamina > 0.25:
    kick_intensity = (progress - 0.75) / 0.25
    reward += 2.0 * kick_intensity

# Proposed:
elif progress > 0.75 and vel > cruise_spd and stamina > 0.30:
    kick_intensity = (progress - 0.75) / 0.25
    reward += 2.0 * kick_intensity * min(stamina / 0.5, 1.0)  # scale by stamina
```

### 2. Scale excess-stamina penalty by track length
82% stamina on a 400m track is wasteful; 82% on 1800m might be necessary. Scale the terminal penalty so short tracks penalize hoarding more.
```python
# Current:
if stamina > 0.40:
    reward -= 20.0 * (stamina - 0.40)

# Proposed:
track_len_norm = obs_curr.get("track_length", 900.0) / 2000.0
excess_threshold = 0.25 + 0.25 * track_len_norm  # 0.36 for 900m, 0.48 for 1800m
if stamina > excess_threshold:
    reward -= 20.0 * (stamina - excess_threshold)
```

### 3. More training on long tracks
Stage 3/4 trains on all tracks equally. Long tracks need disproportionate exposure since they're harder. Options:
- Weight long tracks 2-3x in the track selection
- Add a dedicated "long tracks only" stage between stages 3 and 4
- Increase total timesteps for stage 3

### 4. Multi-agent training (future)
After single-agent pacing is solid, move to RLlib multi-agent with:
- BT opponent ratio 0.5-0.75
- League self-play with checkpoint pool
- Scripted bot anchors to prevent strategy collapse

## Reward Function Summary (current)

| Component | Magnitude | Notes |
|---|---|---|
| Forward progress | 30.0 * delta * (1+2p) * eff | ~30-90 cumulative |
| Speed bonus | 0.03 * (vel/max) | ~90-130 cumulative |
| Exhaustion cliff | -3.0/tick below 30% stamina | Hard wall |
| Pacing (cruise early) | +0.8/tick | progress < 70% |
| Pacing (kick late) | +2.0 * ramp | progress > 75%, ramps 0→2 |
| Near-finish speed | +1.5 * above_cruise_ratio | progress > 85% |
| Excess stamina penalty | -20 * (stam - 0.40) | Terminal, at finish |
| Archetype bonuses | 5x multiplier | ~100-500 cumulative |
| Finish order | [50, 30, 15, 5] | Terminal |
| Cornering (outside) | up to -90/tick | Very harsh |
| Pre-positioning | +0.3 * inside_score | On straights before curves |
