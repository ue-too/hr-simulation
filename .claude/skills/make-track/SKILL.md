---
name: make-track
description: Generate a geometrically correct horse racing track JSON file for RL training
argument-hint: "<track-name> [tight|gentle|short|long]"
---

# /make-track — Generate Horse Racing Track JSON

Generate a geometrically correct horse racing track JSON file for RL training.

## Usage

```
/make-track <name> [options]
```

**Arguments:**
- `name` (required) — Track filename (without `.json`), e.g. `hakodate`
- User may also specify in natural language: straight lengths, turn tightness, track size, etc.

## Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `straight_length` | 250 | Homestretch / backstretch length in meters |
| `turn_radius_min` | 40 | Minimum (apex) curve radius in meters |
| `turn_radius_max` | 150 | Maximum (entry/exit) curve radius in meters |
| `segments_per_turn` | 12 | Number of CURVE sub-segments per 180-degree turn |
| `direction` | `"cw"` | Winding direction: `"cw"` (negative angleSpan) or `"ccw"` (positive angleSpan) |

Adjust defaults based on user description. For example:
- "tight track" → `turn_radius_min=30, turn_radius_max=80`
- "gentle track" → `turn_radius_min=80, turn_radius_max=300`
- "short track" → `straight_length=150`
- "long track" → `straight_length=400`

## Generation Algorithm

### Step 1: Compute Turn Geometry

The track is an oval with two straights and two 180-degree turns.

**Radius profile for each turn** — use a parabolic profile that mimics clothoid spiral transitions:

```
For segment index i in [0, N-1] where N = segments_per_turn:
    t = (i + 0.5) / N                                    # midpoint parameter
    R[i] = R_min + (R_max - R_min) * (2 * (t - 0.5))^2  # parabolic: large at edges, small at apex
```

This gives large radii at turn entry/exit (gentle) and minimum radius at the apex (tightest).

**Angle distribution** — allocate angles inversely proportional to radius so tighter sections subtend less angle (keeps arc lengths more uniform):

```
weight[i] = 1.0 / R[i]
total_angle = pi  (180 degrees)
angleSpan[i] = total_angle * weight[i] / sum(all weights)
```

If `direction = "cw"`, negate all angleSpan values.

### Step 2: Place the First Straight (Homestretch)

The homestretch runs left-to-right along the bottom:

```json
{
    "tracktype": "STRAIGHT",
    "startPoint": {"x": 0, "y": 0, "z": 0},
    "endPoint": {"x": straight_length, "y": 0, "z": 0}
}
```

Initialize tracking state:
```
current_point = (straight_length, 0)
heading_angle = 0  (pointing right, in radians)
```

### Step 3: Generate Turn 1 (Right Side)

For each curve segment `i` in `[0, N-1]`:

1. **Compute the center** perpendicular to current heading at distance `R[i]`:
   ```
   heading_unit = (cos(heading_angle), sin(heading_angle))
   ```
   - For CW (negative angleSpan): center is to the RIGHT of heading
     ```
     perp = (sin(heading_angle), -cos(heading_angle))
     center = current_point + R[i] * perp
     ```
   - For CCW (positive angleSpan): center is to the LEFT of heading
     ```
     perp = (-sin(heading_angle), cos(heading_angle))
     center = current_point + R[i] * perp
     ```

2. **Compute start_angle** (angle from center to current_point):
   ```
   start_angle = atan2(current_point.y - center.y, current_point.x - center.x)
   ```

3. **Compute end_angle and endPoint**:
   ```
   end_angle = start_angle + angleSpan[i]
   endPoint = (center.x + R[i] * cos(end_angle), center.y + R[i] * sin(end_angle))
   ```

4. **Emit the segment**:
   ```json
   {
       "tracktype": "CURVE",
       "startPoint": {"x": current_point.x, "y": current_point.y},
       "endPoint": {"x": endPoint.x, "y": endPoint.y},
       "radius": R[i],
       "center": {"x": center.x, "y": center.y},
       "angleSpan": angleSpan[i]
   }
   ```

5. **Update state**:
   ```
   current_point = endPoint
   heading_angle = heading_angle + angleSpan[i]
   ```

### Step 4: Place the Backstretch (Top Straight)

After Turn 1, the heading should be approximately pointing LEFT (heading_angle ≈ pi for CW, or -pi for CCW).

The backstretch endPoint is directly above (or below) the homestretch startPoint:
```
backstretch_end = (0, current_point.y)
```

But first verify the heading is roughly horizontal. If there's a small angular error from Turn 1 (heading not exactly pi), correct it:
- Compute the direction from `current_point` to `backstretch_end`
- If the angular difference from current heading is small (< 0.1 rad), proceed
- The straight's endPoint.y should match current_point.y (keep the same y to maintain heading)

```json
{
    "tracktype": "STRAIGHT",
    "startPoint": {"x": current_point.x, "y": current_point.y, "z": 0},
    "endPoint": {"x": 0, "y": current_point.y, "z": 0}
}
```

Update state:
```
current_point = (0, current_point.y)
heading_angle = pi  (pointing left — adjust sign based on direction)
```

For CW: heading_angle = pi. For CCW: heading_angle = -pi (equivalently pi).

### Step 5: Generate Turn 2 (Left Side)

Same algorithm as Turn 1, using the same radius profile and angle distribution. This turn brings the path from heading=pi back around to heading=0 (or 2*pi).

Use the same `R[i]` and `angleSpan[i]` arrays (or generate new ones for asymmetric tracks).

### Step 6: Close the Loop

After Turn 2, `current_point` should be close to `(0, 0)`. Two cases:

- **If close enough** (distance < 1m): Snap the last curve segment's `endPoint` to exactly `(0, 0, 0)` — overwrite it.
- **If there's a gap**: Add a short final STRAIGHT segment from `current_point` to `(0, 0)` to close the loop.

Verify the first segment's `startPoint` matches the last segment's adjusted `endPoint`.

### Step 7: Adjust for Non-Standard Shapes (Optional)

If the user requests specific features:
- **Asymmetric turns**: Use different `R_min`/`R_max` for Turn 1 vs Turn 2
- **Sloped sections**: Add `"slope"` field to segments (positive = uphill)
- **Different straight lengths**: Use different lengths for homestretch vs backstretch (adjust Turn 2 endpoint accordingly)

## JSON Output Format

Write to `tracks/<name>.json` as a bare JSON array:

```json
[
  {
    "tracktype": "STRAIGHT",
    "startPoint": {"x": 0, "y": 0, "z": 0},
    "endPoint": {"x": 250, "y": 0, "z": 0}
  },
  {
    "tracktype": "CURVE",
    "startPoint": {"x": 250, "y": 0},
    "endPoint": {"x": ..., "y": ...},
    "radius": 150,
    "center": {"x": ..., "y": ...},
    "angleSpan": -0.234
  },
  ...
]
```

**Field rules:**
- `tracktype`: `"STRAIGHT"` or `"CURVE"` (uppercase)
- `startPoint`, `endPoint`: objects with `"x"`, `"y"` keys (float). STRAIGHT segments include `"z": 0`, CURVE segments omit `z`.
- `center`: object with `"x"`, `"y"` keys (CURVE only)
- `radius`: positive float in meters (CURVE only)
- `angleSpan`: float in radians (CURVE only). Negative = CW, positive = CCW.
- `slope`: optional float (grade, rise/run). Omit if 0.

## Verification Checklist

Before writing the file, verify ALL of these:

1. **G0 Continuity**: For every consecutive segment pair, `|seg[i].endPoint - seg[i+1].startPoint| < 0.001m` in both x and y.
2. **Loop closure**: `|last.endPoint - first.startPoint| < 0.01m`.
3. **Radius floor**: All curve radii >= 25m (must exceed `TRACK_HALF_WIDTH = 13.3m` with margin).
4. **Consistent winding**: All `angleSpan` values have the same sign.
5. **Turn angle sum**: Sum of `|angleSpan|` per turn ≈ pi (within 0.05 radians).
6. **Center-radius consistency**: For each CURVE, verify `|center - startPoint|` and `|center - endPoint|` both ≈ `radius` (within 0.1m).
7. **Tangent continuity (G1)**: At each segment boundary, the exit tangent direction of segment i should match the entry tangent direction of segment i+1 within 0.05 radians.

## Post-Generation Test

After writing the file, run:
```bash
python -c "from horse_racing.track import load_track; t = load_track('tracks/<name>.json'); print(f'{len(t.segments)} segments, {len(t.inner_rails)} inner rails, {len(t.outer_rails)} outer rails loaded successfully')"
```

## Important Constraints

- **Minimum radius**: Never generate curves with radius < 25m. The track half-width is 13.3m, so inner rails at `radius - 13.3m` would collapse or invert below that.
- **Use many small curves for turns**: Never use a single large arc for a 180-degree turn. Always decompose into 8-14 sub-segments with varying radii. This is critical for realistic geometry and smooth rail generation.
- **Compute precisely**: Use full floating-point precision for all coordinates. Do not round intermediate values.
- **The algorithm guarantees G1 continuity** because each curve segment's center is placed perpendicular to the current heading. When the next segment has a different radius, its center shifts accordingly, but the tangent direction is preserved.
