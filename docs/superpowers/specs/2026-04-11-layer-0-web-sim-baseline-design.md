# Layer 0: Web Sim Baseline

## Context

After v60/v61/v62 of the RL training pipeline repeatedly collapsed into cliff-riding exploits despite ~540 lines of hand-tuned reward shaping (11 reward revisions from v4.0 through v6.0, plus two physics redesigns and multiple resume-stability fixes), it became clear that we're fighting the wrong layer. The reward function keeps catching the agent's exploits, but each fix creates a new exploit elsewhere. That pattern is the classic signal that the problem formulation — not the hyperparameters — is wrong.

Rather than continue iterating on the existing stack, we are **starting over from the physics layer and rebuilding one mechanic at a time**. This document describes Layer 0: the smallest runnable baseline we can stand on. Everything above Layer 0 — stamina, burst, skills, archetypes, BT opponents, RL — is deferred to future layers, each of which will get its own design spec.

### Relationship to `2026-04-04-simulation-redesign-design.md`

The April 4 spec proposed a Python-first rebuild: 8-trait horses, 5-stage curriculum, self-play, GRU-based recurrent policies, ONNX export. That spec is a **target ambition** — what the system might look like once the full mechanic stack is rebuilt. It is not a contradiction; it is the destination. This Layer 0 spec is the **first step of the incremental path** toward something like that ambition.

The two differ in staging, not in direction:
- April 4 assumes a top-down rebuild: specify all the mechanics upfront, then implement them together.
- This spec takes the opposite approach: rebuild the ground floor, validate it feels right, add one mechanic per layer, and decide each new mechanic's design after experiencing what the previous layer taught us.

If the April 4 spec is ever revisited, Layer 0 through some later layer will have already landed its foundation, and its shape will reflect hands-on experience rather than a paper plan.

## Goals

1. **Strip the web sim down to pure physics + auto-cruise + manual player control.** No stamina, no burst pool, no cliff collapse, no skills, no modifiers, no archetypes, no BT personalities, no AI/ONNX jockey, no horse attributes, no genome rolls, no reward function.
2. **Ship a playable race** — the user opens `/app`, picks one of four horses (or chooses to watch), clicks Start, drives the horse around the track with arrow keys, and sees a finish order at the end.
3. **Leave the Python sim frozen.** `horse_racing/` is untouched. It becomes dormant reference code until a future layer brings RL back, at which point it will be rebuilt to mirror whatever the web sim has become.
4. **Establish the module shape that future mechanic layers will stack onto.** File boundaries should anticipate "one new mechanic = one new file + a small edit to `sim.ts` and `physics.ts`."

## Non-Goals

Explicitly NOT in scope for Layer 0:

- Horse attributes, genomes, modifiers, or any per-horse variation. All four horses are mechanically identical. Differentiation will come from mechanics in later layers.
- Stamina, burst, cliff collapse, aerobic/anaerobic models. No energy economy of any kind.
- BT opponents, AI opponents, ONNX-driven horses. Non-player horses auto-cruise only.
- Horse-horse collision. Horses pass through each other.
- Multi-lap racing. One lap per race.
- Lap timing, split times, speedometer, HUD, telemetry overlays.
- Mobile / touch input. Keyboard only.
- Track selection UI. Default track is `test_oval.json`, loaded on mount. Changing track = edit a constant and reload.
- Sound, music, or visual polish beyond "horses are colored rectangles on a drawn track."
- Python-side changes of any kind. No edits to `horse_racing/`, `notebooks/`, `scripts/`, or `tracks/`.

## Scope of the Rework

**Web sim (`apps/horse-racing/src/` in the `ue-too/hr-racing` `refactor/hr-racing` worktree):**

- **Rebuilt:** the `src/simulation/` directory, `src/utils/init-app.ts`, the toolbar component, the `App.tsx` page entry.
- **Kept untouched:** `src/simulation/track-types.ts`, `src/simulation/track-navigator.ts`, `src/simulation/track-from-json.ts` (these constitute the "physics infrastructure tied to the web sim's track maker" that is out of scope), the track-maker page at `/track-maker`, the landing page at `/`, the i18n machinery, the `@ue-too/board-pixi-react-integration` wrapper.
- **Deleted at cutover:** `horse-racing-engine.ts` (910 lines), `horse-racing-sim.ts` (1190 lines), `stamina.ts`, `bt-jockey.ts`, `ai-jockey.ts`, `horse-attributes.ts`, `horse-genome.ts`, `modifiers.ts`.

**Python sim (`horse_racing/`):** frozen. No edits.

## Architecture

### Approach: build alongside, then swap

The rework is implemented as a new module `src/simulation/v2/` living alongside the existing `src/simulation/` code. A new route `/app/v2` mounts the new sim. The old `/app` route continues to work throughout the build. When the new sim reaches acceptance criteria, a single cutover commit deletes the old files, renames `v2/` into place, and points `/app` at the new sim.

Rationale:

- The existing `horse-racing-engine.ts` and `horse-racing-sim.ts` carry shape decisions that only made sense when stamina, archetypes, BT, and ONNX were in the mix. Stripping in place would produce a smaller version of the *old* shape, which is the wrong target. A fresh module built for "mechanics stack cleanly on top" is a better foundation.
- Parallel-run means the app remains runnable at every commit. Nothing is broken during the build phase.
- The track-maker page is unaffected throughout.

### Sim/React boundary

The V2 sim is a single imperative class (`V2Sim`) that owns all simulation state (horses, race phase, input, Pixi display objects). React's role is to render a canvas host plus DOM chrome (toolbar, horse picker, end overlay) and to call imperative methods on the sim (`start`, `reset`, `pickHorse`).

Only two pieces of state leak from the sim into React: the current race phase (to decide which overlay to show) and the finish order (to populate the end overlay). These are pushed to React via a sim-emitted `onPhaseChange` event that fires 3–4 times per race, not 60 times per second. All per-tick state (horse positions, velocities, tick count) stays inside the sim and is read only by the Pixi renderer.

Rationale:

- Pushing 60fps state into React state is wasteful and produces no user-visible benefit.
- Splitting race state across two owners creates synchronization bugs (the "I clicked Reset but the Pixi loop didn't stop" class).
- The existing `init-app.ts` + `HorseRacingSimHandle` pattern already points this direction. We are continuing a convention, not inventing one.

### Module layout

```
apps/horse-racing/src/simulation/v2/
├── types.ts          — Horse, RaceState, InputState, RacePhase, constants
├── physics.ts        — pure step function: (horses, input, playerId, track, dt) → mutates horses
├── cruise.ts         — cruise controller: (currentVel, targetVel) → force
├── input.ts          — keyboard listener → InputState ref + dispose
├── race.ts           — Race class: state machine (gate | running | finished), finish detection
├── renderer.ts       — Pixi display objects, syncHorses reader
├── sim.ts            — V2Sim class: owns loop, composes physics + cruise + input + race + renderer
└── index.ts          — exports V2Sim, attachV2Sim(), types
```

Granularity rationale: seven small files, one concern each, 50–150 lines each. Layer 1 will add `stamina.ts` plus edits to `physics.ts` and `sim.ts`. Layer 2 (probably AI opponents of some shape) will add `auto-jockey.ts` plus edits to `sim.ts` to install it. File boundaries are drawn to match the mechanic-per-layer rebuild pattern.

New React surfaces:

```
apps/horse-racing/src/pages/race-v2.tsx               — new page, mounts V2Sim
apps/horse-racing/src/components/race-v2/
  ├── HorsePicker.tsx        — gate-phase color picker + Watch button
  ├── RaceToolbar.tsx        — Start / Reset buttons, phase indicator
  └── RaceEndOverlay.tsx     — finished-phase finish-order display
apps/horse-racing/src/utils/init-race-v2.ts           — mirrors init-app.ts for the new sim
```

Routing: `main-react.tsx` gains `<Route path="/app/v2" element={<RaceV2Page />} />` during the parallel-run phase. At cutover, `/app/v2` is removed and `/app` points at `RaceV2Page` (or `RaceV2Page`'s content is moved back into `App.tsx`, exact packaging decided in implementation).

## Data Model

```ts
// types.ts
export const TARGET_CRUISE = 15;      // m/s
export const F_T_MAX = 5;             // player tangential force cap (m/s²)
export const F_N_MAX = 3;             // player normal force cap (m/s²)
export const K_CRUISE = 0.5;          // cruise controller gain (1/s)
export const C_DRAG = 0.1;            // linear drag coefficient (1/s)
export const TRACK_HALF_WIDTH = 10;   // meters (matches existing track-types.ts)
export const FIXED_DT = 1 / 60;       // seconds per physics step

export interface Horse {
    id: number;                       // 0..3
    color: number;                    // 0xRRGGBB for rendering + UI swatches
    pos: { x: number; y: number };    // world-space position
    tangentialVel: number;            // along-track velocity
    normalVel: number;                // across-track velocity
    trackProgress: number;            // 0..1 along centerline (cached from navigator)
    navigator: TrackNavigator;        // per-horse track-state tracker (owned by this horse)
    finished: boolean;
    finishOrder: number | null;       // null until crossed the line
}

export interface InputState {
    tangential: -1 | 0 | 1;           // Down / none / Up
    normal: -1 | 0 | 1;               // Left / none / Right
}

export type RacePhase = 'gate' | 'running' | 'finished';

export interface RaceState {
    phase: RacePhase;
    horses: Horse[];                  // canonical, length 4
    playerHorseId: number | null;     // null = watch mode
    tick: number;                     // physics steps since race started
    finishOrder: number[];            // horse ids in finish order
}
```

### Notes on the data model

- **Each horse owns its own `TrackNavigator` instance.** The existing `TrackNavigator` class (kept as infrastructure) holds per-horse state internally: `currentIndex`, `curveEntryRadius`, `_completedLap`. Sharing one navigator across horses would corrupt that state. Per-horse instances are constructed from the same `segments` array in `spawnHorses()` and are mutated by `physics.ts` via `navigator.updateSegment(pos)` and `navigator.computeProgress(pos)` each tick.
- **`trackProgress` is cached on the `Horse`** so the renderer and race state machine can read it without re-calling `navigator.computeProgress(pos)`. The canonical value lives in the navigator; the `Horse.trackProgress` field is a read-through cache refreshed each physics step.
- **`InputState` is tri-state discrete**, not continuous, because the only input device is a keyboard and keys are held/not-held. If a later layer introduces gamepad support, the type swaps to `{ tangential: number, normal: number } ∈ [-1, 1]` and `input.ts` maps analog values into it; `physics.ts` does not need to change because it already multiplies by `F_T_MAX` / `F_N_MAX`.
- **`RaceState.horses` is the canonical store.** The renderer reads from it; it never writes. The race state machine is the only writer.
- **`playerHorseId: null` denotes watch mode.** An alternative `-1` sentinel was considered but rejected because TypeScript's null checking gives better diagnostics.

## Physics and Race Loop

### Per-tick physics step

```ts
// physics.ts (pseudocode; not the final implementation)
export function stepPhysics(
    horses: Horse[],
    input: InputState,
    playerHorseId: number | null,
    dt: number,
): void {
    for (const h of horses) {
        if (h.finished) continue;

        // 1. Forces
        const cruiseForce = computeCruiseForce(h.tangentialVel, TARGET_CRUISE);
        let F_t = cruiseForce;
        let F_n = 0;
        if (h.id === playerHorseId) {
            F_t += input.tangential * F_T_MAX;
            F_n += input.normal * F_N_MAX;
        }

        // 2. Drag (linear, on both components)
        F_t -= C_DRAG * h.tangentialVel;
        F_n -= C_DRAG * h.normalVel;

        // 3. Integrate velocity (forward Euler)
        h.tangentialVel += F_t * dt;
        h.normalVel += F_n * dt;

        // 4. Integrate position using the horse's current track frame
        const frame = h.navigator.getTrackFrame(h.pos);
        h.pos.x += (frame.tangential.x * h.tangentialVel + frame.normal.x * h.normalVel) * dt;
        h.pos.y += (frame.tangential.y * h.tangentialVel + frame.normal.y * h.normalVel) * dt;

        // 5. Segment advance + progress refresh
        h.navigator.updateSegment(h.pos);
        h.trackProgress = h.navigator.computeProgress(h.pos);

        // 6. Rail clamp (soft wall). `displacement` is computed from the
        // frame because TrackFrame itself does not expose it — on curves
        // it is `turnRadius - targetRadius`; on straights it is the
        // projection of (pos - segment.startPoint) onto frame.normal.
        // We call a helper `lateralDisplacement(frame, h.pos, navigator.segment)`
        // defined in physics.ts.
        const disp = lateralDisplacement(frame, h.pos, h.navigator);
        if (Math.abs(disp) > TRACK_HALF_WIDTH) {
            h.normalVel = 0;
            const excess = disp - Math.sign(disp) * TRACK_HALF_WIDTH;
            h.pos.x -= frame.normal.x * excess;
            h.pos.y -= frame.normal.y * excess;
        }
    }
}
```

```ts
// cruise.ts
export function computeCruiseForce(currentVel: number, targetVel: number): number {
    return K_CRUISE * (targetVel - currentVel);
}
```

### Terminal velocity note

The default constants (`K_CRUISE = 0.5`, `C_DRAG = 0.1`, `TARGET_CRUISE = 15`) give equilibrium velocity `v* = K_CRUISE * TARGET / (K_CRUISE + C_DRAG) = 12.5 m/s`, which is below the nominal cruise target. This is a known inconsistency; it is to be resolved during polish by one of:

1. Raising `K_CRUISE` until `v*` approaches `TARGET_CRUISE` (e.g., `K_CRUISE = 2.0` → `v* ≈ 14.3`).
2. Removing drag from the cruise-force path and applying it only above cruise (reformulates drag as "air resistance when pushing past cruise").
3. Reframing `TARGET_CRUISE` as the equilibrium speed directly and back-calculating the controller gain.

No constants are tuned in this spec. Tuning happens after the sim runs for the first time.

### Race state machine

```ts
// race.ts (pseudocode)
export class Race {
    state: RaceState;

    constructor(track: TrackSegment[]) {
        this.state = { phase: 'gate', horses: spawnHorses(), playerHorseId: null, tick: 0, finishOrder: [] };
    }

    start(playerHorseId: number | null): void {
        this.state.playerHorseId = playerHorseId;
        this.state.phase = 'running';
        this.state.tick = 0;
    }

    tick(input: InputState): void {
        if (this.state.phase !== 'running') return;

        stepPhysics(this.state.horses, input, this.state.playerHorseId, FIXED_DT);

        for (const h of this.state.horses) {
            if (!h.finished && h.trackProgress >= 1.0) {
                h.finished = true;
                h.finishOrder = this.state.finishOrder.length + 1;
                this.state.finishOrder.push(h.id);
            }
        }

        const isPlayerMode = this.state.playerHorseId !== null;
        const player = isPlayerMode ? this.state.horses[this.state.playerHorseId!] : null;
        const allFinished = this.state.horses.every(h => h.finished);
        if ((isPlayerMode && player!.finished) || (!isPlayerMode && allFinished)) {
            this.state.phase = 'finished';
        }

        this.state.tick++;
    }

    reset(): void {
        this.state = { phase: 'gate', horses: spawnHorses(), playerHorseId: null, tick: 0, finishOrder: [] };
    }
}
```

### End-of-race condition

- **Player mode** (`playerHorseId !== null`): the race ends when the player's horse crosses the finish line. Other horses that haven't finished are simply left where they are; they do not need to all complete.
- **Watch mode** (`playerHorseId === null`): the race ends when all four horses have crossed.

In both modes, `finishOrder` captures every horse that crossed before the end condition triggered, in order. In player mode, horses that didn't finish have `finishOrder === null` and are rendered as "DNF" in the end overlay.

### Fixed timestep

The physics step uses a fixed `dt = 1/60 s`. The Pixi ticker's frame delta is not used directly. If frame pacing produces more than one step worth of time, the sim runs multiple steps per frame (classic fixed-timestep game loop). Simpler alternative: always run exactly one step per frame and accept that slow frames make the race slower. We will pick the simpler approach in implementation and switch to multi-step only if the race visibly stutters on the target hardware.

## Input and Rendering

### Input handler

```ts
// input.ts
export function createInputHandler(): { state: InputState; dispose: () => void } {
    const state: InputState = { tangential: 0, normal: 0 };
    const keys = new Set<string>();

    const update = () => {
        state.tangential = keys.has('ArrowUp') ? 1 : keys.has('ArrowDown') ? -1 : 0;
        state.normal = keys.has('ArrowRight') ? 1 : keys.has('ArrowLeft') ? -1 : 0;
    };

    const onDown = (e: KeyboardEvent) => {
        if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) return;
        e.preventDefault();
        keys.add(e.key);
        update();
    };
    const onUp = (e: KeyboardEvent) => {
        keys.delete(e.key);
        update();
    };

    window.addEventListener('keydown', onDown);
    window.addEventListener('keyup', onUp);

    return { state, dispose: () => {
        window.removeEventListener('keydown', onDown);
        window.removeEventListener('keyup', onUp);
    } };
}
```

Design notes:

- **Mutable ref, not React state.** The physics loop reads `state` every tick. React state would force a re-render 60 times per second for no visible benefit.
- **Opposing keys cancel.** Holding Left and Right simultaneously yields `normal: 0`. Simpler than "last-pressed wins" and avoids tracking press order. If it feels bad in practice, the swap to last-pressed-wins is a few lines.
- **`preventDefault` on arrow keys** stops the page from scrolling while playing.
- **No focus handling.** The canvas is assumed to have focus during gameplay. If another element eats arrow keys in practice, we deal with it during implementation.

### Renderer

```ts
// renderer.ts (shape only)
export class RaceRenderer {
    private horseGfx: Map<number, PIXI.Graphics>;
    private trackGfx: PIXI.Graphics;
    private stage: PIXI.Container;

    constructor(stage: PIXI.Container, track: TrackSegment[]) {
        this.stage = stage;
        this.trackGfx = drawTrack(track);
        stage.addChild(this.trackGfx);
        this.horseGfx = new Map();
    }

    syncHorses(horses: Horse[], playerHorseId: number | null): void {
        for (const h of horses) {
            let gfx = this.horseGfx.get(h.id);
            if (!gfx) {
                gfx = drawHorse(h.color, h.id === playerHorseId);
                this.stage.addChild(gfx);
                this.horseGfx.set(h.id, gfx);
            }
            gfx.position.set(h.pos.x, h.pos.y);
            const frame = h.navigator.getTrackFrame(h.pos);
            gfx.rotation = Math.atan2(frame.tangential.y, frame.tangential.x);
        }
    }

    dispose(): void {
        this.horseGfx.forEach(g => g.destroy());
        this.trackGfx.destroy();
    }
}
```

- **Horse visual**: rectangle ~0.65m × 2m (matching the existing dimensions referenced in `init-app.ts`), filled with horse color. Player horse gets a thin white outline stroke. No sprites, no textures.
- **Track drawing**: a minimal `drawTrack` helper copied into v2, not imported from the legacy `horse-racing-sim.ts`. Goal: keep v2 self-contained so the cutover commit can delete the legacy module cleanly. Estimated 80–120 lines of track-geometry Pixi drawing.
- **Camera**: set once at race start. Player mode: camera follows `horses[playerHorseId].pos` each tick via `components.camera.setPosition(pos)`. Watch mode: zoom-to-fit the track bounds on race start and do not update after.
- **Finish-order overlay is NOT drawn by this module.** It is React DOM, positioned absolutely over the canvas.

## React Chrome

### `RaceV2Page`

```tsx
// pages/race-v2.tsx
export function RaceV2Page(): React.ReactNode {
    const [simHandle, setSimHandle] = useState<V2SimHandle | null>(null);
    const [phase, setPhase] = useState<RacePhase>('gate');
    const [finishOrder, setFinishOrder] = useState<number[]>([]);

    return (
        <div className="app">
            <Wrapper
                option={{ fullScreen: true, boundaries: { min: { x: -4000, y: -4000 }, max: { x: 4000, y: 4000 } } }}
                initFunction={(canvas, opt) => initRaceV2(canvas, opt, {
                    onReady: (handle) => setSimHandle(handle),
                    onPhaseChange: (p, order) => { setPhase(p); setFinishOrder(order); },
                })}
            >
                <ScrollBarDisplay />
                <RaceToolbar sim={simHandle} phase={phase} />
                {phase === 'gate' && simHandle && <HorsePicker sim={simHandle} />}
                {phase === 'finished' && simHandle && <RaceEndOverlay order={finishOrder} sim={simHandle} />}
            </Wrapper>
        </div>
    );
}
```

### `V2SimHandle`

The imperative API the sim exposes to React:

```ts
interface V2SimHandle {
    pickHorse(id: number | null): void;   // gate phase only; null = watch mode
    start(): void;                          // gate → running
    reset(): void;                          // any → gate
    getPhase(): RacePhase;                  // for toolbar button enabling
    onPhaseChange(cb: (phase: RacePhase, finishOrder: number[]) => void): () => void;
}
```

- `onPhaseChange` is subscription-based and returns an unsubscribe function, allowing React `useEffect` cleanup to tear down listeners on unmount.
- The sim calls the callback synchronously whenever `race.state.phase` transitions.
- React's `setPhase` triggers the re-render that swaps overlays in `RaceV2Page`.

### Chrome components

**`HorsePicker.tsx`** — shown only during `gate` phase. A row of four color-swatch buttons plus a "Watch" button. Click a color → `sim.pickHorse(id)`. Click Watch → `sim.pickHorse(null)`. Currently selected option is visually highlighted. This component does not start the race; it only selects.

**`RaceToolbar.tsx`** — always visible on `/app`. Contents depend on phase:
- `gate`: "Start" button, disabled until a selection has been made; a small label showing the current selection (e.g., "Horse 2" or "Watching").
- `running`: an optional "Reset" button to bail out and return to gate.
- `finished`: "Reset" button.

**`RaceEndOverlay.tsx`** — shown only during `finished` phase. Centered card showing:

```
    1st    ● [color]   (Horse N)
    2nd    ● [color]   (Horse N)
    3rd    ● [color]   (Horse N)
    4th    ● [color]   (Horse N)    (or "DNF" if not finished)

                 [ Reset ]
```

No timer, no "You placed!" callout for the player beyond their row being in the list. No animations. Celebration polish is for a later layer.

### What the React chrome does NOT do

- Does not read horse positions, velocities, or any per-tick state. All of that is Pixi-rendered from sim state.
- Does not hold canonical `phase` or `finishOrder`. Those belong to the sim. React's copies are a cache updated on phase-change events.
- Does not intercept keyboard input. `input.ts` listens on `window` directly.
- Does not style the race view. The Pixi canvas is full-screen; React chrome is positioned absolutely with a high `z-index`.

## Cutover Plan

Implementation proceeds as a sequence of commits, each of which leaves the repo in a runnable state.

1. **Scaffold.** Create `src/simulation/v2/` with empty files. Add `/app/v2` route pointing at a stub `RaceV2Page` that renders "hello". Verify `/app` still works.
2. **Physics + race, no rendering.** Implement `types.ts`, `physics.ts`, `cruise.ts`, `race.ts` as pure TypeScript. No Pixi. Write the single unit test (below) that spawns four horses, runs ~5000 ticks in watch mode, and asserts all four cross the finish line. This is the gate that proves the physics is sane before any rendering work.
3. **Renderer + Pixi integration.** Implement `renderer.ts`, `input.ts`, `sim.ts`, `init-race-v2.ts`. Wire the Pixi canvas to `/app/v2`. Temporarily hardcode `playerHorseId = 0` and `race.start(0)` on mount. Goal: open `/app/v2`, see horses move, arrow keys steer horse 0.
4. **React chrome.** Implement `HorsePicker`, `RaceToolbar`, `RaceEndOverlay`, phase event subscription. The full gate → running → finished loop is playable via DOM.
5. **Polish pass.** Tune `K_CRUISE` vs `C_DRAG` so terminal velocity matches `TARGET_CRUISE`. Adjust horse rectangle rotation if the visuals look wrong. Add camera follow smoothing if the player view is jarring. Fix anything else uncovered in steps 2–4.
6. **Cutover commit.** One large commit:
    - Delete `src/simulation/horse-racing-engine.ts`, `horse-racing-sim.ts`, `stamina.ts`, `bt-jockey.ts`, `ai-jockey.ts`, `horse-attributes.ts`, `horse-genome.ts`, `modifiers.ts`.
    - Rename `src/simulation/v2/` → `src/simulation/`.
    - Update imports in `src/utils/init-app.ts` (or replace it with the renamed `init-race-v2.ts`).
    - Update `App.tsx` to mount `RaceV2Page` content, or replace `App.tsx` with `RaceV2Page` outright.
    - Remove the `/app/v2` route from `main-react.tsx`; `/app` now points at the new sim.
    - Delete `src/components/toolbar/HorseRacingToolbar.tsx` if nothing still references it. (The `LanguageSwitcher` in the same directory is used by the landing page and stays.)

Commits 1–5 are additive and do not touch the existing `/app` route's code path. Commit 6 is the only one that removes files. If commit 6 lands cleanly and the acceptance criteria pass, Layer 0 is done.

## Acceptance Criteria

Layer 0 is complete when all of the following are observable on the cut-over `/app` route:

1. **Watch mode works.** Opening `/app`, clicking Start without picking a horse, all four horses cross the finish line, the end overlay shows an ordered 1st / 2nd / 3rd / 4th.
2. **Player mode works.** Picking Horse 2, clicking Start, holding the Up arrow, Horse 2 accelerates past the auto-cruising pack and crosses the line first; end overlay shows Horse 2 as 1st.
3. **Steering works.** In any mode, holding Right drifts the player horse toward the outer rail; holding Left drifts it toward the inner rail. Rails stop the drift without visual tunneling.
4. **Reset works.** Clicking Reset from the finished state returns to gate, allowing another horse to be picked and another race to be run.
5. **Old code is gone.** `horse-racing-engine.ts`, `horse-racing-sim.ts`, `stamina.ts`, `bt-jockey.ts`, `ai-jockey.ts`, `horse-attributes.ts`, `horse-genome.ts`, `modifiers.ts` do not exist in `src/simulation/`.
6. **Track-maker still works.** `/track-maker` loads, renders, and can edit tracks as before.
7. **Python sim is untouched.** `git status` in the `hr-simulation` repo shows no changes under `horse_racing/`, `notebooks/`, `scripts/`, or `tracks/` as a result of this work.

## Testing

Deliberately minimal.

- **One unit test**, `test/physics.test.ts`: construct a `Race` over `test_oval.json`, call `start(null)`, loop `race.tick(zeroInput)` until `state.phase === 'finished'` or a 10,000-tick safety bound. Assert all four horses reached `finished === true` and that their tick-count-to-finish is within 5% of each other (sanity: identical horses under identical forces should finish nearly simultaneously). This test catches physics-step regressions caused by future layer edits.
- **No React component tests.** The three chrome components are simple enough that manually clicking through the page is the test.
- **No Playwright / e2e tests.** Overkill for Layer 0. If we regret it, a future layer adds them.

The existing `apps/horse-racing/test/displacement-probe.test.ts` covers the track navigator's sign convention. It does not depend on engine internals and should keep passing through the cutover because the track infrastructure is not touched. If it breaks, the break represents a track-infrastructure regression and blocks the cutover commit.

## Open Questions

None blocking. Two items flagged for resolution during implementation rather than spec:

- **Terminal velocity tuning** (physics section above) — one of three options will be selected when the sim first runs and the equilibrium speed is observed.
- **Track drawing helper extraction vs copy** (renderer section above) — final decision depends on whether a clean extraction from the legacy module is possible without pulling in unrelated dependencies. The default is to copy.

## What Layer 1 Might Be

Not part of this spec, but worth noting so the module layout makes sense:

The most likely Layer 1 is a **stamina economy**: horses have a shared `stamina` pool that drains when `tangentialVel > TARGET_CRUISE` and recovers when at or below cruise. Players can spend stamina for short bursts of speed; watch-mode horses drain/refill based on a simple policy (e.g., "always cruise, never burst"). This adds one file (`stamina.ts`), minor edits to `physics.ts` (add drain/recover term), one field to `Horse` in `types.ts`, and a small overlay in `renderer.ts` or React chrome to visualize the stamina bar. The point of starting this simple is that we will decide Layer 1's exact shape after Layer 0 feels right, not now.
