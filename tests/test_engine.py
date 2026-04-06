"""Tests for the horse racing engine."""

import json

import numpy as np

from horse_racing.engine import EngineConfig, HorseRacingEngine
from horse_racing.types import HORSE_COUNT, HorseAction


SIMPLE_OVAL = "tracks/test_oval.json"


def test_engine_creates_horses():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    assert len(engine.horses) == HORSE_COUNT


def test_engine_initial_positions_differ():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    positions = [hs.body.position.copy() for hs in engine.horses]
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            assert dist > 0.5, "Horses should start at different positions"


def test_engine_step_zero_actions():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    actions = [HorseAction() for _ in range(HORSE_COUNT)]

    initial_positions = [hs.body.position.copy() for hs in engine.horses]
    engine.step(actions)
    final_positions = [hs.body.position.copy() for hs in engine.horses]

    # Horses should have moved (auto-cruise kicks in)
    for i in range(HORSE_COUNT):
        dist = float(np.linalg.norm(final_positions[i] - initial_positions[i]))
        assert dist > 0.0, f"Horse {i} did not move"


def test_engine_step_forward_action():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    actions = [HorseAction(extra_tangential=5.0) for _ in range(HORSE_COUNT)]
    engine.step(actions)

    # Horses should have positive forward velocity
    for hs in engine.horses:
        if hs.frame is not None:
            tang_vel = float(np.dot(hs.body.velocity, hs.frame.tangential))
            assert tang_vel > 0, "Horse should be moving forward"


def test_engine_observations():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    actions = [HorseAction() for _ in range(HORSE_COUNT)]
    engine.step(actions)

    obs = engine.get_observations()
    assert len(obs) == HORSE_COUNT

    for o in obs:
        assert "tangential_vel" in o
        assert "track_progress" in o
        assert "stamina_ratio" in o


def test_engine_obs_to_array():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    actions = [HorseAction() for _ in range(HORSE_COUNT)]
    engine.step(actions)

    obs = engine.get_observations()
    arr = engine.obs_to_array(obs[0])
    assert arr.shape == (111,)
    assert arr.dtype == np.float32


def test_relatives_sorted_by_progress():
    """Relative horse obs should be sorted by track progress (ahead first)."""
    from horse_racing.engine import EngineConfig

    engine = HorseRacingEngine(SIMPLE_OVAL, EngineConfig(horse_count=5))
    # Give horse 1 a head start by accelerating it
    for _ in range(50):
        actions = [HorseAction()] * 5
        actions[1] = HorseAction(extra_tangential=10.0)
        engine.step(actions)

    obs = engine.get_observations()
    obs0 = obs[0]

    # relatives should have 19 entries (padded)
    assert len(obs0["relatives"]) == 19

    # 4 actual horses, 15 zero-padded
    non_zero = [r for r in obs0["relatives"] if any(v != 0.0 for v in r)]
    assert len(non_zero) == 4

    # Each entry has 4 features: tang_off, norm_off, rel_tang_vel, rel_norm_vel
    for r in obs0["relatives"]:
        assert len(r) == 4

    # First relative should be the horse furthest ahead (horse 1 which accelerated)
    arr = engine.obs_to_array(obs0)
    assert arr.shape == (111,)
    # rel_horse_1 tang offset (index 8) should be positive (ahead)
    assert arr[8] > 0, f"Expected first relative horse ahead, got tang_off={arr[8]}"


def test_engine_reset():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    actions = [HorseAction(extra_tangential=5.0) for _ in range(HORSE_COUNT)]

    # Run a few steps
    for _ in range(10):
        engine.step(actions)

    # Reset
    engine.reset()
    assert engine.tick == 0
    for hs in engine.horses:
        assert hs.track_progress == 0.0
        assert not hs.finished


def test_engine_progress_increases():
    engine = HorseRacingEngine(SIMPLE_OVAL)
    actions = [HorseAction(extra_tangential=3.0) for _ in range(HORSE_COUNT)]

    for _ in range(50):
        engine.step(actions)

    for hs in engine.horses:
        assert hs.track_progress > 0.0, "Progress should increase after running"


def test_engine_custom_horse_count():
    config = EngineConfig(horse_count=2)
    engine = HorseRacingEngine(SIMPLE_OVAL, config=config)
    assert len(engine.horses) == 2


def test_auto_cruise_blend():
    """With zero jockey input, auto-cruise drives the horse. With large input, jockey overrides."""
    engine = HorseRacingEngine(SIMPLE_OVAL)

    # Run 100 steps with zero actions — horse should approach cruise speed
    for _ in range(100):
        engine.step([HorseAction() for _ in range(HORSE_COUNT)])
    obs_cruise = engine.get_observations()
    cruise_vel = obs_cruise[0]["tangential_vel"]
    cruise_speed = obs_cruise[0]["effective_cruise_speed"]
    assert abs(cruise_vel - cruise_speed) < 1.0, (
        f"Zero input: vel {cruise_vel:.2f} should be near cruise {cruise_speed:.2f}"
    )

    # Now apply large positive input — horse should exceed cruise speed
    for _ in range(100):
        engine.step([HorseAction(extra_tangential=5.0) for _ in range(HORSE_COUNT)])
    obs_fast = engine.get_observations()
    fast_vel = obs_fast[0]["tangential_vel"]
    assert fast_vel > cruise_speed + 0.5, (
        f"Large input: vel {fast_vel:.2f} should be well above cruise {cruise_speed:.2f}"
    )

    # Apply large negative input — horse should go below cruise speed
    engine.reset()
    for _ in range(100):
        engine.step([HorseAction() for _ in range(HORSE_COUNT)])
    for _ in range(50):
        engine.step([HorseAction(extra_tangential=-5.0) for _ in range(HORSE_COUNT)])
    obs_slow = engine.get_observations()
    slow_vel = obs_slow[0]["tangential_vel"]
    assert slow_vel < cruise_speed - 0.5, (
        f"Negative input: vel {slow_vel:.2f} should be well below cruise {cruise_speed:.2f}"
    )


def test_obs_array_matches_schema():
    """Verify obs_to_array() output matches the shared obs_schema.json."""
    with open("obs_schema.json") as f:
        schema = json.load(f)

    engine = HorseRacingEngine(SIMPLE_OVAL)
    actions = [HorseAction() for _ in range(HORSE_COUNT)]
    engine.step(actions)

    obs = engine.get_observations()
    arr = engine.obs_to_array(obs[0])

    assert arr.shape == (schema["size"],), (
        f"obs_to_array size {arr.shape[0]} != schema size {schema['size']}"
    )
    assert len(schema["fields"]) == schema["size"], (
        f"Schema field count {len(schema['fields'])} != declared size {schema['size']}"
    )


# ── Skill-physics modifier tests ──────────────────────────────────────


def test_skill_modifiers_change_effective_attrs():
    """pace_pressure skill should increase max_speed and drain_rate_mult."""
    engine = HorseRacingEngine(SIMPLE_OVAL)
    engine.active_skills = {"pace_pressure"}
    actions = [HorseAction() for _ in range(HORSE_COUNT)]
    engine.step(actions)

    hs0 = engine.horses[0]
    base_max_speed = hs0.base_attrs.max_speed
    base_drain = hs0.base_attrs.drain_rate_mult

    # Effective should be boosted by skill modifier
    assert hs0.effective_attrs.max_speed > base_max_speed, (
        f"pace_pressure should increase max_speed: {hs0.effective_attrs.max_speed} <= {base_max_speed}"
    )
    assert hs0.effective_attrs.drain_rate_mult > base_drain, (
        f"pace_pressure should increase drain: {hs0.effective_attrs.drain_rate_mult} <= {base_drain}"
    )


def test_skill_modifiers_only_affect_horse_0():
    """Skill modifiers should only apply to horse 0 (trainee)."""
    config = EngineConfig(horse_count=3)
    engine = HorseRacingEngine(SIMPLE_OVAL, config=config)
    engine.active_skills = {"pace_pressure"}
    actions = [HorseAction() for _ in range(3)]
    engine.step(actions)

    # Horse 1 and 2 should have base attrs (no skill boost)
    for i in [1, 2]:
        hs = engine.horses[i]
        assert abs(hs.effective_attrs.max_speed - hs.base_attrs.max_speed) < 2.0, (
            f"Horse {i} should not get skill modifier boost"
        )


def test_sprint_timing_progress_gated():
    """sprint_timing should only activate after 75% progress."""
    engine = HorseRacingEngine(SIMPLE_OVAL)
    engine.active_skills = {"sprint_timing"}

    # Run a few steps (horse is at start, progress < 0.75)
    actions = [HorseAction(extra_tangential=3.0) for _ in range(HORSE_COUNT)]
    for _ in range(10):
        engine.step(actions)

    hs0 = engine.horses[0]
    assert hs0.track_progress < 0.75, "Horse should not be at 75% yet"
    # sprint_timing should NOT be active
    active_ids = {m.id for m in hs0.runtime.active_modifiers}
    assert "skill_sprint_timing" not in active_ids, (
        "sprint_timing should not activate before 75% progress"
    )

    # Run until past 75%
    for _ in range(500):
        engine.step(actions)
        if engine.horses[0].track_progress > 0.75:
            break

    if engine.horses[0].track_progress > 0.75:
        engine.step(actions)
        active_ids = {m.id for m in engine.horses[0].runtime.active_modifiers}
        assert "skill_sprint_timing" in active_ids, (
            "sprint_timing should activate after 75% progress"
        )


def test_stamina_management_reduces_drain():
    """stamina_management skill should reduce drain_rate_mult."""
    engine = HorseRacingEngine(SIMPLE_OVAL)
    engine.active_skills = {"stamina_management"}
    actions = [HorseAction() for _ in range(HORSE_COUNT)]
    engine.step(actions)

    hs0 = engine.horses[0]
    assert hs0.effective_attrs.drain_rate_mult < hs0.base_attrs.drain_rate_mult, (
        f"stamina_management should reduce drain: "
        f"{hs0.effective_attrs.drain_rate_mult} >= {hs0.base_attrs.drain_rate_mult}"
    )


def test_no_skills_uses_base_attrs():
    """With no skills, effective attrs should match base (plus genome modifiers only)."""
    engine = HorseRacingEngine(SIMPLE_OVAL)
    # active_skills defaults to empty set
    actions = [HorseAction() for _ in range(HORSE_COUNT)]
    engine.step(actions)

    hs0 = engine.horses[0]
    # Should be close to base (genome modifiers may differ slightly)
    assert abs(hs0.effective_attrs.max_speed - hs0.base_attrs.max_speed) < 2.0


def test_skill_biased_genome_distribution():
    """skill_biased_genome should produce genes biased toward skill traits."""
    from horse_racing.genome import skill_biased_genome, express_genome

    # Generate many genomes for pace_pressure
    max_speeds = []
    for _ in range(100):
        genome = skill_biased_genome({"pace_pressure"})
        attrs = express_genome(genome)
        max_speeds.append(attrs.max_speed)

    mean_max_speed = np.mean(max_speeds)
    # Bias center is 0.8, trait range is [16.0, 20.0], so expected ~19.2
    # With Beta(6.4, 1.6) std ~0.12 → trait std ~0.48
    # Mean should be well above midpoint (18.0)
    assert mean_max_speed > 18.5, (
        f"pace_pressure genomes should bias max_speed high: mean={mean_max_speed:.2f}"
    )
