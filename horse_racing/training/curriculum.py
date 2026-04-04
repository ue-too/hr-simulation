"""Training curriculum — 5-stage progression from basic to competitive racing.

Each stage defines tracks, field size, max steps, training budget, and a
gate condition that must be met before advancing to the next stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StageConfig:
    """Configuration for a single training stage."""

    name: str
    tracks: list[str]
    min_horses: int
    max_horses: int
    max_steps: int
    timesteps: int  # total training timesteps for this stage
    randomize_horses: bool = True
    randomize_jockey_style: bool = False
    self_play: bool = False
    opponent_paths: list[str] = field(default_factory=list)

    # Gate: minimum metric value to advance
    gate_metric: str = "finish_rate"  # key in evaluation metrics
    gate_threshold: float = 0.8


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES: list[StageConfig] = [
    # Stage 1: Basic effort control on a straight track
    # Learn to push forward and finish the race
    StageConfig(
        name="stage_1_straight",
        tracks=["tracks/curriculum_1_straight.json"],
        min_horses=1,
        max_horses=1,
        max_steps=3000,
        timesteps=500_000,
        randomize_horses=True,
        randomize_jockey_style=False,
        gate_metric="finish_rate",
        gate_threshold=0.90,
    ),
    # Stage 2: Cornering and lane positioning
    # Learn to navigate curves, use left/right grip
    StageConfig(
        name="stage_2_curves",
        tracks=[
            "tracks/curriculum_2_gentle_oval.json",
            "tracks/curriculum_3_tight_oval.json",
        ],
        min_horses=1,
        max_horses=1,
        max_steps=5000,
        timesteps=750_000,
        randomize_horses=True,
        randomize_jockey_style=False,
        gate_metric="finish_rate",
        gate_threshold=0.80,
    ),
    # Stage 3: Multi-horse awareness
    # Learn collision avoidance, drafting, basic positioning
    StageConfig(
        name="stage_3_multi_horse",
        tracks=[
            "tracks/tokyo.json",
            "tracks/kokura.json",
            "tracks/curriculum_2_gentle_oval.json",
        ],
        min_horses=2,
        max_horses=4,
        max_steps=5000,
        timesteps=1_000_000,
        randomize_horses=True,
        randomize_jockey_style=False,
        gate_metric="top2_rate",  # fraction of episodes finishing 1st or 2nd
        gate_threshold=0.60,
    ),
    # Stage 4: Competitive self-play
    # Learn tactics against frozen opponents
    StageConfig(
        name="stage_4_self_play",
        tracks=[
            "tracks/tokyo.json",
            "tracks/kokura.json",
            "tracks/hanshin.json",
            "tracks/kyoto.json",
        ],
        min_horses=4,
        max_horses=10,
        max_steps=6000,
        timesteps=2_000_000,
        randomize_horses=True,
        randomize_jockey_style=True,  # start seeing jockey style variation
        self_play=True,
        gate_metric="win_rate",
        gate_threshold=0.30,
    ),
    # Stage 5: Full field with diverse jockey styles
    # Learn to condition on jockey style parameters
    StageConfig(
        name="stage_5_diverse",
        tracks=[
            "tracks/tokyo.json",
            "tracks/kokura.json",
            "tracks/hanshin.json",
            "tracks/kyoto.json",
            "tracks/tokyo_2600.json",
            "tracks/niigata.json",
        ],
        min_horses=6,
        max_horses=20,
        max_steps=8000,
        timesteps=3_000_000,
        randomize_horses=True,
        randomize_jockey_style=True,
        self_play=True,
        gate_metric="style_diversity",  # custom metric: do different styles produce different behavior?
        gate_threshold=0.50,
    ),
]


def get_stage(stage_num: int) -> StageConfig:
    """Get stage config by 1-indexed stage number."""
    if stage_num < 1 or stage_num > len(STAGES):
        raise ValueError(f"Stage {stage_num} does not exist (1-{len(STAGES)})")
    return STAGES[stage_num - 1]
