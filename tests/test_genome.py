"""Tests for genome expression and breeding."""

from horse_racing.attributes import TRAIT_RANGES
from horse_racing.genome import (
    Gene,
    HorseGenome,
    breed,
    default_genome,
    express_core_trait,
    express_genome,
    modifier_is_present,
    modifier_strength,
    random_genome,
)


def test_express_core_trait_midpoint():
    gene = Gene(sire=0.5, dam=0.5)
    val = express_core_trait(gene, (10.0, 20.0))
    assert abs(val - 15.0) < 0.01


def test_express_core_trait_min():
    gene = Gene(sire=0.0, dam=0.0)
    val = express_core_trait(gene, (10.0, 20.0))
    assert abs(val - 10.0) < 0.01


def test_express_core_trait_max():
    gene = Gene(sire=1.0, dam=1.0)
    val = express_core_trait(gene, (10.0, 20.0))
    assert abs(val - 20.0) < 0.01


def test_default_genome_gives_midpoint_attributes():
    genome = default_genome()
    attrs = express_genome(genome)
    # midpoint of cruise_speed: (8+18)/2 = 13
    assert abs(attrs.cruise_speed - 13.0) < 0.01
    # midpoint of weight: (400+600)/2 = 500
    assert abs(attrs.weight - 500.0) < 0.01


def test_modifier_presence():
    assert modifier_is_present(Gene(sire=0.5, dam=0.1))  # max=0.5 >= 0.4
    assert not modifier_is_present(Gene(sire=0.3, dam=0.2))  # max=0.3 < 0.4


def test_modifier_strength_blending():
    strength = modifier_strength(Gene(sire=0.8, dam=0.4))
    assert abs(strength - 0.6) < 0.01


def test_breed_produces_valid_genome():
    sire = random_genome()
    dam = random_genome()
    child = breed(sire, dam)

    attrs = express_genome(child)
    for trait in TRAIT_RANGES:
        lo, hi = TRAIT_RANGES[trait]
        val = getattr(attrs, trait)
        assert lo <= val <= hi, f"{trait}={val} outside [{lo}, {hi}]"


def test_random_genome_has_all_traits():
    genome = random_genome()
    for trait in TRAIT_RANGES:
        assert trait in genome.core
