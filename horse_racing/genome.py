"""Horse genome: genes, alleles, expression, and breeding."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import random

from horse_racing.attributes import TRAIT_RANGES, CoreAttributes


@dataclass
class Gene:
    sire: float  # [0, 1]
    dam: float  # [0, 1]


@dataclass
class HorseGenome:
    core: dict[str, Gene] = field(default_factory=dict)  # one gene per core trait
    modifiers: dict[str, tuple[Gene, Gene]] = field(
        default_factory=dict
    )  # (presence_gene, strength_gene) per modifier


# ---------------------------------------------------------------------------
# Expression
# ---------------------------------------------------------------------------


def express_core_trait(gene: Gene, trait_range: tuple[float, float]) -> float:
    expressed = gene.sire * 0.5 + gene.dam * 0.5
    min_val, max_val = trait_range
    return min_val + expressed * (max_val - min_val)


def express_genome(genome: HorseGenome) -> CoreAttributes:
    attrs: dict[str, float] = {}
    for trait, gene in genome.core.items():
        attrs[trait] = express_core_trait(gene, TRAIT_RANGES[trait])
    return CoreAttributes(**attrs)


# ---------------------------------------------------------------------------
# Modifier presence / strength from genome
# ---------------------------------------------------------------------------

MODIFIER_PRESENCE_THRESHOLD: float = 0.4


def modifier_is_present(presence_gene: Gene) -> bool:
    return max(presence_gene.sire, presence_gene.dam) >= MODIFIER_PRESENCE_THRESHOLD


def modifier_strength(strength_gene: Gene) -> float:
    return strength_gene.sire * 0.5 + strength_gene.dam * 0.5


# ---------------------------------------------------------------------------
# Breeding
# ---------------------------------------------------------------------------


def breed_gene(sire_gene: Gene, dam_gene: Gene, mutation_rate: float = 0.05) -> Gene:
    from_sire = sire_gene.sire if random() < 0.5 else sire_gene.dam
    from_dam = dam_gene.sire if random() < 0.5 else dam_gene.dam

    def mutate(v: float) -> float:
        if random() < mutation_rate:
            return max(0.0, min(1.0, v + (random() - 0.5) * 0.1))
        return v

    return Gene(sire=mutate(from_sire), dam=mutate(from_dam))


def breed(sire: HorseGenome, dam: HorseGenome, mutation_rate: float = 0.05) -> HorseGenome:
    child = HorseGenome()

    # Core traits
    for trait in TRAIT_RANGES:
        sg = sire.core.get(trait, Gene(0.5, 0.5))
        dg = dam.core.get(trait, Gene(0.5, 0.5))
        child.core[trait] = breed_gene(sg, dg, mutation_rate)

    # Modifier genes
    all_mod_keys = set(sire.modifiers) | set(dam.modifiers)
    for key in all_mod_keys:
        s_pres, s_str = sire.modifiers.get(key, (Gene(0.0, 0.0), Gene(0.5, 0.5)))
        d_pres, d_str = dam.modifiers.get(key, (Gene(0.0, 0.0), Gene(0.5, 0.5)))
        child.modifiers[key] = (
            breed_gene(s_pres, d_pres, mutation_rate),
            breed_gene(s_str, d_str, mutation_rate),
        )

    return child


# ---------------------------------------------------------------------------
# Utility: create a default genome (all traits at midpoint)
# ---------------------------------------------------------------------------


def default_genome() -> HorseGenome:
    genome = HorseGenome()
    for trait in TRAIT_RANGES:
        genome.core[trait] = Gene(sire=0.5, dam=0.5)
    return genome


def random_genome() -> HorseGenome:
    genome = HorseGenome()
    for trait in TRAIT_RANGES:
        genome.core[trait] = Gene(sire=random(), dam=random())
    # Random modifier genes
    from horse_racing.modifiers import MODIFIER_REGISTRY

    for mod_id in MODIFIER_REGISTRY:
        genome.modifiers[mod_id] = (
            Gene(sire=random(), dam=random()),
            Gene(sire=random(), dam=random()),
        )
    return genome
