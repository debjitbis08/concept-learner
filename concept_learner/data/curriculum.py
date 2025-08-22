from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class PhaseSpec:
    name: str
    relations: List[str]
    notes: str = ""


PHASES: Dict[int, PhaseSpec] = {
    0: PhaseSpec(
        name="Phase 0 — Sensorimotor properties & shapes",
        relations=["has_color", "has_shape", "has_size", "is_a"],
        notes="shared codes for color/shape/size across contexts",
    ),
    1: PhaseSpec(
        name="Phase 1 — Objects & categories",
        relations=["is_a", "part_of"],
        notes="taxonomic structure; part-whole",
    ),
    2: PhaseSpec(
        name="Phase 2 — Spatial & actions",
        relations=["on", "in", "under", "next_to", "before", "after"],
        notes="basic spatial/temporal",
    ),
    3: PhaseSpec(
        name="Phase 3 — Pre-number sense",
        relations=["count", "more_than", "fewer_than", "equal_count"],
        notes="subitizing & comparison",
    ),
    4: PhaseSpec(
        name="Phase 4 — Numbers",
        relations=["successor_of", "predecessor_of", "ordinal_of", "count"],
        notes="digits/number-words and successor",
    ),
    5: PhaseSpec(
        name="Phase 5 — Operations",
        relations=["add_1", "add_2", "add_k", "remove_1", "sum_of", "makes_ten_with"],
        notes="transformations",
    ),
    6: PhaseSpec(
        name="Phase 6 — Place value & number line",
        relations=["has_tens", "has_ones", "less_than", "greater_than", "distance_to"],
        notes="composition and ordering",
    ),
    7: PhaseSpec(
        name="Phase 7 — Word problems & mixed",
        relations=["cost_of", "length_of", "time_of_day"],
        notes="apply numbers to contexts",
    ),
}


SEEDS = ["apple", "ball", "cat", "dog", "1", "2", "3", "ten", "red", "round"]


def allowed_relations_for_phase(phase: int) -> List[str]:
    spec = PHASES.get(int(phase), PHASES[0])
    return spec.relations


def make_generator_prompt_for_phase(phase: int) -> str:
    spec = PHASES.get(int(phase), PHASES[0])
    rels = spec.relations
    return (
        f"You are creating kindergarten-level knowledge for {spec.name}.\n"
        f"Keep concepts concrete and culture-neutral. Use allowed relations only: {rels}.\n\n"
        "Produce STRICT JSON with:\n"
        "- concepts (nouns, number words/digits as needed),\n"
        "- relations (subset of allowed),\n"
        "- triples (20–60),\n"
        "- analogies (8–16) that reflect the current phase transformations,\n"
        "- equivalences (e.g., synonyms, number word ↔ digit),\n"
        "- negatives (15–25%) that are plausible but wrong.\n\n"
        f"Ensure cross-context overlap using these shared seeds: {SEEDS}.\n"
        "Output JSON only."
    )

