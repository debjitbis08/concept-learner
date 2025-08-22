Child‑like Curriculum (with numeric extensions)

This document records the training phases and how they map to the current codebase. Each phase lists: goals, allowed relations, synthetic data ideas, example triples/analogies, heads/losses, metrics, and current implementation status.

Legend for status
- Done: implemented and usable in repo
- Partial: some pieces implemented; more to do
- TODO: not implemented yet

Phase 0 — Sensorimotor “properties & shapes”
- Goals: shared codes for color/shape/size across contexts
- Relations: has_color, has_shape, has_size (small/medium/big), is_a
- Data: single objects in varied contexts (home, park, classroom), paraphrases
- Episode template: k views of 1–2 objects across 2 contexts (e.g., <home> apple vs <market> apple)
- Triples: ["apple","has_color","red"], ["ball","has_shape","round"], ["triangle","has_sides","3"]
- Analogies: [["apple","red"],["grass","green"]]
- Heads/losses: ↑ contrastive, ↑ VQ entropy/usage; light relation loss
- Metrics: codebook utilization, cross-context code sharing, nearest-neighbor purity
- Status: TODO (no shapes/objects domain yet)

Phase 1 — Objects & categories (early nouns)
- Goals: stable taxonomic structure; begin part‑whole
- Relations: is_a, part_of (keep Phase‑0)
- Data: animals, foods, toys; synonyms (bike/bicycle), cross‑situations
- Triples: ["kitten","is_a","cat"], ["wheel","part_of","bicycle"]
- Analogies: [["puppy","dog"],["kitten","cat"]]
- Heads/losses: ↑ DistMult (triples), keep contrastive; add equivalence pairs for match head
- Metrics: link‑prediction Hits@K, synonym matching accuracy
- Status: TODO (LLMTeacher scaffolding available; no generator/domain yet)

Phase 2 — Spatial & actions (situated reasoning)
- Goals: basic spatial/temporal relations; simple actions
- Relations: on, in, under, next_to, optional before/after
- Data: short captions or paired sentences
- Triples: ["ball","on","table"], ["book","in","bag"]
- Analogies: [["cup","on_table"],["book","on_shelf"]]
- Heads/losses: keep DistMult; small speaker–listener head helps grounding
- Metrics: spatial link prediction; cross‑context generalization
- Status: TODO

Phase 3 — Pre‑number sense (subitizing & comparison)
- Goals: recognize quantity without counting (1–4), compare counts
- Relations: count, more_than, fewer_than, equal_count
- Data: same count across different objects/contexts (e.g., “two apples”, “two cars”); textual and token‑count views (e.g., “apple apple” ×2)
- Triples: ["2_apples","count","2"], ["three_ducks","count","3"]
- Analogies: [["2","more_than","1"],["3","more_than","2"]]
- Heads/losses: contrastive (same count, diff objects), plus DistMult for count/compare
- Metrics: count recognition across objects; comparison accuracy
- Status: Partial
  - Available: numeric_gold_atoms includes one count triple; LLMTeacher prompt/validators in place
  - TODO: dedicated count episode generator (text views like "apple apple") and compare relations

Phase 4 — Numbers (cardinality, ordinality, successor)
- Goals: discrete codes for digits/number words; successor/predecessor
- Relations: successor_of, predecessor_of, ordinal_of, keep count
- Data: align “3”, “three”, “III”; link sets ↔ numerals; “3rd”, “first/second/third”
- Triples: ["3","successor_of","2"], ["2","predecessor_of","3"], ["3_apples","count","3"]
- Analogies: Add‑1 ([["2","3"],["5","6"]]), Add‑2 ([["2","4"],["3","5"]]), Ordinal ([["1","1st"],["3","3rd"]])
- Heads/losses: strong analogy (+k offsets); relation loss
- Metrics: analogy offset cosine for +k; numeral/word alignment accuracy
- Status: Partial
  - Done: successor_of, predecessor_of (EpisodeGenerator), add_2 analogy family, LLMTeacher integration, equivalence example ("three" ↔ "3")
  - TODO: ordinal_of relation and stronger number‑word alignment/match head

Phase 5 — Operations as transformations (add/remove/compose)
- Goals: addition/subtraction as relations; part‑whole; make‑ten
- Relations: add_1, add_2, add_k, remove_1, sum_of, makes_ten_with
- Data: tiny word problems; part‑whole decompositions
- Triples: ["3","add_2","5"], ["5","remove_1","4"], ["2+3","sum_of","5"], ["7","makes_ten_with","3"]
- Analogies: Add‑k families; Make‑ten ([["7","10"],["6","?"]] → "4")
- Heads/losses: keep DistMult; analogy offsets for each add‑k; contrastive ties “2+3” ↔ “5”
- Metrics: link prediction for add/remove; generalize to unseen k
- Status: Partial
  - Done: add_2, makes_ten_with (EpisodeGenerator)
  - TODO: add_1/add_k generalization, remove_1, sum_of, expression ↔ result contrastive

Phase 6 — Place value & number line
- Goals: tens/ones composition; ordering; approximate magnitude
- Relations: has_tens, has_ones, less_than, greater_than, distance_to
- Data: decompose 2‑digit numbers; near‑neighbors on a number line
- Triples: ["42","has_tens","4"], ["42","has_ones","2"], ["37","less_than","41"]
- Analogies: Place value patterns ([["12","1_ten_2_ones"],["34","3_tens_4_ones"]]); Magnitude ([["4","<","7"],["40","<","70"]])
- Heads/losses: strong relation loss; analogy for place‑value patterns
- Metrics: compose/decompose accuracy; ordering consistency
- Status: Partial
  - Done: has_tens, has_ones (EpisodeGenerator)
  - TODO: less_than/greater_than/distance_to; number line sampling/metrics

Phase 7 — Word problems & mixed domains
- Goals: apply numbers to time/money/lengths; child‑simple
- Relations: reuse earlier + cost_of, length_of, time_of_day
- Data: 1–3 sentence problems with explicit triples + targets
- Triples: ["pencil","cost_of","2"], ["rope","length_of","5"]
- Analogies: unit rate starter ([["2_rupees","buy","pencil"],["4_rupees","buy","?"]] → "2 pencils")
- Heads/losses: add task head for numeric answers; keep core heads
- Metrics: extraction → reasoning accuracy pipeline
- Status: TODO

How to synthesize the data (LLM prompts)
- Generator prompt: implemented in LLMTeacher.build_generator_prompt(phase, rel_set, seeds)
- Critic/validator prompt: implemented in LLMTeacher.build_validator_prompt()
- Programmatic checks: LLMTeacher.validate_numbers (successor/pred/make‑ten/place‑value) and validate_taxonomy (cycle checks). Extendable to spatial/temporal.
- Status: Partial (more validators for other phases TBD)

Numeric “gold” atoms to anchor training
- Implemented: EpisodeGenerator.numeric_gold_atoms() returns a tiny fixed set of triples/analogies/equivalences aligned with this doc.
- Status: Partial (available; to be mixed into training batches explicitly)

Mapping to architecture
- Backbone + VQ: present (TinyBackbone + EMA VQ)
- DistMult head: present (successor/add_2/has_tens/ones/makes_ten with string→id mapping)
- AnalogyProjector: present (+k/make‑ten offsets)
- Auxiliary heads:
  - match: TODO (number word ↔ digit ↔ roman)
  - contrastive (count‑aligned views; expr↔result): TODO
- Status: Partial

Loss & schedule nudges by phase
- Suggested schedule recorded here; implementation TBD in the main training loop
- Status: TODO

Episode shapes (text‑only is fine)
- Phase 3 (pre‑number): episode with context tokens and views like “two apples”, “two cars”, “apple apple”
- Phase 5 (ops): views like “2 + 3”, “five apples after adding two”, “three removed from six”; positives: expression ↔ result
- Status: TODO (generators for these text views)

Checkpoints & gates for progression
- Code reuse threshold, analogy offset cosine, add/remove Hits@10, place value accuracy
- Status: TODO (add probes and gates)

Integration notes
- LLMTeacher integrated into training via --use_llm_teacher and --llm_model flags. When enabled, numbers KG triples are sourced from the LLM (or validated synthetic fallback) and rendered through EpisodeGenerator for consistency.
- curriculum helpers: concept_learner/data/curriculum.py contains PHASES, SEEDS, and per‑phase prompt builder.

