Concept Learner – Next Steps (GPU Run Plan + Modules)

This document summarizes the plan and code skeletons to continue training and expand the system on a GPU machine.

Status snapshot (after latest patches)
- Data: PAD+mask fixed, hard negatives for triples, A≠B/C≠D in analogies, curriculum controls for views and analogies.
- Model: Masked Transformer, EMA VQ with anti‑collapse, usage entropy, projection heads with stop‑grad, relation projector with scaling.
- Training: Param‑group AdamW, temperature anneal, scheduled weights, logging codebook stats.

Run recommendations on GPU
- Start simple and ramp difficulty.
  - Steps 0–4k: identical views (same base+remap), parity‑only analogies.
  - 4k–6k: same base, remap varies 50% of time; still parity‑only analogies.
  - 6k–8k: introduce base changes (20%); add mod3 relation.
  - 8k+: base changes 100%; add next_in_sequence relation.
- Target metrics (green lights):
  - Codebook: perplexity ≥ 10–16 (K=32), uniq ≥ 12/32 early; no collapse.
  - Relations: hinge < 1.0 and trending down; analogy loss decreasing.
  - Analogy accuracy (in‑batch parity) > 0.2–0.4 once stabilized.

If stuck
- Codebook ppx→1.0 or uniq→1/32: increase usage loss, reinit dead codes (already implemented), reduce commitment beta.
- Relations flat: temporarily boost loss_rel/analogy, keep parity‑only longer, or upgrade relation head to RESCAL.
- Contrastive hard: keep easy_same_remap longer; lower loss_contrastive early; use z (pre‑quantized) for InfoNCE.

Modules to add next
1) LLM‑based training module (teacher + validator)
   - Generates multi‑domain episodes in a single JSON schema; runs validators before training.
   - See concept_learner/teacher/llm_teacher.py (skeleton added).

2) Human interaction module (playground + feedback)
   - Simple Gradio app to query analogies and capture feedback.
   - See apps/playground.py and concept_learner/api.py (skeletons added).

3) Checkpoint manager
   - Save/load latest; keep last N checkpoints.
   - See utils/checkpoints.py (implemented).

4) Continuous learning (day / mini‑nap / sleep)
   - Hooks for EMA, periodic eval, and light consolidation with replay + EWC.
   - See utils/ema.py and the pseudo‑hooks in this doc to wire into train.py later.

5) Long‑run schedule (50k–150k)
- 0–20k: stabilize VQ (K=32), remap‑only views, parity only, impostor analogies.
- 20–60k: introduce base changes, add mod3, hard negatives.
- 60–120k: add more domains (numbers + words + rhythm), analogy‑heavy batches.
- 120k+: optional capacity bump (K→64) if probes plateau.

Wiring checklist
- [ ] Add --use_llm_teacher to train.py to swap EpisodeGenerator with LLMTeacher.
- [ ] Add ConceptAPI usage in apps/playground.py with a saved checkpoint.
- [ ] Call CheckpointManager.save every N steps and at curriculum transitions.
- [ ] Add EMA wrapper, periodic eval (mini‑nap), and replay‑based sleep consolidation.
- [ ] Expand logs: analogy accuracy by relation, triple AUC, probe accuracy for parity/mod.

