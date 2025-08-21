Concept Learner – Next Steps (GPU Run Plan + Modules)

This document summarizes the plan and code skeletons to continue training and expand the system on a GPU machine.

Status snapshot (after latest patches)

- Data: PAD+mask fixed; hard negatives; A≠B/C≠D in analogies; curriculum controls; optional canonicalization to remove remap difficulty early.
- Model: Masked Transformer; EMA VQ with anti‑collapse, dead‑code reinit, usage entropy; projection heads with stop‑grad; relation projector with scaling.
- Training: Param‑group AdamW; temperature anneal; scheduled weights; EMA checkpoints and resume; watchdog rollback+recovery; codebook stats and EMA in‑batch analogy accuracy (an_acc).

Run recommendations on GPU

- Keep views trivial much longer and ramp slower:
  - 0–6k: identical remap, same base (easy_same_remap=1.0; change_base_prob=0.0); parity‑only analogies.
  - 6–10k: easy_same_remap=0.5; change_base_prob=0.0; still parity‑only.
  - 10–14k: easy_same_remap=0.2; change_base_prob=0.5; add mod3.
  - 14k+: easy_same_remap=0.0; change_base_prob=1.0; add next_in_sequence.
- Keep EpisodeConfig.canonicalize=True early to remove remap noise entirely; disable later (e.g., ≥10k) to re‑introduce remaps once stable.
- Early weights: w_rel≈2.0, w_an≈1.5, w_c≈0.1. In recovery: w_rel≥3.0, w_an≥2.0, w_c→0, w_mdl→0, vq_usage≥0.3; InfoNCE on z_q, clip=0.2.
- Target metrics (green lights):
  - Codebook: ppx ≥ 10–16 (K=32), uniq ≥ 12/32; watchdog rarely triggers.
  - Relations (CE): trending well below ln(batch) and decreasing.
  - Analogy loss decreasing; EMA in‑batch analogy acc rises above random (~1/N on N‑way probe).

If stuck

- Codebook ppx→1.0 or uniq→1/32: watchdog rollback and recovery; extend recovery window; keep canonicalization on; reduce encoder LR (optional).
- Relations flat: keep parity‑only longer; further boost w_rel/w_an; consider upgrading relation head to RESCAL.
- Contrastive hard: keep easy views; lower w_c (0 in recovery); use z_q for InfoNCE during recovery.

Conditional invariance (later)

- Option A (current): canonicalize digits early for stability; turn off later.
- Option B: add BOS tokens for base and remap descriptor; enforce invariance conditional on context for stronger generalization.

Modules to add next

1. LLM‑based training module (teacher + validator)

   - Generates multi‑domain episodes in a single JSON schema; runs validators before training.
   - See concept_learner/teacher/llm_teacher.py (skeleton added).

2. Human interaction module (playground + feedback)

   - Simple Gradio app to query analogies and capture feedback.
   - See apps/playground.py and concept_learner/api.py (skeletons added).

3. Checkpoint manager

   - Save/load latest; keep last N checkpoints.
   - See utils/checkpoints.py (implemented).

4. Continuous learning (day / mini‑nap / sleep)

   - Hooks for EMA, periodic eval, and light consolidation with replay + EWC.
   - See utils/ema.py and the pseudo‑hooks in this doc to wire into train.py later.

5. Long‑run schedule (50k–150k)

- 0–20k: stabilize VQ (K=32), canonicalization ON, trivial views; parity only; impostor analogies; watchdog active.
- 20–60k: gradually disable canonicalization; ramp remaps/base per schedule; add mod3; keep hard negatives.
- 60–120k: add more domains (numbers + words + rhythm), analogy‑heavy batches; consider conditional invariance tokens.
- 120k+: optional capacity bump (K→64) or head upgrades if probes plateau.

Wiring checklist

- [x] Checkpointing: periodic saves + latest + best_rel/best_an (EMA included); resume supported.
- [x] Watchdog + recovery guardrails in training.
- [x] an_acc logging: EMA in‑batch probe, printed with random baseline.
- [x] Playground/API scaffolding; Colab support (--share).
- [ ] --use_llm_teacher flag to swap data source.
- [x] Continuous eval/sleep hooks with EMA and replay.
- [ ] Expanded logs: analogy accuracy by relation, triple AUC, probe accuracy for mod as enabled.

Playground quick‑start

- Local: poetry install gradio; python apps/playground.py --ckpt checkpoints/latest.pt --device auto
- Colab: pip install gradio torch; python apps/playground.py --ckpt checkpoints/latest.pt --device auto --share
- ConceptAPI.complete_analogy: implemented for the toy numbers domain (EMA applied if present).

New tasks (roadmap additions)

# (1) Scaling to many domains with overlap

## A. Condition the model on domain—but keep a shared core

- Domain token (or one‑hot domain id) prepended to the sequence.
- Lightweight adapters/LoRA/FiLM in the backbone keyed by domain.
- Keep the global transformer weights + global relation space mostly shared.

## B. “Shared + Private” vector‑quantization

- Two‑stage (residual) VQ:
  - Stage‑1 (Global codebook): captures universal concepts; frozen after warm‑up.
  - Stage‑2 (Domain codebook): small, per‑domain residual codes for domain quirks.
- Or Mixture of Codebooks: one global + K domain codebooks; a tiny gate (from the domain token) picks which codebook(s) to draw from.
- Regularizers to encourage overlap:
  - Cross‑domain contrastive alignment for known equivalent items (e.g., price_finance ↔ price_ecom).
  - Code reuse bonus / entropy penalties so the model prefers global codes when possible.
  - Orthogonality between private codes of different domains to avoid accidental collisions.

## C. Relations: shared where possible, private where needed

- Keep a global DistMult over the shared concept space.
- Add per‑domain relation embeddings/heads only when semantics diverge.
- AnalogyProjector is shared; add a small domain adapter if analogies differ in scale.

## D. Training recipe

- Balanced multi‑task batching across domains.
- EMA usage per domain for VQ; dead‑code reinit can be domain‑aware (reseed from that domain’s minibatch stats).
- Replay + regularization for continual addition of domains (EWC/LwF) so old codes don’t drift.
- Periodic codebook surgery: split overfull codes, merge near‑duplicates, retire dead ones.

## E. Metrics to know it’s working

- Code sharing ratio: fraction of items that map to Stage‑1 codes across domains.
- Mutual information between code assignments and domain (lower is more shared).
- Transfer delta: train on A, test B (before/after light domain‑adapter finetune).
- Analogy consistency: cosine similarity of relation vectors for the “same” relation across domains.

---

# (2) Best ways to demo strengths

Quick, compelling demos (build with Gradio/Streamlit):

1. Concept Browser

   - Show each codebook entry with its top nearest items from multiple domains.
   - Add filters: “Global codes only”, “Private codes only”.
   - Metric badges: usage %, purity, cross‑domain coverage.

2. Triple Scorer Playground

   - Inputs: (subject, relation, object) with auto‑complete from any domain.
   - Show DistMult score, nearest corruptions (replace subject/object), and why (nearest concept codes).

3. Analogy Board

   - Users type A:B :: C:? across domains (e.g., “Paris:France :: Tokyo: ?”; “Python:Guido :: Linux: ?”).
   - Visualize the relation vector arrow in the AnalogyProjector space and candidates ranked.

4. Cross‑domain Transfer Mini‑Study

   - Train on Domain‑A only → evaluate on Domain‑B.
   - Then add 50 labeled examples from Domain‑B with just adapters + Stage‑2 codes; show the jump.
   - Plot time‑to‑adapt vs. accuracy; this sells the “shared symbolic core”.

5. Compression/Efficiency Card
   - Compare storage/latency with & without VQ (bits per concept, QPS).
   - Great for stakeholders who care about deployment.

Suggested benchmark set (small, convincing):

- Knowledge graphs: WN18RR (lexical), FB15k‑237 (facts), + one niche KG (e.g., a tiny biomedical or e‑commerce KG).
- Analogies: Google Analogy Test Set or BATS (use a filtered subset).
- Concept matching: duplicate/canonicalization pairs from two domains (e.g., product names vs. finance tickers).

What to plot/show

- UMAP/t‑SNE of code vectors colored by domain (overlap pops visually).
- Chord diagram: which domains reuse which global codes.
- Relation quiver plot: arrows for the same relation across domains (parallel = good).

A tidy two‑week demo plan

- Week 1: Implement domain token + adapters; switch to 2‑stage VQ; add code‑sharing metrics.
- Week 2: Ship the three UIs (Concept Browser, Triple Scorer, Analogy Board) + a 1‑page transfer study and a small efficiency card.
