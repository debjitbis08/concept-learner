# Scaling the current CLModel to graduate-student-level mathematical capability

## Scope

- This document describes how to scale the existing architecture in this repository (Encoder → RVQ bottleneck → conditioning → multi-step Reasoner → Decoder) to reach graduate-student-like capability in mathematics, **without breaking current APIs or tests**.

## High-level architecture (kept stable)

- Keep `CLModel.forward` contract and shapes unchanged:

  - **Inputs:** `ids (B,T)`, `mask (B,T)`
  - **Outputs:** `logits_tok (B,T,C)`, `logits_seq (B,C)`, `vq_loss (scalar)`, `indices (list per quantizer)`, `stop_logits (B,steps)`, `action_logits (B,steps,num_ops)`

- Internals remain modular: `Encoder`, `ResidualVQLayer`, conditioning module (FiLM or replacement), `ReasonerV2` with typed state and operators, `UnifiedDecoder`.

## Tokenizer and input handling

- Upgrade tokenizer to be **math-aware**: include LaTeX tokens, Unicode math symbols, and common algebraic patterns.
- Optional LaTeX parsing: parse to an AST and deterministically serialize to tokens; **fallback** to raw text if parsing fails.
- **Retrieval hooks:** prepend retrieved definitions, lemmas, and relevant examples to the input, preserving the `EncodeOutput` interface.

## Encoder (replace TinyEncoder with a scalable backbone)

- Long-context Transformer or MoE Transformer with the **same forward interface** returning pooled `h` and per-token `H`.
- **Two-speed path:** a small **Skim Encoder** runs first (drives early routing/early-exit). Invoke **Heavy Encoder** (long-context / optional MoE, cross-attn to retrieval) only when needed; **reuse KV cache**.
- Targets: `d_model` 1k–4k, 48–96 layers, context 32k–128k; for MoE, 64–256 experts with 2–8 active per token.
- Rotary/ALiBi positions, FlashAttention, activation checkpointing, efficient parallelism.
- Add **retrieval cross-attention** blocks to fuse external context (retrieved lemmas, prior work).

## Residual vector quantization (concept inventory)

- Increase `rvq_dim`, `codebook_size` (e.g., 4096–32768), and `num_quantizers` (4–8+).
- True quantization with **EMA updates**, dead-code recycling, and diversity/perplexity regularizers.
- **Append-only ids (no reindexing)** for stability; support **code split/merge** via a remap table (old ids remain valid for inference).
- **Lazy VQ:** compute **VQ-L1** always; compute deeper VQs **on demand** (per active branch need).
- Expose **discrete indices as concept tokens** and `z_q` embeddings; log code usage/perplexity for interpretability.
- **Training strategy:** staged warm-up (L1→Lk) with earlier codebooks temporarily frozen, then **joint fine-tune** with small LRs. Add **auxiliary index-prediction heads** on the encoder to keep gradients flowing.

## Conditioning module

- Replace/augment FiLM with **cross-attention over K concept slots** derived from `z_q`.
- Provide **gated residual adapters** so conditioning strength is controllable; keep the **same call signature and output shapes**.

## Adaptive complexity routing (entry + step)

- **Entry Router (slow LR):** picks initial branch $b_0$ from `[e1; pooled h; prompt stats]`. KL-to-old-router penalty on legacy data during new-branch training.
- **Step Router (per reasoning step):** may **promote/demote by ±1** using:

  - **Lookahead confidence:** top-k mass, entropy, margin, invalid-op probability.
  - **Shadow one-step outlook** $L_b$: cheap typed-state rollout (no tools).
  - **Value gap:** $V_{b+1}(s) - V_b(s) - \lambda_{cost}$, calibrated per branch; optional tool-free value head $V_b^{nf}$.
  - **Hysteresis window & budgets:** prevent ping-pong; respect latency/tool budgets.

- Discrete decisions via **Gumbel-Straight-Through**; add an **aux branch-prediction loss** for stability.
- **Branch Context Stack:** `PUSH_BRANCH(b+1)` on promotion for a harder subgoal; `POP_BRANCH()` on completion. At any time, CALL lower-branch ops as read-only subroutines.

## Reasoner (planner with tools and verification)

- Typed state grows beyond `mask/val/bool` to include: current subgoal, **stack of subgoals** (for branch stack), assumptions, scratch memory for expressions/equations.
- **Operator library** expands:

  - _Symbolic:_ simplify, factor, expand, differentiate, integrate, solve (linear/nonlinear), substitution, induction step, case split, inequality manipulation, rewrite by rule.
  - _Set/logic:_ union, intersection, subset tests, contradiction, contraposition.
  - _Numeric:_ high-precision evaluation, interval arithmetic, root finding.
  - _Proof control:_ introduce lemma, apply theorem, instantiate schema, discharge goal.
  - _CALL older branches:_ read-only, versioned APIs (e.g., `B1.Add/Sub/Compare`).
  - _Tool ops (optional):_ CAS (SymPy/Sage), SMT/ATP (Z3/Vampire), Proof (Lean/Coq). **No-grad on tool calls**; the controller learns **when** to call them via verifier rewards.

- **Dual-path ops for tool independence:** maintain learned implementations alongside tool-backed ones; use **tool-dropout** during training and **confidence gating** at inference.
- **Action policy** becomes a distribution over (operator, arguments, tool flag). Add **argument heads** to pick spans/tokens/constants.
- **Light search:** beam 2–8, depth 5–20 with **early-exit schemas** and **verifier-pruned** expansions.
- **Verifiers in the loop:** CAS equivalence, unit tests, proof assistant checking. STOP head to terminate.
- Maintain current outputs (`H_reasoned`, `s_final`, `stop_logits`, `action_logits`) for compatibility; optionally return extra logs behind a `debug` flag.

## UnifiedDecoder (multi-task outputs)

- Keep current token and sequence classification heads.
- Optional heads:

  - Rationale generation (process supervision),
  - Program/proof emission (Lean/Coq/SymPy) **when requested**.

- Defaults remain unchanged to preserve existing behavior and tests. Main decoder weights can be **frozen after v1** with tiny **per-branch adapters** (LoRA/FiLM).

## Training and supervision

- **Pretraining data:** math textbooks/notes, arXiv math, code (Python/Julia/C++), CAS sessions, formal libraries (Lean mathlib, Coq, Isabelle).
- **Supervised fine-tuning:** datasets with **verified endpoints** (proof objects, hidden unit tests).

  - Emphasize **spec→proof** and **spec→program** tasks; mine **verified traces** via tools, then **distill** into learned ops.

- **Auxiliary losses:** VQ commitment & diversity; STOP; action policy; argument selection; **branch prediction**; **invalid-op**; optional rationale loss.
- **Verifier-guided RL:** propose steps, verify with tools, assign rewards, update **policy (and optionally backbone)**; use rejection sampling and data aggregation with verified traces. Add a **process reward model** for logical locality/soundness.
- **Tool independence schedule:** increase **no-tool episodes** over time; track **tool-call rate** and push it down.
- **Curriculum & synthesis:** algebra/calculus → topology, measure theory, algebraic geometry, analysis, PDEs, optimization, probability, numerical analysis. Self-play synthesis: mutate statements, seek proofs/counterexamples, check formally, add to training.

## Infrastructure and efficiency

- Mixed precision (BF16 or FP8 where applicable), accurate norms, activation checkpointing.
- Distributed strategies (FSDP, ZeRO), tensor and pipeline parallelism.
- Long-context attention with paged KV cache; **two-speed encoder** path.
- **Latency controls:** early-exit schemas, **lazy VQ** (on-demand deeper levels), **tool budgets**, **batched & cached** tool calls (canonicalize→hash→memo).
- Retrieval store over curated textbooks, lecture notes, and formal libraries.

## Concrete target configurations

- **Minimal strong configuration:**

  - 30–70B active-parameter MoE (or a strong dense long-context encoder), 128k context, robust retrieval.
  - Expanded operator set with **SymPy, Z3, Lean** integration; process reward model; small beam search; **adaptive routing enabled**.

- **Ambitious configuration:**

  - 100–200B dense or 200B–1T total MoE with 20B–60B active parameters.
  - Strong verifier RL and full formal proof toolchain integration; deeper operator set; larger beam/depth with strict budgets.

## Evaluation milestones

- **API invariants:** unit-shape and gradient tests continue to pass (current repo tests).
- **Math capability:** accuracy on graduate-level problem sets; **acceptance rate** of formal proofs (Lean/Coq) for targeted theorems.
- **Computation correctness:** pass hidden unit tests for program/compute tasks.
- **Routing quality:** promotion/demotion counts, router entropy, **no ping-pong** under hysteresis; success rate after promotions.
- **Quantizer health:** code usage/perplexity, dead-code %, split/merge stability.
- **Tool independence:** **no-tool eval** suite, tool-ablation robustness, **tool-call rate** trend ↓ over time.
- **Latency:** median/95p end-to-end latency, tool-budget hit rate, early-exit frequency.

## Migration plan tied to current code

- Keep `CLModel` API stable so existing scripts and tests run unchanged.
- Swap implementations behind the same interfaces:

  - `TinyEncoder` → long-context or MoE Transformer with same outputs (`h`, `H`), plus **Skim Encoder** in front (internal).
  - `ResidualVQLayer` → larger, real RVQ; indices used as concept tokens; **lazy VQ** wrapper.
  - `FiLM` → cross-attention conditioning module with same call signature.
  - `ReasonerV2` → planner with richer state, **adaptive routing (entry + step)**, operators, and verifier tools; same external outputs.
  - `UnifiedDecoder` → optional rationale/proof heads; defaults unchanged; add per-branch adapters internally.

- Add:

  - **Tools module** for CAS/SMT/ATP/proof assistants (with canonicalizer + cache),
  - **Verifier module** that normalizes & checks tool outputs,
  - **Value heads / invalid-op head / shadow updater** (internal; exposed only via `debug=True`).

## Rationale

- The model must not only generate text but also **plan, compute, prove, and verify**. Tight integration with symbolic tools and formal verifiers, combined with **adaptive routing**, **light search**, **RVQ concept inventory**, and **process supervision**, moves the system toward graduate-student-level **reliability, breadth, and latency discipline**—while keeping the public API and tests stable.

## Architecture

```
                                ┌─────────────────────────────────────────────────────────────┐
[TEXT PROMPT] ─► Tokenizer ────►│ INPUTS (stable): ids (B,T), mask (B,T)                      │
(LaTeX-aware, Unicode math)     └─────────────────────────────────────────────────────────────┘
               │
               ▼
     ┌─────────────────────────────── Optional Retrieval Hook ────────────────────────────────┐
     │ retrieve {defs, lemmas, examples}; concat/prefix to encoder input (shapes unchanged)  │
     └────────────────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
   ┌──────────────────────────────────────── ENCODER (2-speed) ─────────────────────────────────────────┐
   │ Skim Encoder (small, fast) → drives early routing/early-exit                                      │
   │ If non-trivial: Heavy Encoder (long-context / (optional) MoE, cross-attn to retrieval)            │
   │ Returns pooled h, per-token H (same iface as TinyEncoder)                                         │
   └────────────────────────────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
   ┌──────────────────────────────────────── RVQ (concept inventory) ───────────────────────────────────┐
   │ VQ-L1 … VQ-Lk (k=4–8): EMA codebooks, dead-code recycle, commitment+diversity losses              │
   │ • Lazy VQ: L1 always; compute deeper L{2..k} on demand (per-branch need)                          │
   │ • Expose z_q (embeds) + indices (append-only ids; no reindex).                                    │
   └────────────────────────────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
   ┌──────────────────────────────────────── ROUTING (adaptive) ────────────────────────────────────────┐
   │ Entry Router (slow LR): picks initial branch b₀ from [e1; pooled h; prompt stats].                │
   │ Step Router (per step): may promote/demote by ±1 using:                                           │
   │   • Lookahead confidence: top-k mass, entropy, margin, invalid-op prob                            │
   │   • Shadow one-step outlook L_b (cheap typed-state rollout, no tools)                             │
   │   • Value gap: (V_{b+1} − V_b − λ_cost), calibrated per branch                                    │
   │   • Hysteresis window + Budget guards (latency/tool budgets)                                      │
   │ (Gumbel-ST for discretes; KL-to-old-router only on legacy slices)                                 │
   └────────────────────────────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
        ┌────────────────────────────────── BRANCH SELECTOR ──────────────────────────────────┐
        │ Activates exactly ONE sub-tower; maintains Branch Context Stack:                    │
        │   PUSH_BRANCH(b+1) on promotion; POP_BRANCH() on subgoal completion.                │
        └─────────────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
==========================================================================================================
||                                     ACTIVE SUB-TOWER (branch b)                                       ||
||                                                                                                       ||
||   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                                             ||
||   │  VQ-L2       │→→→│  VQ-L3       │→→→│  VQ-L4       │  (append-only codes; branch-local expand)   ||
||   │  Domain/Sub  │   │  Op Family   │   │  Args/Values │                                             ||
||   └──────────────┘   └──────────────┘   └──────────────┘                                             ||
||            │                     │                   │                                               ||
||            └───────────────┬─────┴───────────────────┴───────────────┐                               ||
||                            ▼                                         ▼                               ||
||      ┌────────────────────────────────────────────────────────────────────────────┐                  ||
||      │ CONDITIONING (API-compatible): Cross-attn over K concept slots + gated ADP │                  ||
||      └────────────────────────────────────────────────────────────────────────────┘                  ||
||                            │                                                                         ||
||                            ▼                                                                         ||
||  ┌──────────────────────────────────────── REASONER / PLANNER (ReasonerV2 iface) ──────────────────┐ ||
||  │ Typed State: {mask, val, bool, subgoal stack, scratch exprs/equations, …}                      │ ||
||  │ Heads (public): action_logits (B,steps,num_ops), stop_logits (B,steps)                          │ ||
||  │ Internals: argument pickers (spans/consts), schema tags, simple/complex flags                   │ ||
||  │                                                                                                 │ ||
||  │ Operator Registry:                                                                              │ ||
||  │   • Learned Ops (end-to-end)  • CALL Older Branch Ops (read-only, versioned)                    │ ||
||  │   • Tool Ops (optional, no-grad): CAS(SymPy/Sage), SMT(Z3/Vampire), Proof(Lean/Coq)             │ ||
||  │ Light Search: beam 2–8, depth 5–20; verifier-pruned; Early Exit for simple schemas              │ ||
||  │ Tool Budget + Cache: canonicalize→hash; cap calls/query; batch where possible                   │ ||
||  │                                                                                                 │ ||
||  │ STEP ROUTER LOOP (inside planner):                                                              │ ||
||  │   1) Read policy p(op|state,b); compute top-k mass, entropy, margin, invalid-prob.              │ ||
||  │   2) Shadow lookahead (no tools) → L_b; read V_b(state), V_{b+1}(state).                        │ ||
||  │   3) Promotion score S_prom = f(1−mass_k, H, margin, invalid, V_gap, cost, budget).             │ ||
||  │   4) If S_prom>τ and L_b<τ_lookahead and budgets OK → b←b+1 (PUSH + hysteresis).                │ ||
||  │      Else consider demote if easy (save latency).                                               │ ||
||  │   5) Execute op: prefer learned; CALL lower-branch lib if available; else gated tool (budget).  │ ||
||  │   6) Update typed state; STOP when stop_head fires or goal verified.                             │ ||
||  └────────────────────────────────────────────────────────────────────────────────────────────────┘ ||
||                                                                                                       ||
||      ┌───────────────────────────────── UNIFIED DECODER ──────────────────────────────────┐          ||
||      │ single vocab; main weights frozen after v1; tiny per-branch adapters (LoRA/FiLM)  │          ||
||      │ outputs: logits_tok (B,T,C), logits_seq (B,C)                                      │          ||
||      └────────────────────────────────────────────────────────────────────────────────────┘          ||
==========================================================================================================
               │
               ▼
┌──────────────────────────────────────────────── OUTPUTS (stable) ──────────────────────────────────────┐
│ logits_tok (B,T,C) | logits_seq (B,C) | vq_loss (scalar) | indices (per quantizer; pad -1)            │
│ stop_logits (B,steps) | action_logits (B,steps,num_ops)                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘


Side subsystems (shared; train-time + runtime)
-----------------------------------------------------------------------------------------
┌──────────────────────────────┐  ┌──────────────────────────────┐  ┌───────────────────┐
│ Value Heads per branch       │  │ Invalid-Op Head              │  │ Shadow Updater    │
│ V_b(state), V_b^nf(state)    │  │ predicts op validity masks   │  │ cheap 1-step      │
│ (calibrated; no-tool variant)│  │ (shape/type + verifier errs) │  │ typed rollout      │
└──────────────────────────────┘  └──────────────────────────────┘  └───────────────────┘

┌──────────────────────────────────────┐      ┌─────────────────────────────────────────┐
│ Tool Canonicalizer + Cache           │      │ Trace Store (plans, spans, constants)  │
│ normalize exprs; memoize results     │      │ distill + tool-dropout training        │
└──────────────────────────────────────┘      └─────────────────────────────────────────┘


Training notes (where the updates bite)
---------------------------------------
• RVQ warmup staged (L1→Lk), then joint finetune; aux index prediction from encoder.
• Entry router small LR; step router full LR with verifier rewards + hysteresis.
• Tools are no-grad. Controller learns when to call them; tool-dropout for independence.
• Early-exit + lazy VQ + two-speed encoder keep latency in check.
```
