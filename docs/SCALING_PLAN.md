Title: Scaling the current CLModel to graduate-student-level mathematical capability

Scope
- This document describes how to scale the existing architecture in this repository (Encoder → RVQ bottleneck → conditioning → multi-step Reasoner → Decoder) to reach graduate-student-like capability in mathematics, without breaking current APIs or tests.

High-level architecture (kept stable)
- Keep CLModel forward contract and shapes unchanged:
  - Inputs: ids (B,T), mask (B,T)
  - Outputs: logits_tok (B,T,C), logits_seq (B,C), vq_loss (scalar), indices (list per quantizer), stop_logits (B,steps), action_logits (B,steps,num_ops)
- Internals remain modular: Encoder, ResidualVQLayer, conditioning module (FiLM or replacement), ReasonerV2 with typed state and operators, UnifiedDecoder.

Tokenizer and input handling
- Upgrade tokenizer to be math-aware: include LaTeX tokens, Unicode math symbols, and common algebraic patterns.
- Optional LaTeX parsing: parse to an AST and serialize deterministically to tokens; fall back to raw text if parsing fails.
- Add retrieval hooks: prepend retrieved definitions, lemmas, and relevant examples to the input, preserving the EncodeOutput interface.

Encoder (replace TinyEncoder with a scalable backbone)
- Long-context transformer or MoE transformer with the same forward interface returning pooled h and per-token H.
- Target ranges: d_model 1k–4k, 48–96 layers, context 32k–128k; for MoE, 64–256 experts with 2–8 active per token.
- Use rotary or ALiBi positional encodings, FlashAttention, activation checkpointing, and efficient parallelism.
- Add retrieval cross-attention blocks to fuse external context (retrieved lemmas, prior work).

Residual vector quantization (concept inventory)
- Increase rvq_dim, codebook_size (e.g., 4096–32768), and num_quantizers (4–8+).
- Use true quantization with EMA updates, dead-code recycling, and diversity/perplexity regularizers.
- Expose discrete indices as concept tokens that the Reasoner can attend to; log usage for interpretability.

Conditioning module
- Replace or augment FiLM with cross-attention over a small set of concept slots derived from z_q.
- Provide gated residual adapters so conditioning strength can be controlled; keep the same call signature and output shapes.

Reasoner (planner with tools and verification)
- Expand the typed state beyond mask/val/boolean to include:
  - current subgoal, stack of subgoals, assumptions, scratch memory for expressions and equations.
- Grow the operator library:
  - Symbolic operators: simplify, factor, expand, differentiate, integrate, solve (linear and nonlinear), substitution, induction step, case split, inequality manipulation, rewrite by rule.
  - Set and logic operators: union, intersection, subset tests, contradiction, contraposition.
  - Numeric operators: high-precision evaluation, interval arithmetic, root finding.
  - Proof control: introduce lemma, apply theorem, instantiate schema, discharge goal.
  - Tool calls: SymPy or Sage for CAS, Z3 or Vampire for SMT/ATP, Lean or Coq bridge for formal proof steps.
- Action policy becomes a distribution over (operator, arguments, tool). Add argument heads to pick spans/tokens/constants.
- Add a light search procedure (beam width 2–8, depth 5–20) with pruning.
- Integrate verifiers in the loop (CAS equivalence, unit tests, proof assistant checking). Use STOP head to terminate.
- Maintain current outputs (H_reasoned, s_final, stop_logits, action_logits) for compatibility, optionally returning extra logs behind flags.

UnifiedDecoder (multi-task outputs)
- Keep current token and sequence classification heads.
- Optionally add heads for:
  - rationale generation (process supervision),
  - program or proof emission (Lean, Coq, or SymPy code) when requested.
- Defaults remain unchanged to preserve existing behavior and tests.

Training and supervision
- Pretraining data: math textbooks and notes, arXiv math, code (Python/Julia/C++), CAS sessions, formal libraries (Lean mathlib, Coq, Isabelle).
- Supervised fine-tuning:
  - Datasets with verified endpoints: theorem statements with proof objects, computational problems with unit tests, modeling problems with code and data.
  - Emphasize spec-to-proof and spec-to-program tasks.
- Auxiliary losses:
  - VQ commitment and diversity, STOP and action policy, argument selection, rationale generation.
- Verifier-guided reinforcement learning:
  - Propose steps, verify with tools, assign rewards, update policy and backbone; use rejection sampling and data aggregation with verified traces.
  - A process reward model scores intermediate steps for logical locality and soundness.
- Curriculum and synthesis:
  - Progress from algebra and calculus to topology, measure theory, algebraic geometry, analysis, PDEs, optimization, probability, and numerical analysis.
  - Self-play style synthesis: mutate statements, ask for proofs or counterexamples, check formally, and add to training.

Infrastructure and efficiency
- Mixed precision (BF16 or FP8 where applicable), accurate norms, activation checkpointing.
- Distributed strategies (FSDP, ZeRO), tensor and pipeline parallelism.
- Long-context attention with paged KV cache.
- Retrieval store over curated textbooks, lecture notes, and formal libraries.

Concrete target configurations
- Minimal strong configuration:
  - 30–70B active-parameter MoE encoder, 128k context, robust retrieval.
  - Expanded operator set with SymPy, Z3, and Lean integration; process reward model; small beam search.
- Ambitious configuration:
  - 100–200B dense or 200B–1T total MoE with 20B–60B active parameters.
  - Strong verifier RL and full formal proof toolchain integration.

Evaluation milestones
- Unit-shape and gradient tests continue to pass (current repo tests).
- Accuracy on graduate-level problem sets (informal) and acceptance rate of formal proofs (Lean or Coq) for targeted theorems.
- Correctness on computational tasks with hidden unit tests.
- Robustness under tool ablations and distribution shifts.

Migration plan tied to current code
- Keep CLModel API stable so existing scripts and tests run unchanged.
- Swap implementations behind the same interfaces:
  - TinyEncoder → long-context or MoE Transformer with same outputs (h, H).
  - ResidualVQLayer → larger, real RVQ; indices used as concept tokens.
  - FiLM → cross-attention conditioning module with same call signature.
  - ReasonerV2 → planner with richer state, operators, and verifier tools; same external outputs.
  - UnifiedDecoder → additional optional heads; defaults unchanged.
- Add a Tools module for CAS, SMT/ATP, and proof assistants, and a Verifier module that normalizes and checks outputs from tools.

Rationale
- The model must not only generate text but also plan, compute, prove, and verify. Tight integration with symbolic tools and formal verifiers, combined with light search and process supervision, moves the system toward graduate-student-level reliability and breadth.

