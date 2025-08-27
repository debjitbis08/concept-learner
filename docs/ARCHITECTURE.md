Concept Learner: Numeric-Aware Reasoning (HLD)

Overview
- Tokenizer: numeric-aware per-digit tokenization with sign preservation and optional digit-position channel.
- Encoder: Transformer encoder with sinusoidal PE, modulo-10 learned PE, and a small local window mixer to help carry/borrow.
- VQ layer: Residual VQ with parallel + serial codebooks; exposes both a projected z_q and a richer concatenated z_q_all fed to reasoning.
- Reasoner: Multi-step controller with discrete operator selection, STOP head with cumulative halting, and broadcast back to tokens. Typed state includes mask, scalar val, boolean flag, optional pointer span, and a small environment (stack) buffer.
- Operators: Filter, Count, Add, Compare, plus integer-locked add (soft projection) and span-aware variants.
- Decoder: Token and sequence heads; auxiliary numeric head v(x) for ranking/regression.

Tokenizer: numeric awareness
- HFTokenizerWrapper encodes per-digit (and sign) tokens via a pre-tokenization pass.
- encode() also emits optional per-token digit_pos indices (ones=0, tens=1, …) to enable downstream digit-position embeddings.

Scalar numeric channel v(x)
- Model maintains a small MLP head to predict a scalar magnitude from pooled digit states. Used for:
  - Training: ranking loss (hinge) on compare pairs; optional numeric regression on log1p(n).
  - Inference/Reasoning: available as a stable magnitude signal.

Reasoner typed state and control
- Typed state: {mask, val, boolean, ptr_start, ptr_end, env[4]} enabling span selection and small carry/stack memory.
- Controller: softmax over ops with sparsity penalty; cumulative halting with overrun penalty; hard selection via Gumbel-Softmax in training, argmax in eval.

Operators
- OpFilterMLP/OpCount: legacy ops (mask/val updates).
- OpFilterSpan/OpCountSpan: operate only within pointer spans if set.
- OpAdd: generic add; OpAddInt: integer-locked via soft projection of k onto {−10, −1, +1, +10}.
- OpCompareGeneric: unified comparator for gt/lt/eq.

Training/eval contract (frozen sampler mix)
- Each batch composes compare tasks with fixed ratios:
  - 60% in-dist, 20% boundary, 15% range-OOD, 5% counterfactual.
  - Boundary: |a−b| ∈ {0,1} with carry/borrow cases (…9+1, …0−1) and small negatives (−1,0,+1).
  - Range-OOD: sampled from a held-out numeric range with width balancing.
  - Optional hard-negative mining: keep the hardest 25% (smallest margin) in boundary/OOD.
- Auxiliary losses (light weights):
  - Ranking loss (λ≈0.1), numeric regression (λ≈0.05), boundary-weighted BCE (≥1.3×), OpAdd integer regularizer (≈1e−3), controller sparsity+halt penalties (≈1e−3 each).
- Dynamic reweighting:
  - If boundary accuracy < target, increase λ_bd by 1.1× (cap 2×). Same for range-OOD.
- Calibration: temperature scaling at eval; optional per-split temperatures (extendable).

Acceptance metrics
- Accuracy vs digits (2→8): no post-train cliff; unseen widths ≥ 0.70.
- Carry/Borrow sets: ≥ 0.80; Equals ≥ 0.90.
- Range-OOD: ≥ 0.70; Boundary-OOD stable (±2 pts) over last 20% steps.
- Calibration: ECE per split ≤ 3% after temperature scaling.

