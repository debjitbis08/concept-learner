Architecture overview (HLD)

Purpose

Small, testable, text‑only concept learner. A tiny Transformer encodes input tokens; a compact factorized VQ bottleneck provides discrete features; a lightweight reasoning loop updates a typed state with a few general operators; a unified decoder produces token‑level and pooled sequence logits.


Key components

- Tokenizer (tokenizer.py)
  - HFTokenizerWrapper uses a Hugging Face tokenizer when available (bert-base-cased)
  - Falls back to a tiny whitespace tokenizer with [CLS]/[SEP]/[PAD] if downloads aren’t possible
  - Exposes ids and attention mask with fixed max length

- Encoder (encoder.py)
  - TinyEncoder: 2‑layer Transformer encoder (batch_first, norm_first)
  - Returns
    - h: pooled vector (uses [CLS] if present, else masked mean)
    - H: per‑token states (B, T, d)

- Vector‑quantization bottleneck (vq_layer.py)
  - ResidualVQLayer implements a factorized VQ composed of:
    - Parallel heads: multiple linear projections of the pooled encoder state, each quantized independently with a small codebook (vector_quantize_pytorch.VectorQuantize).
    - Optional serial refiners: small MLP stages that operate on the concatenated parallel embeddings; each refiner has its own quantizer. This is not classic residual VQ; there is no numeric residual pass‑through between stages.
  - Returns a continuous embedding z_q (projection of the concatenated quantized features back to model dim), a list of index tensors (one per quantizer stage: all parallel heads first, then serial refiners), and a summed VQ loss.
  - Includes optional regularizers and training knobs:
    - Pre‑VQ Gaussian noise
    - Projection orthogonality penalty across parallel heads
    - Entropy bonus encouraging codebook usage
  - Accepts selected VectorQuantize kwargs (e.g., use_cosine_sim, kmeans_init) which are forwarded to the underlying quantizers.

- Conditioning (model.py)
  - FiLM module applies z_q to per‑token H: H_cond = H * (1 + gamma(z_q)) + beta(z_q)
  - Keeps the interface simple and end‑to‑end trainable

- Reasoning (reasoning_v2.py, reasoning_ops.py)
  - Typed state per batch:
    - mask: selection over tokens (float 0..1)
    - val: scalar (B, 1)
    - boolean: scalar (B, 1)
  - Operators (generic and learnable):
    - Filter: per‑token scorer in [0,1]; mask’ = min(mask, score)
    - Count: DeepSets aggregator to produce val = sum(phi(item) * mask)
    - Add: val’ = beta * val + alpha * k (k may be predicted from z_q)
    - Compare: boolean’ = sigmoid(w · f(val − k) + b) with non‑negative basis
  - Controller (ReasonerV2):
    - For up to max_steps:
      - Compute action logits over operators from concat(state, z_q)
      - Soft mixture of operator outputs updates (mask, val, boolean)
      - STOP head predicts halting at each step (used in training scripts)
    - Broadcast final state back to tokens via a small MLP and residual add

- Decoder (decoder.py)
  - token_head: per‑token logits (B, T, C)
  - seq_head: pooled sequence logits (B, C)
  - Training scripts primarily use seq_head for tasks (classification/regression‑as‑classification)


Data flow (forward)

1) Tokenize: ids, mask = tokenizer.encode(text)
2) Encode: h, H = TinyEncoder(ids, mask)
3) Quantize: z_q, indices, vq_loss = ResidualVQLayer(h)
4) Condition tokens with FiLM: H_cond = FiLM(H, z_q)
5) Reasoning: H_reasoned, s_final, stop_logits, action_logits = ReasonerV2(H_cond, z_q, mask)
6) Decode: logits_tok, logits_seq = UnifiedDecoder(H_reasoned, mask)


Training signals (as used in repo)

- Cross‑entropy on the pooled sequence head (logits_seq) for tasks in scripts/train.py, scripts/train_count.py, scripts/train_episodes.py
- Optional token‑level loss in tests/trainer for smoke checks
- VQ regularizer: vq_loss (weighted)
- STOP head: binary cross‑entropy on stop_logits in trainer/scripts using synthetic or zero targets


Constraints and non‑goals (current code)

- Factorized VQ with parallel heads and optional serial refiners; not a hierarchical bank of subject/domain/operator codebooks
- No explicit schema‑based MoE registry; operator mixing is softmax‑based
- No external tools/retrieval; text‑only inputs
- Single‑token classification decoder (no autoregressive decoding)


Extensibility guide

- New operators: follow the OpBase(state, z_q) -> (mask, val, boolean) contract in reasoning_ops.py
- Larger encoder: swap TinyEncoder for a deeper Transformer; keep output (h, H)
- Different quantizer: replace ResidualVQLayer with your module if it returns (z_q, indices, loss)
- Multi‑token outputs: replace decoder.seq_head with an autoregressive head; training code will need adjustment


References and entry points

- Model wiring: src/concept_learner/model.py (Encoder → RVQ → FiLM → ReasonerV2 → Decoder)
- Operators and typed state: src/concept_learner/reasoning_ops.py
- Controller: src/concept_learner/reasoning_v2.py
- Scripts: scripts/train.py, scripts/train_count.py, scripts/train_episodes.py
- Tests: tests/concept_learner/*
