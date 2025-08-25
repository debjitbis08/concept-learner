Subject-Aware Concept Learner (text-only)

Short version: a tiny Transformer encoder, a small residual VQ bottleneck, a few typed reasoning ops, and a simple unified decoder. It’s all text-only and comes with tests and minimal training scripts.

This README focuses on how to use and extend the code. The original research sketch was broader; the shipped code intentionally keeps the surface area small and practical.

What’s inside (implemented)

- Tokenizer wrapper with offline fallback
  - Uses a Hugging Face tokenizer when available (bert-base-cased)
  - Falls back to a tiny whitespace tokenizer with [CLS]/[SEP]/[PAD] when downloads aren’t possible
- TinyEncoder: 2-layer Transformer encoder that returns
  - h: pooled vector (prefers [CLS] if present)
  - H: per-token states
- ResidualVQLayer: single residual vector-quantization bottleneck
  - Uses vector-quantize-pytorch if installed; otherwise a lightweight pass-through fallback keeps shapes/losses consistent for tests
- ReasonerV2: multi-step reasoning over a typed state with a STOP head
  - Typed state: mask (selection over tokens), val (scalar), boolean (scalar)
  - Operators: Filter (per-token scorer), Count (DeepSets), Add (generic), Compare (generic)
  - Soft mixture over operators per step + broadcast of the final state back to tokens
- UnifiedDecoder: two simple heads
  - token_head for per-token logits (used in tests/demo)
  - seq_head for pooled sequence classification (used by training scripts)
- Training helpers and toy tasks
  - scripts/train.py: quick smoke training on synthetic batches
  - scripts/train_count.py: trains successor / predecessor / between with a tiny hand-rolled vocab
  - scripts/train_episodes.py: natural-language pairs + counting using a Hugging Face tokenizer
- Tests: tokenizer, encoder, VQ layer, model wiring, and a minimal train step

What’s not included (by design for now)

- No hierarchical 4-level VQ; there’s one ResidualVQ bottleneck
- No explicit subject/domain/operator hierarchy or schema-based MoE registry
- No external tools or retrieval plugins
- Decoder is single-token classification (no autoregressive decoder)

Install

Prereqs: Python 3.13+

Option A — pip

- python -m venv .venv && source .venv/bin/activate
- pip install -e .

Option B — uv (optional)

- uv sync

Notes

- vector-quantize-pytorch is listed as a dependency, but the code has a safe fallback if it’s not importable at runtime (useful for CI/offline).
- transformers is optional; we fall back to a tiny whitespace tokenizer when HF downloads aren’t available.

Quickstart

- Run tests

  - uv run pytest -q

- Minimal training smoke test

  - python scripts/train.py

- Counting tasks (successor / predecessor / between)

  - python scripts/train_count.py --steps 1000
  - Uses a tiny fixed vocab (digits + a few relation tokens) and the pooled seq_head to predict the numeric answer.

- Natural-language pairs + counting (HF tokenizer)
  - python scripts/train_episodes.py train --steps 1000
  - You can resume/evaluate with the same script (see --help).

Basic usage (from Python)

- Build a tokenizer and model, then run a forward pass

  - from concept_learner.tokenizer import HFTokenizerWrapper
  - from concept_learner.model import CLModel
  - tok = HFTokenizerWrapper("bert-base-cased")
  - enc = tok.encode("2 : 3 :: 5 : ?", max_len=24)
  - ids = torch.tensor([enc.ids])
  - mask = torch.tensor([enc.mask])
  - model = CLModel(vocab_size=tok.vocab_size, d_model=128, num_classes=3)
  - logits_tok, logits_seq, vq_loss, indices, stop_logits, action_logits = model(ids, mask)

Repo layout

- src/concept_learner
  - tokenizer.py — HF wrapper with a fallback
  - encoder.py — 2-layer Transformer + pooled vector
  - vq_layer.py — ResidualVQLayer wrapper with a safe fallback
  - reasoning_ops.py — typed-state operators (Filter, Count, Add, Compare)
  - reasoning_v2.py — multi-step controller with STOP and operator mixing
  - decoder.py — token and pooled (sequence) heads
  - model.py — end-to-end wiring (Encoder → RVQ → FiLM → ReasonerV2 → Decoder)
  - trainer.py — tiny training utilities (losses, step, synthetic batches)
  - episodes.py — toy episode/number generators for curriculum-like tasks
- scripts — runnable demos (see Quickstart)
- tests — shape checks, smoke training, determinism

Design notes that match the code

- Text-only: everything is token sequences; no images or external tools
- Typed state: mask/val/boolean are maintained explicitly and updated by operators
- Multi-step: ReasonerV2 runs up to max_steps with a STOP head; action logits expose which operator the model leans on
- VQ bottleneck: a single residual VQ sits between encoder and reasoning; if vector-quantize-pytorch is missing, a small differentiable fallback keeps training/tests working
- Unified outputs: decoder exposes token-level and pooled sequence logits; training scripts primarily use the pooled head

Contributing / extending

- Add operators:
  - See reasoning_ops.py for patterns. Each op reads the typed state and returns updated (mask, val, boolean).
- Swap or deepen the encoder:
  - TinyEncoder is purposefully small; you can replace it with a larger Transformer.
- Replace the VQ:
  - ResidualVQLayer is wrapped behind a tiny interface; you can plug a different VQ.
- Multi-token decoding:
  - decoder.py is minimal; replace seq_head with an AR decoder if you need multi-token outputs.

License

BSD-3-Clause (see LICENSE.md)
