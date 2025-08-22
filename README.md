Concept Learner

Lightweight PyTorch framework for learning discrete concept codes with invariances and simple relations/analogies over symbolic domains (e.g., toy numbers). Includes a tiny transformer encoder, a VQ bottleneck, relation/analogy heads, and utilities for training/eval/checkpointing.

Repository layout
- concept_learner/data/episode_gen.py — synthetic episodes over integers with hidden factors (parity/mod/magnitude), masking, curriculum knobs, hard negatives.
- concept_learner/model/ — tiny backbone, VQ bottleneck (EMA), relation and analogy heads.
- concept_learner/losses.py — InfoNCE, entropy regularizer, simple EWC proxy.
- concept_learner/train.py — training loop with curriculum, EMA, checkpoints, and probes.
- concept_learner/eval.py — quick in-batch analogy probe; supports loading EMA checkpoints.
- apps/playground.py — minimal Gradio playground (optional) to query analogies.
- utils/checkpoints.py — CheckpointManager (save/load, GC).
- utils/ema.py — simple EMA wrapper and serialization helpers.
- PLANS_NEXT.md — extended plan and scaffolding for LLM teacher, playground/API, and long runs.

What's new (child-like grounding & overlap)
- Context tokens: each domain can prepend a special token (e.g., <home>, <park>) to encourage shared codes across contexts. Enable via TrainConfig.use_context_tokens (default True).
- Hierarchical codes: global shared VQ plus optional per-domain private residual VQs (TrainConfig.use_private_vq).
- Instance permanence head: lightweight head that learns to predict whether two views are the same instance (TrainConfig.use_instance_head). Trained alongside contrastive and relation/analogy losses.
- Sleep/consolidation: tiny replay buffer with periodic replay-only steps; stability regularizer (EWC-style) during sleep.
- Cross-domain alignment: optional regularizer aligning equivalent pairs across domains.
- Metrics: periodic reporting of codebook perplexity/usage, in-batch analogy accuracy, cross-context code sharing ratio, and nearest-neighbor domain diversity.

Requirements
- Python 3.10+
- PyTorch (CUDA build recommended for GPU)
- Optional: gradio (for playground)

Install
- Recommended: create a venv and install dependencies as needed.
- If you plan to run the playground: pip install gradio

Training
- Auto device selection: --device auto chooses CUDA if available, else CPU.
- Checkpoints: saved to --ckpt_dir (default checkpoints) every --save_every steps and at curriculum boundaries.

Examples
- Train for 10k steps on GPU (auto):
  python -m concept_learner.train --device auto --steps 10000 --batch 128 --ckpt_dir checkpoints --save_every 1000

- Resume from latest checkpoint in directory:
  python -m concept_learner.train --device auto --steps 2000 --batch 128 --ckpt_dir checkpoints --resume_latest

- Resume from a specific checkpoint path:
  python -m concept_learner.train --device auto --steps 2000 --batch 128 --resume_path checkpoints/ckpt_0008000.pt

Evaluation
- Evaluate with a specific checkpoint (EMA applied if present):
  python -m concept_learner.eval --device auto --ckpt checkpoints/latest.pt

Playground (optional)
- See docs/PLAYGROUND.md for details.
- Quick start (after training and checkpoint creation):
  pip install gradio
  python apps/playground.py --ckpt checkpoints/latest.pt --device auto

Notes
- The integer toy domain uses masked sequences with PAD distinct from digits; views curriculum and hard negatives are implemented.
- EMA VQ with dead code reinit and usage entropy is used to stabilize the bottleneck.
- Relation loss uses in-batch multiclass CE; analogy loss mixes a temperature-controlled classifier with an offset penalty.
- Checkpoints include model, optimizer, EMA, step, and metrics (including an EMA-based analogy accuracy probe).
