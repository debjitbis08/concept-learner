Concept Learner

Lightweight PyTorch framework for learning discrete concept codes with invariances and simple relations/analogies over symbolic domains (e.g., toy numbers). Includes a tiny transformer encoder, a VQ bottleneck, relation/analogy heads, and utilities for training/eval/checkpointing.

Update (curriculum + numbers)
- Numeric relations added in the synthetic domain: successor/predecessor, add_2, makes_ten_with, has_tens/has_ones (toggle via EpisodeConfig.enable_numeric_relations).
- Numeric “gold atoms” available via EpisodeGenerator.numeric_gold_atoms() to anchor training each epoch.
- LLM teacher scaffold with generator/critic prompts and programmatic validators (numbers, taxonomy) in concept_learner/teacher/llm_teacher.py.
- Phase helpers and per‑phase prompt builder in concept_learner/data/curriculum.py to synthesize per‑phase JSON.

Repository layout
- concept_learner/data/episode_gen.py — synthetic episodes over integers with hidden factors (parity/mod/magnitude), masking, curriculum knobs, hard negatives; extended numeric relations and gold atoms.
- concept_learner/data/curriculum.py — compact phase specs and prompt builder for LLM data synthesis.
- concept_learner/model/ — tiny backbone, VQ bottleneck (EMA), relation and analogy heads.
- concept_learner/losses.py — InfoNCE, entropy regularizer, simple EWC proxy.
- concept_learner/train.py — minimal learner definition (ConceptLearner, TrainConfig) and a tiny smoke‑test loop for triples.
- concept_learner/eval.py — quick in-batch analogy probe; supports loading EMA checkpoints.
- apps/playground.py — minimal Gradio playground (optional) to query analogies.
- utils/checkpoints.py — CheckpointManager (save/load, GC).
- utils/ema.py — simple EMA wrapper and serialization helpers.
- PLANS_NEXT.md — extended plan and scaffolding for LLM teacher, playground/API, and long runs.
 - docs/CURRICULUM.md — phase‑by‑phase curriculum with statuses and synthesis prompts.

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
 - Optional: torchview (to render the model architecture image)

Install
- Recommended: create a venv and install dependencies as needed.
- If you plan to run the playground: pip install gradio

Training
- Auto device selection: --device auto chooses CUDA if available, else CPU.
- Checkpoints: saved to --ckpt_dir (default checkpoints) every --save_every steps and at curriculum boundaries.
- LLM data: you can pre-generate numeric knowledge JSON using the LLM teacher, then train against it deterministically with --llm_data_json.
- Optimization knobs:
  - --lr, --sched {none,cosine}, --warmup, --lr_min: basic LR and warmup+cosine annealing (default: cosine with warmup).
  - --vq_weight/--vq_final_weight: weight per VQ commitment term (applied to both subject and object; anneals linearly from initial to final; defaults favor CE early).
  - --ce_temp: temperature for the in-batch CE over triples. Lower is sharper (default 0.2). Training uses in-batch negatives plus one hard negative per row.
  - --ce_mode {pair,inbatch}: pair trains a 2-class classifier (pos vs its hard-negative) per row; inbatch trains over all in-batch positives plus hard negatives (2B classes). Default pair for faster, more stable early learning.
  - --log_every: logging period; prints EMA-smoothed metrics and VQ code usage diagnostics.
  - --val_every: if >0, runs a small validation probe (fresh batch) every N steps and logs ce/acc.

Examples
- Train for 10k steps on GPU (auto):
  python -m concept_learner.train --device auto --steps 10000 --batch 128 --ckpt_dir checkpoints --save_every 1000 \
    --sched cosine --warmup 1000 --vq_weight 0.25 --val_every 200 --log_every 50

- Resume from latest checkpoint in directory:
  python -m concept_learner.train --device auto --steps 2000 --batch 128 --ckpt_dir checkpoints --resume_latest

- Resume from a specific checkpoint path:
  python -m concept_learner.train --device auto --steps 2000 --batch 128 --resume_path checkpoints/ckpt_0008000.pt

- Pre-generate LLM data (offline synthetic or online if OPENAI_API_KEY is set):
  python -m concept_learner.generate_data --out data/llm_numbers.json --count 4 --min_triples 512 --offline

- Train using pre-generated LLM JSON (no API calls during training):
  python -m concept_learner.train --use_llm_teacher --llm_data_json data/llm_numbers.json --steps 5000 --batch 128 --save_every 500

Evaluation
- Evaluate with a specific checkpoint (EMA applied if present):
  python -m concept_learner.eval --device auto --ckpt checkpoints/latest.pt

Model architecture image
- An architecture diagram is kept at docs/model_architecture.svg.
- To (re)generate it using torchview:
  - Install the optional dependency: poetry add --group dev torchview or pip install torchview graphviz
  - Run: poetry run update-model-arch --out docs/model_architecture.svg
  - If torchview/graphviz are unavailable, a placeholder SVG is written instead.

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
