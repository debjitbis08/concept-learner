Concept Learner (minimal PyTorch skeleton)

Layout
- concept_learner/data/episode_gen.py: synthetic episodes over integers with hidden factors
- concept_learner/model/: tiny backbone, VQ bottleneck, relation and analogy heads, fast binder
- concept_learner/losses.py: InfoNCE, entropy regularizer, simple EWC proxy
- concept_learner/train.py: day-like training loop with mixed objectives
- concept_learner/eval.py: quick in-batch analogy probe

Quick start
1) Train for a few steps on CPU
   python -m concept_learner.train --steps 50 --batch 64

2) Evaluate analogy accuracy (randomly initialized model)
   python -m concept_learner.eval

Notes
- This is a minimal, single-file-per-component skeleton intended to be expanded. It runs on CPU and a single GPU.
- Replace the integer toy domain with your own descriptors; the loop stays the same.

