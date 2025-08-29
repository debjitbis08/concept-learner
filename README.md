# Subject-Aware Concept Learner

Concept Learner is an experimental project to build a model that learns concepts directly, starting with Grade 1 mathematics. The aim is to progress through school-level math one grade at a time, exploring how a model can acquire and apply concepts rather than just memorize patterns.

This is not a LLM and not a wrapper around existing LLMs, it's a ground-up attempt to rethink how AI learns reasoning.

This project is also an experiment in vibe-coding. Many developers today use LLMs to vibe-code apps; here, the twist is to see if we can vibe-code an AI model itself, in other words, use AI (with human guidance) to help design and build another AI.

Architecture in short: a tiny Transformer encoder, a multi-VQ bottleneck (parallel + sequential), a few typed reasoning ops, and a simple unified decoder.

## Install

### Prereqs: Python 3.13+

#### uv (optional)

- uv sync

## Quickstart

- Run tests

  - uv run pytest -q

- Training from scratch

  - `python scripts/train_episodes.py train \
--steps 12000 --base10 --batch_size 512 --d_model 256 \
--save_dir runs/episodes_b10 \
--ckpt_every 500 --log_every 100 \
--lr 8e-5 --sched cosine --warmup_ratio 0.03 --min_lr 1e-6 \
--ema_decay 0.995 --weight_decay 0.003 --label_smoothing 0.03 \
--lambda_vq 0.25 --lambda_stop 0.3`

- If you hit CUDA OOM, try:
  - Lower `--batch_size` (e.g., 64 → 32 or 16)
  - Enable mixed precision: add `--amp` (CUDA only)
  - Set env var to reduce fragmentation: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

- Resume Training

  - `python scripts/train_episodes.py train \
--steps 12000 --base10 --batch_size 512 --d_model 256 \
--save_dir runs/episodes_b10 \
--ckpt_every 500 --log_every 100 \
--lr 8e-5 --sched cosine --warmup_ratio 0.03 --min_lr 1e-6 \
--ema_decay 0.995 --weight_decay 0.003 --label_smoothing 0.03 \
--lambda_vq 0.25 --lambda_stop 0.3 \
--resume runs/episodes_b10/best.pt`

- Evaluation

  - `python scripts/train_episodes.py eval --checkpoint runs/episodes_b10/best.pt --d_model 256 --eval_batches 100 --base10`

## Sample Evaluation Output

```
Eval accuracy over 6400 examples: 0.7180
Per-relation accuracy:
  same_parity : 330/690 = 0.478
  successor   : 705/727 = 0.970
  predecessor : 664/674 = 0.985
  add_2       : 420/705 = 0.596
  same_tens   : 378/727 = 0.520
  same_ones   : 350/725 = 0.483
  makes_ten   : 373/729 = 0.512
  greater     : 690/716 = 0.964
  smaller     : 685/707 = 0.969
Acc@thr=0.4: 0.707
Acc@thr=0.5: 0.718
Acc@thr=0.6: 0.725
Best fixed-threshold acc in [0.35,0.65]: 0.732 at thr=0.52
greater close (|a-b|<=1): 396 ex, acc=0.977
greater far   (|a-b|>1):  320 ex, acc=0.947
smaller close (|a-b|<=1): 361 ex, acc=0.978
smaller far   (|a-b|>1):  346 ex, acc=0.960
Sample QA predictions:
  Q: Do 11 and 12 have the same parity?
     gold=no pred=no p(yes)=0.469 rel=same_parity
  Q: Is 28 the predecessor of 29?
     gold=yes pred=yes p(yes)=0.992 rel=predecessor
  Q: Is 7 greater than 6?
     gold=yes pred=yes p(yes)=1.000 rel=greater
  Q: Do 69 and 79 have the same tens digit?
     gold=no pred=yes p(yes)=0.528 rel=same_tens

Sample equality QA (varied phrasing):
  Q: Is the successor of 93 equal to 93 + 1?
     gold=yes pred=yes p(yes)=1.000
  Q: Is next of 33 = 33 + 1?
     gold=yes pred=yes p(yes)=1.000
  Q: Is previous of 27 = 27 - 1?
     gold=yes pred=yes p(yes)=1.000
  Q: Is the successor of 44 = 45?
     gold=yes pred=yes p(yes)=0.999

Counting evaluation:
Counting eval accuracy: 1.000
  Q: What number comes between 34 and 36?
     gold=35 pred=35
  Q: What is the predecessor of 65?
     gold=64 pred=64
  Q: What number comes before 36?
     gold=35 pred=35
  Q: What number comes after 43?
     gold=44 pred=44
```

## License

BSD-3-Clause (see LICENSE.md)
