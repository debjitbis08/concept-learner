Reading the training logs

This document explains every item in the training logs emitted by scripts/train_episodes.py and how to interpret them. Example log:

step 1100/6000 loss=1.5965 seq=1.4563 vq=0.0086 stop=0.4600 p(yes)~0.338 val_pair=0.541 val_cnt=0.577 avg=0.559 lr=7.52e-05
  VQ: h0:util=1.00,ppl=0.83 h1:util=1.00,ppl=0.82 h2:util=1.00,ppl=0.72 | stab h0:chg=0.102 h1:chg=0.141 h2:chg=0.062
  eval in-dist=0.578 range-OOD=0.558 template-OOD=0.583 boundary-OOD=0.547
  Saved best checkpoint: runs/episodes_b10/best.pt (val_acc=0.559)


Core training line
- step a/b: iteration index and total steps for this run.
- loss: total objective used for backprop.
  - Computed as: loss = seq + lambda_vq * vq + lambda_stop * stop.
- seq: classification loss from the unified decoder on the concatenated batch (cross-entropy with optional label smoothing).
- vq: vector-quantization loss returned by the VQ layer (raw, before multiplying by lambda_vq in the total loss).
- stop: STOP-head loss (masked binary cross-entropy over max_steps; raw, before multiplying by lambda_stop in the total loss).
- p(yes)~x.xxx: average probability of the YES class over the batch on the two-class slice {NO, YES}. This is a rough sanity signal; because the batch also includes numeric and parity targets, it is not a calibrated class prior.
- val_pair: quick validation accuracy on pair relations (binary yes/no), averaged over a few mini-batches. Uses the same natural-language templates as training unless a template holdout is configured.
- val_cnt: quick validation accuracy on counting tasks (successor, predecessor, between) with numeric targets.
- avg: the metric used for checkpoint selection = 0.5 * (val_pair + val_cnt).
- lr: current learning rate after scheduling (cosine, plateau, or none).


VQ diagnostics line
- VQ: hK:util=U,ppl=P for each quantizer head:
  - util: fraction of codebook entries used in the current mini-batch = unique_codes / codebook_size. Values near 1.00 indicate wide usage; very low values suggest many dead codes.
  - ppl: normalized perplexity of code usage = exp(H) / K, where H is the entropy of the code histogram and K the codebook size. Higher means more uniform usage; very low means peaky usage.
- stab hK:chg=C: stability probe for each head. Two forward passes are run in train mode (noise active) on the same mini-batch; chg is the fraction of assignments that changed. Smaller is more stable; exactly 0 would indicate deterministic assignments (or a broken probe).


Eval diagnostics line
- eval in-dist: accuracy on in-distribution numbers (idx in train_range), using the same NL templates as training. Measures how well the semantics are learned in-range.
- range-OOD: accuracy on out-of-range numbers (idx in ood_range), still using the training templates. Isolates generalization to new magnitudes. With digit-level tokenization, numbers share digit tokens and this typically improves.
- template-OOD: accuracy on in-range numbers but with held-out natural-language templates (tmpl_ood). Measures robustness to wording/paraphrases.
- boundary-OOD: accuracy near decision boundaries for greater/smaller (|a-b| in {0,1,2}). Measures margin quality on hard negatives.


Checkpoint line
- Saved best checkpoint: path (val_acc=avg): emitted when avg (0.5 * val_pair + val_cnt) improves over the best so far. If EMA is enabled, EMA weights are saved alongside.


Tokenizer backend log
- At startup you will see one line identifying the tokenizer:
  - Tokenizer backend: hf (bert-base-cased), vocab_size=30522
  - Tokenizer backend: simple (_simple_fallback), vocab_size=…
- This helps verify that the HuggingFace tokenizer is active. With the fallback tokenizer, unseen numbers at eval time used to create new token IDs and hurt range-OOD. The code now applies digit-level tokenization so numerals are split into per-digit tokens for both backends.


How to interpret common patterns
- In-dist and template-OOD tracking each other and reaching high values means the model learned the task semantics and is robust to wording.
- Range-OOD significantly below in-dist indicates poor magnitude generalization. After digit-level tokenization, expect an uplift because unseen numbers share digit tokens with seen ones.
- Boundary-OOD below in-dist suggests a weak margin near hard cases; emphasizing greater/smaller relations or mining harder negatives often helps.
- p(yes) hovering near 0.5 is normal; it is not a calibrated prior and shouldn’t be over-interpreted.


Where metrics come from in code (for reference)
- Quick evals live in _quick_eval in scripts/train_episodes.py.
- Range and template splits are controlled with idx_range (train_range / ood_range) and template_filter (tmpl_train / tmpl_ood).
- Boundary eval uses EpisodeGenerator.sample_posneg_pairs_boundary.
- VQ diagnostics and stab are printed inside the train loop when log_every triggers.

