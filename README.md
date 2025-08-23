# Subject-Aware Concept Learner

A compact, grow‑able architecture that learns **discrete concepts** (VQ), applies **small reasoning steps**, and answers with a **single decoder**. Scales from **Kindergarten → Nursery → Professional** without changing the skeleton.

---

## 0) Goals

- Learn **stable, reusable concepts** (e.g., successor, compare, attribute-of, color, shape, is-a).
- Support **multiple subjects** (math, shapes, colors, animals, mixed) with **one model**.
- Execute short **reasoning chains** (1–4 steps) over a typed working state.
- Keep it **text-only**; plug in tiny lookup tools (tables) where needed.

---

## 1) High-Level Architecture

### Visual Pipeline (ASCII)

```
[ TEXT PROMPT ]
  e.g., "8 + 2 = ?", "color of banana?", "Is number of red fruits > 4?"

        |
        v
+--------------------------------------+
| Tiny Transformer Encoder (2 layers)  |
|  - tokens: numbers/words/symbols     |
|  - outputs: pooled h and sequence H  |
+--------------------------------------+
        |
        | (pooled h)
        v
+-------------------------------+
| VQ_subject (K≈8–12, unsup.)   |
| -> subject index s            |
| -> subject emb e_s            |
+-------------------------------+
        |
        |  (conditioning: bias / FiLM / CLN)
        v
+--------------------------------------+
| Subject Conditioning                 |
|  - modulate h/H with e_s             |
|  - optionally bias/mask concepts     |
+--------------------------------------+
        |
        | (conditioned h~)
        v
     +--------------------+       +--------------------+
     | VQ_type (K≈24)     |       | VQ_param (K≈48)    |
     | -> e_type          |       | -> e_param         |
     +--------------------+       +--------------------+
              \                         /
               \                       /
                \                     /
                 v                   v
                +-----------------------+
                |  z = [e_type; e_param]|
                +-----------------------+
                         |
                         |   (plus h~, e_s)
                         v
              +-------------------------------+
              | Reasoning Cells (1–4 steps)   |
              |  s_{t+1} = s_t + g⊙EXEC(...)  |
              +-------------------------------+
                         |
                         v
              +-------------------------------+
              | Unified Decoder               |
              |  - one vocab (numbers/words)  |
              |  - can score MCQ options too  |
              +-------------------------------+
                         |
                         v
                      ANSWER
```

### Typed Working State (internal)

```
STATE = {
  ITEMS:  list of candidate objects (from text or lookup),
  MASK:   boolean mask over ITEMS,
  VAL:    scalar slot (e.g., a count / numeric value),
  BOOL:   boolean slot (yes/no),
}
```

---

## 2) Components

### 2.1 Tokenizer / Vocab

- Integers 0–100 (extendable), symbols `+ − × ÷ < > = :: ?`, words: `next, before, between, odd, even, count, number, color, shape, sides, is, has, of, which, bigger, smaller, more, fewer, yes, no`, subject nouns (`banana, apple, cat, dog, triangle, square, red, blue, ...`).

### 2.2 Encoder (Tiny Transformer)

- 2 layers, d≈128, 4 heads; pooled vector `h` and sequence states `H`.

### 2.3 VQ_subject (Unsupervised Routing)

- Codebook K≈8–12, dim≈32, EMA updates; commitment loss β≈0.25.
- Regularizers: code-usage entropy; paraphrase-stability (InfoNCE) across rewordings.
- Conditioning: start with **bias** (add transformed `e_s` to hidden), upgrade to FiLM/CLN if needed.

### 2.4 Concept Bottleneck (Factored VQs)

- **VQ_type (K≈24, dim≈64):** operator family (e.g., `successor, predecessor, add, subtract, multiply, compare, attribute-of, is-a, has-a, filter, count, groupby, argmax`).
- **VQ_param (K≈48, dim≈64):** small integers (−5…+10), tokens like `{even, odd, <, >, =, red, blue, circle, square, fruit, animal, sides, color}`.
- Outputs concatenated: `z = [e_type; e_param]`.
- Losses: commitment + EMA + mild code-usage entropy.

### 2.5 Reasoning Cells (Concept Application)

- Controlled operator application (not an RNN):

  - Inputs: working state `s`, concepts `e_type, e_param`, subject `e_s`.
  - Gate: `g = σ(W_g · [s; e_type; e_param; e_s])`.
  - Micro-execution: `u = EXEC(e_type, e_param, s)`.
  - Residual update: `s_next = s + g ⊙ u`.

- Start with **1 step**; later allow **2–4 steps** + optional **STOP** concept.

#### EXEC Registry (initial operators)

- `Filter(category=X)` → update `MASK` by `is_a(item, X)`.
- `Filter(attr=A, value=B)` → `MASK ∧= (attr(item,A)==B)`.
- `Count()` → `VAL = sum(MASK)`.
- `Compare(op, k)` → `BOOL = op(VAL, k)`.
- `AttributeOf(name)` → map item to attribute (e.g., sides(shape)).
- `Successor/Predecessor` / `Add(k)` / `Sub(k)` / `Mul(k)`.
- `IsA`, `HasA`, simple `Map`.
- **Tool hooks (later):** calculator, table lookup, code runner.

### 2.6 Unified Decoder

- **Single classifier** over a shared vocab (numbers + words + `yes/no`).
- Input: projection of `[s_final ; z ; e_s]`.
- If multi-token answers later, replace with a tiny **1-layer autoregressive decoder** (same input conditioning) — MCQ scoring uses log-probs from the same decoder.

---

## 3) Data & Knowledge

### 3.1 Mini Knowledge Tables (text-only tools)

- `color(noun)` table: `banana→yellow, apple→red/green, cherry→red, ...`
- `sides(shape)` table: `triangle→3, square→4, ...`
- `is_a(noun, category)` table: `apple→fruit, cow→animal, ...`

### 3.2 Exercise Format (JSONL)

```json
{
  "id": "ex-001",
  "prompt": "Is number of red fruits greater than 4?",
  "options": ["yes", "no"], // optional; omit for open answer
  "answer": "yes",
  "subject": null, // optional; can be null for unsup
  "tags": ["mixed", "filter", "count", "compare", ">"],
  "facts": { "basket": ["apple", "banana", "cherry", "strawberry", "apple"] }
}
```

- Paraphrase pairs for invariance: tie via contrastive loss.

---

## 4) Training Objectives

- **Main:** cross-entropy on decoder output.
- **VQ:** commitment + EMA; mild code-usage entropy.
- **Subject stability:** contrastive (same meaning → same subject code).
- **Equivalence:** pull together prompts that are equivalent (`8+2` ↔ `next next of 8`).
- **Sparsity (optional):** small penalty on number of reasoning steps used.

---

## 5) Curriculum (Grow Up)

- **Phase KG (starter):** successor/predecessor, compare `< = >`, parity, add/sub small k, color-of, shape-of (sides), simple is-a.
- **Phase Nursery:** animals, mixed prompts (filters + count + compare), group-by/argmax, attribute chains.
- **Replay:** keep a small rehearsal buffer to avoid forgetting.

---

## 6) Minimal Hyperparameters

- Encoder: 2-layer Transformer, d=128, heads=4.
- VQ_subject: K=8–12, dim=32, β=0.25, EMA=0.99.
- VQ_type: K=24, dim=64; VQ_param: K=48, dim=64.
- Reasoning: 1 step (start), hidden d=128 for gates/MLP.
- Decoder: MLP 128→(|vocab|).
- Optim: AdamW, lr=3e-4, batch=256, label smoothing=0.05.

---

## 7) Evaluation

- Accuracy per task family (math, shapes, colors, animals, mixed).
- **Concept purity:** mutual info between VQ indices and human labels (e.g., `+1`, `red`, `>`, `sides`).
- **Code utilization:** % active codes; entropy.
- **Length generalization:** train on short steps, test on longer compositions.
- **Cross-subject analogies:** e.g., `triangle:square :: 3:?` → `4`; `banana:yellow :: apple:?` → `red/green`.
- Ablations: no-VQ, single-VQ, no-reasoning, AR decoder vs classifier.

---

## 8) Implementation Sketch (PyTorch-style pseudocode)

### 8.1 Modules (interfaces)

```python
class TinyEncoder(nn.Module):
    def forward(self, tokens) -> tuple[h, H]:
        # pooled h (B,d), sequence H (B,T,d)
        ...

class VQ(nn.Module):
    # EMA codebook quantizer
    def forward(self, x) -> tuple[e, idx, vq_loss]:
        ...

class SubjectCondition(nn.Module):
    # start with bias; can switch to FiLM/CLN later
    def forward(self, H, e_s) -> tuple[h_tilde, H_tilde]:
        ...

class ExecRegistry(nn.Module):
    def forward(self, e_type, e_param, state) -> delta_state:
        # switch on e_type; use e_param; may call lookup/calculator
        ...

class ReasoningCell(nn.Module):
    def __init__(self, exec_registry): ...
    def forward(self, state, e_type, e_param, e_s):
        x = torch.cat([state.to_vec(), e_type, e_param, e_s], dim=-1)
        g = torch.sigmoid(self.gate(x))
        u = self.exec_registry(e_type, e_param, state).to_vec()
        return state.from_vec(state.to_vec() + g * u)

class UnifiedDecoder(nn.Module):
    def forward(self, state, z, e_s) -> logits:
        x = proj(torch.cat([state.to_vec(), z, e_s], dim=-1))
        return head(x)  # logits over shared vocab
```

### 8.2 Forward Pass

```python
def forward(prompt_tokens, kb=None):
    h, H = encoder(prompt_tokens)

    e_s, s_idx, vq_s_loss = vq_subject(h)
    h_tilde, H_tilde = subject_condition(H, e_s)

    e_type, t_idx, vq_t_loss = vq_type(h_tilde)
    e_param, p_idx, vq_p_loss = vq_param(h_tilde)
    z = torch.cat([e_type, e_param], dim=-1)

    state = init_state_from_text(prompt_tokens, kb)  # ITEMS/MASK/VAL/BOOL
    for t in range(NUM_STEPS):  # start with 1
        state = reasoning_cell(state, e_type, e_param, e_s)

    logits = decoder(state, z, e_s)
    return logits, (vq_s_loss + vq_t_loss + vq_p_loss)
```

---

## 9) Example Walk-Throughs

### 9.1 "Is number of red fruits > 4?"

- Subject ≈ mixed; Type/Param → `Filter(is_a, fruit)`, `Filter(color, red)`, `Count`, `Compare(>, 4)` (may be composed over 1–2 steps by EXEC).
- STATE updates: `MASK` by filters → `VAL = sum(MASK)` → `BOOL = (VAL>4)` → decoder → `yes/no`.

### 9.2 "2 : 3 :: 5 : ?"

- Type `Successor`, Param `+1`; apply to `5` → `6`.

### 9.3 "Which has 3 sides?"

- Type `AttributeOf('sides')`; compare equals `3` over candidates; decoder emits the correct noun (or scores the given options).

---

## 10) Scaling to Professional

- Deeper/wider encoder; retrieval-augmented context.
- Hierarchical VQ (coarse→fine) or residual VQ if concepts tangle.
- More reasoning steps (2–4), learned STOP.
- Tool hooks: calculator, code runner, SQL, search.
- Adapters/LoRA per domain; shared codebooks keep concepts transferable.

---

## 11) Build Order (Checklist)

1. Tokenizer + datasets (JSONL + tiny knowledge tables).
2. Encoder + VQ_subject (bias conditioning) + VQ_type/VQ_param.
3. Minimal EXEC with: `Filter(is_a)`, `Filter(color)`, `Count`, `Compare`, `Add/Sub/Successor`, `AttributeOf('sides')`.
4. One reasoning step; unified decoder (classifier over shared vocab).
5. Losses: CE + VQ commitment/EMA + simple contrastive (paraphrase pairs).
6. Curriculum: KG basics → mixed → nursery; add replay.
7. Eval + ablations; then add 2nd reasoning step or AR decoder if needed.

---

## 12) Visual: Internal Reasoning Flow (ASCII)

```
[h, H] --VQ_s--> e_s --> condition --> h~
                      \
                       +--VQ_type--> e_type --+           +--> logits → answer
                       +--VQ_param-> e_param -+--> z ---->
                                             |
                STATE_init (ITEMS,MASK,VAL,BOOL) --Reasoning Cell(s)--> STATE_final
```

---

**This document is meant to be the starting blueprint.** You can keep the skeleton fixed and iterate on (a) concept codebooks, (b) EXEC catalog, (c) reasoning step count, and (d) decoder style as the curriculum grows.
