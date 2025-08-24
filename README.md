# Subject-Aware Concept Learner (Text-Only)

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
+--------------------------------------+
        |
        v
+---------------------------+
| VQ-L1 : REGION / SUBJECT  |  (unsupervised)
+---------------------------+
        |
        v
+---------------------------+
| VQ-L2 : DOMAIN / SUBAREA  |
+---------------------------+
        |
        v
+---------------------------+
| VQ-L3 : OPERATOR FAMILY   |
+---------------------------+
        |
        v
+---------------------------+
| VQ-L4 : ARGUMENT / VALUE  |
+---------------------------+
        |
        v
      [e1;e2;e3;e4]  →  Reasoning Core (Schema + MoE Experts)
                            • schema ← f(e3)  (S->S, S->R, R->R, (R,R)->B/R)
                            • gate 1–2 experts per schema with [e1,e2,e3]
                            • apply 1–4 steps with learned STOP
        |
        v
         Unified Decoder (single vocab)
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

## 2) Components Components

### 2.1 Tokenizer / Vocab

- Integers 0–100 (extendable), symbols `+ − × ÷ < > = :: ?`, words: `next, before, between, odd, even, count, number, color, shape, sides, is, has, of, which, bigger, smaller, more, fewer, yes, no`, subject nouns (`banana, apple, cat, dog, triangle, square, red, blue, ...`).

### 2.2 Encoder (Tiny Transformer)

- 2 layers, d≈128, 4 heads; pooled vector `h` and sequence states `H`.

### 2.3 VQ_subject (Unsupervised Routing)

- Codebook K≈8–12, dim≈32, EMA updates; commitment loss β≈0.25.
- Regularizers: code-usage entropy; paraphrase-stability (InfoNCE) across rewordings.
- Conditioning: start with **bias** (add transformed `e_s` to hidden), upgrade to FiLM/CLN if needed.

### 2.4 Concept Bottleneck (Hierarchical VQs: L1→L4)

Implement a **brain-style hierarchy** with four VQ levels; each level sees the encoder state **and** the parent codes (top‑down conditioning).

**Top‑down quantization:**

```
h1 = Proj1(h)                           → VQ‑L1 (Region/Subject) → e1
h2 = Proj2([h, e1])                    → VQ‑L2 (Domain/Subarea)  → e2
h3 = Proj3([h, e1, e2])                → VQ‑L3 (Operator family) → e3
h4 = Proj4([h, e1, e2, e3])            → VQ‑L4 (Argument/Value)  → e4
z = [e1; e2; e3; e4]
```

- Each level uses EMA VQ with commitment loss and mild code‑usage entropy.
- Conditioning biases children to specialize without brittle hard masks.
- Optional **ResidualVQ within a level** (later) if that level saturates.

**Starter codebook sizes (tunable):**

- L1 (Region): K=8–12, dim=32
- L2 (Domain): K=16–24, dim=48
- L3 (Operator family): K=24–32, dim=64
- L4 (Argument/Value): K=48–96, dim=64

**Routing:** straight‑through (STE) or Gumbel‑Softmax w/ annealing. Add per‑level load‑balancing to avoid dead branches.### 2.4b VQ Architecture (Detailed)
A brain‑style hierarchical quantizer with **top‑down conditioning**, combining **TSVQ‑like** routing at upper levels with **Residual / Product VQ** where the space fans out.

#### Summary table

| Level  | Role             | Quantization style                                                | K (start) |     Dim |
| ------ | ---------------- | ----------------------------------------------------------------- | --------: | ------: |
| **L1** | Region / Subject | TSVQ‑like (flat VQ with top‑down prior)                           |      8–12 |      32 |
| **L2** | Domain / Subarea | TSVQ‑like (conditioned on L1)                                     |     16–24 |      48 |
| **L3** | Operator family  | **Flat VQ**, add **Residual VQ** if crowded                       |     24–32 |      64 |
| **L4** | Argument / Value | **Grouped / Product VQ** (e.g., numeric / attr‑type / attr‑value) | 3×(16–32) | 3×32–64 |

#### Top‑down conditioning (end‑to‑end trainable)

```
h1 = Proj1(h)                             → VQ_L1 → e1, idx1, Lvq1
h2 = Proj2([h, e1])                      → VQ_L2 → e2, idx2, Lvq2
h3 = Proj3([h, e1, e2])                  → VQ_L3 → e3, idx3, Lvq3   # + optional Residual3
# Grouped / Product VQ at L4
u4_num, u4_attr, u4_val = Split4( Proj4([h, e1, e2, e3]) )
e4_num, i4n, Lvq4n = VQ_L4_num(u4_num)
e4_attr, i4a, Lvq4a = VQ_L4_attr(u4_attr)
e4_val,  i4v, Lvq4v = VQ_L4_val(u4_val)

e4 = concat(e4_num, e4_attr, e4_val)
path = [e1, e2, e3, e4]
```

- **TSVQ‑like**: we keep levels **separate** but condition children on parent embeddings; we learn soft **P(Lk|Lk‑1)** priors (no hard masks) to bias routing.
- **Residual\@L3 (optional):** if operator families saturate:

```
e3c, idx3c = VQ3_coarse(h3)
r3 = h3 - e3c
e3f, idx3f = VQ3_fine(r3)
e3 = concat(e3c, e3f)
```

#### Routing and robustness

- **STE or Gumbel‑Softmax** at each level; anneal temperature if using Gumbel.
- **Load‑balancing / entropy** per level to avoid dead branches.
- **Top‑2 late binding (optional):** at one chosen level, keep the best **two** codes and evaluate both downstream; pick the better via decoder score. Cuts error propagation from early splits.

#### Losses (per level + global)

- **Task**: `L_task = CE(decoder(...), y)`
- **VQ per level**: `Σ λ_vq^k · L_vq_k` (commitment + EMA updates)
- **Usage / balance**: `Σ λ_ent^k · H(idx_k)` and small **load‑balance** term
- **Hierarchy priors**: `λ_h · [ CE(idx2 | idx1) + CE(idx3 | idx2) + CE(idx4 | idx3) ]` (learned soft priors)
- **Equivalence / counterfactuals / step‑sparsity**: as defined earlier
- **Anchors** when expanding codebooks (keep old centroids stable)

#### Monitoring (by level)

- Utilization (% active), perplexity of code usage
- Purity (MI with human tags: region/domain/family/arg)
- Drift vs previous release (centroid L2 distance; top‑k agreement)
- Prior calibration (quality of P(Lk|Lk‑1))

#### Library wiring (`vector-quantize-pytorch`)

```python
from vector_quantize_pytorch import VectorQuantize

VQ_L1 = VectorQuantize(dim=32, codebook_size=12, decay=0.99,
                       commitment_weight=0.25, kmeans_init=True,
                       use_cosine_sim=True, codebook_dim=32,
                       threshold_ema_dead_code=2)
# Repeat with sizes for L2/L3 and three grouped VQs for L4.
# Feed Proj_k([...]) as (B, 1, dim) to get (z_q, indices, vq_loss).
```

#### ASCII: hierarchy sketch

```
L1: REGION      ──┬─ math
                  ├─ shapes
                  ├─ colors
                  └─ animals / mixed
                      |
L2: DOMAIN        ───┬─ numbers / relations / ops (for math)
                      ├─ geometry / attributes (for shapes)
                      └─ taxonomy / attributes (for animals/colors)
                          |
L3: OP FAMILY      ──── { compare | attribute-of | filter | group | count | add | mul | ... }
                          |
L4: ARG / VALUE    ──── (grouped) [ numeric | attr-type | attr-value ]
```

---

### 2.5 Reasoning Core (Schema + MoE Experts)

A scalable executor that operates over a **typed state** and is **driven by the hierarchical path**.

**Typed State (unchanged):** `STATE = { ITEMS, MASK, VAL, BOOL }`

**Schema binding:**

- **Schema** is determined by **L3 (operator family)**:
  L3→`S→S` (filter/map), `S→R` (reductions), `R→R` (arithmetic), `(R,R)→B/R` (compare/arith).
- **Expert gating (MoE)** within the chosen schema uses `[e1,e2,e3]` so areas/subareas learn style without forking the model.
- **Parameters** come from **L4 (e4)**; retrieval/tools can enrich arguments.

**One reasoning step:**

```
x = concat(state.to_vec(), e1, e2, e3, e4)
expert = gate_schema(e3, context=[e1,e2,e3])
Δ = expert.apply(state, param=e4, context=[e1,e2,e3])   # Deep Sets for sets; MLP/tool for scalars
g = σ(Wg x)
state_next = state.from_vec(state.to_vec() + g ⊙ Δ)
```

Run **1–4 steps** with a learned **STOP** token. Optional early consistency to simple hard ops (e.g., Count) on synthetic data.### 2.6 Unified Decoder

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

### 2.7 Minimal Operator Basis (logic‑derived)

A principled, minimal set of operators chosen from **finite model theory** and **Presburger arithmetic**. They align with your typed state and scale cleanly.

**Typed state:** `STATE = { ITEMS, MASK, VAL, BOOL }`

#### Core operators (Tier‑0)

- **Filter(p)** — _S→S_: select items satisfying predicate `p`.

  - `p` built from **atoms** (e.g., `attr=item[color]==red`, `is_a(item, fruit)`) and **connectives** {AND, NOT}.
  - **Update:** `MASK ← min(MASK, score_p(items))` (idempotent, commutative, monotone).
  - Implement as per‑item MLP scorer + `min` with current mask (or `softmin_τ`).

- **Count()** — _S→R_: number of selected items.

  - Deep‑Sets form: `VAL ← Σ_i φ(item_i)·MASK_i` (φ can be 1 or a tiny MLP).
  - Gives **Exists** (`VAL>0`) and **ForAll** (`VAL == |S|`) for free.

- **Compare(op, k)** — _(R,R)→B_: numeric comparison with `op ∈ {<, =, >}`.

  - Monotone comparator: `BOOL ← σ( a · (VAL − k) )` with `a≥0`.

- **Add(k)** — _R→R_: arithmetic on a scalar with small integer `k`.

  - Linear update: `VAL ← VAL + α·k` (α≈1), yielding **Successor** for `k=+1`; **Sub** via negative `k`.

#### Why these 4 suffice

- **Filter**/**Count** capture **FO logic with counting** over finite structures (selection + counting quantifiers).
- **Add**/**Compare** capture **Presburger arithmetic** (numbers with 0,1,+,<,=).
- **Deep Sets** justifies a single invariant aggregator (sum) for set→scalar reasoning.

#### Derived/macros (no new primitives)

- **OR** from De Morgan: `p ∨ q ≡ ¬(¬p ∧ ¬q)` (still only AND, NOT inside `Filter`).
- **Exists/ForAll** from `Count` + `Compare`.
- **Range tests**: `(a ≤ x ≤ b)` via two `Compare`s.
- **ArgMax/ArgMin**: fold using `Compare` (or keep as a macro on S→S + S→R).
- **Make‑ten / complements**: arithmetic with `Add`.
- **GroupBy/Sum(attr)**: macros built from repeated `Filter` and `Count` (and `Sum` if you expose a numeric attribute aggregator later).

---

## 4) Training Objectives

- **Main:** cross-entropy on unified decoder output.
- **VQ:** commitment + EMA; mild code-usage entropy; anchors when expanding codebooks.
- **Subject stability:** contrastive (same meaning → same subject code).
- **Equivalence / invariance:** tie paraphrases and equivalent forms (e.g., `8+2` ≡ `next next of 8`).
- **Counterfactuals:** edit attributes/values; enforce correct deltas.
- **Schema regularizer:** encourage consistent **Type→schema** mapping (CE); prevent schema drift.
- **MoE regularizers:** load balancing + sparsity on expert selection (top-1/top-2 gate).
- **Re-exec consistency (optional early):** align neural expert outputs with simple hard ops on synthetic cases.
- **Sparsity on steps:** small penalty to keep reasoning short; learned **STOP**.

---

## 5) Curriculum (Grow Up)

- **Phase KG (starter):** successor/predecessor, compare `< = >`, parity, add/sub small k, color-of, shape-of (sides), simple is-a.
- **Phase Nursery:** animals, mixed prompts (filters + count + compare), group-by/argmax, attribute chains.
- **Replay:** keep a small rehearsal buffer to avoid forgetting.

---

## 6) Minimal Hyperparameters

- Encoder: 2-layer Transformer, d=128, heads=4.
- **VQ‑L1 (Region):** K=8–12, dim=32, decay=0.99, commitment=0.25, cosine sim, kmeans init, dead‑code threshold=2.
- **VQ‑L2 (Domain):** K=16–24, dim=48, same settings.
- **VQ‑L3 (Operator family):** K=24–32, dim=64, same settings.
- **VQ‑L4 (Argument/Value):** K=48–96, dim=64, same settings.
- Reasoning: start with 1 step (hidden d=128); add STOP when moving to 2–4 steps.
- Decoder: MLP 128→(|vocab|) (swap to 1-layer AR decoder if multi-token answers needed).
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

- Deepen/widen encoder; retrieval-augmented context.
- **Grow hierarchy:** add codes per level when utilization >\~80% and purity drops; optionally add **ResidualVQ within crowded levels**.
- **Reasoning:** expand to 2–4 steps with learned STOP; schema-based MoE experts per signature; grow experts gradually (top‑1/top‑2 gating + load balance).
- **Tools:** plug calculator, lookup, SQL/search as specialized experts behind schema APIs.
- **Continuity:** anchor centroids per level; lineage map when splitting codes; replay + distill from prior version to preserve “self”.

### Growth Protocol & Pitfalls

- **Protocol:** warmup L1 → add L2 (anchor L1) → add L3 (anchor L1–L2) → add L4; promote codes only when utilization high. Keep soft priors P(Lk|Lk−1) learnable, not hard.
- **Dead branches:** per‑level entropy + dead‑code refresh; periodic k‑means reinit for inactive codes.
- **Misrouting:** avoid hard masks early; use teacher forcing on shallow synthetic labels for a few epochs, then relax.
- **Drift:** centroid anchoring (L2‑SP), KD on a frozen probe set; do not recycle indices within a major version.

---

## 11) Build Order (Checklist)

1. Tokenizer + datasets (JSONL + tiny knowledge tables or retrieval stubs).
2. Encoder + VQ_subject (bias conditioning) + VQ_type/VQ_param.
3. **Schema executors (learned):**

   - Implement **S→S** and **S→R** with Deep Sets; **R→R** and **(R,R)→B/R** with tiny MLPs.
   - One reasoning step; no STOP yet.

4. Unified decoder (classifier over shared vocab). Train end-to-end on KG tasks.
5. Add **STOP** + 2nd step; enable **equivalence/counterfactual** objectives.
6. Add **MoE** per schema (start with 2 experts); enable load-balancing.
7. Add retrieval/tool hooks behind schema APIs (lookup `color`, `sides`, calculator).
8. Replay + anchors for continuity; evaluate code utilization, schema mapping stability.

---

## References (selected)

- **Sheffer, H. M.** (1913). _A set of five independent postulates for Boolean algebras, with applications to logical constants._ Transactions of the AMS.
- **Post, E. L.** (1921). _Introduction to a general theory of elementary propositions._ American Journal of Mathematics.
- **Codd, E. F.** (1970). _A relational model of data for large shared data banks._ Communications of the ACM.
- **Presburger, M.** (1929 / 1991 English trans.). _On the completeness of a certain system of arithmetic of whole numbers with addition._
- **Immerman, N.** (1999). _Descriptive Complexity._ Springer.
- **Grädel, E.; Otto, M.; Rosen, E.** (1997). _Two–variable logic with counting is decidable._ Bulletin of Symbolic Logic.
- **Zaheer, M. et al.** (2017). _Deep Sets._ NeurIPS.
- **Andreas, J. et al.** (2016). _Neural Module Networks._ CVPR.
- **Johnson, J. et al.** (2017). _CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning._ CVPR.
- **Kozen, D.** (1997). _Kleene Algebra with Tests._ ACM Transactions on Programming Languages and Systems.

## 12) Visual: Internal Reasoning Flow (ASCII)

```
[h, H] → VQ‑L1 → e1
   └──→ VQ‑L2([h,e1]) → e2
         └─→ VQ‑L3([h,e1,e2]) → e3
               └→ VQ‑L4([h,e1,e2,e3]) → e4

path = [e1,e2,e3,e4]
schema = map_family(e3)
expert = moe[schema](context=[e1,e2,e3])

STATE_init ── step: Δ=expert.apply(STATE, param=e4) ──> STATE_1 ── ... (STOP) ──> STATE_final
                               │
                               └─ gate g = σ(Wg[STATE,e1..e4]) and residual update

logits = decoder(STATE_final, path)
```
