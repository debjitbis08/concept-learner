from dataclasses import dataclass
import random, numpy as np, torch
import torch.nn.functional as F
from concept_learner.tokenizer import HFTokenizerWrapper
from concept_learner.model import CLModel
from concept_learner.reasoning_v2 import ReasonerV2


@dataclass
class CLConfig:
    d_model: int = 128
    rvq_dim: int = 64
    codebook_size: int = 24
    num_quantizers: int = 3
    num_classes: int = 3
    lambda_vq: float = 0.1
    lambda_stop: float = 0.1
    lr: float = 3e-4
    max_len: int = 24
    pretrained_tok: str = "bert-base-cased"
    device: str = "cpu"  # or set by setup()


def setup(seed: int = 0, device: str | None = None) -> str:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def make_synth_batch(tok, B=4, T=16, C=3, max_steps=4, device="cpu"):
    texts = ["2 : 3 :: 5 : ?", "color of banana ?", "banana banana ?", "is 3 > 2 ?"]
    ids, mask = [], []
    for i in range(B):
        e = tok.encode(texts[i % len(texts)], max_len=T)
        ids.append(e.ids)
        mask.append(e.mask)

    ids = torch.tensor(ids, dtype=torch.long, device=device)
    mask = torch.tensor(mask, dtype=torch.long, device=device)

    # random labels (for smoke testing)
    y_tok = torch.randint(0, C, (B, T), device=device)
    y_seq = torch.randint(0, C, (B,), device=device)

    # STOP: pick a random step for each example
    y_stop = torch.zeros(B, max_steps, dtype=torch.long, device=device)
    for i in range(B):
        stop_idx = torch.randint(0, max_steps, (1,)).item()
        y_stop[i, stop_idx] = 1

    return ids, mask, y_tok, y_seq, y_stop


def train_step(
    model: CLModel,
    batch,
    optimizer: torch.optim.Optimizer,
    lambda_vq: float = 0.1,
    lambda_stop: float = 0.1,
):
    ids, mask, y_tok, y_seq, y_stop = batch
    model.train()
    logits_tok, logits_seq, vq_loss, _, stop_logits, action_logits = model(ids, mask)

    # token loss (mask-aware)
    B, T, C = logits_tok.shape
    loss_tok_per_pos = F.cross_entropy(
        logits_tok.view(-1, C), y_tok.view(-1), reduction="none"
    ).view(B, T)
    mask_f = mask.float()
    loss_tok = (loss_tok_per_pos * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    # sequence loss
    loss_seq = F.cross_entropy(logits_seq, y_seq)

    # STOP loss
    stop_loss = F.binary_cross_entropy_with_logits(stop_logits, y_stop.float())

    # total
    loss = loss_tok + loss_seq + lambda_vq * vq_loss + lambda_stop * stop_loss

    # action loss
    num_ops = action_logits.size(-1)
    y_act = torch.randint(
        0, num_ops, (ids.size(0), action_logits.size(1)), device=ids.device
    )
    act_loss = F.cross_entropy(action_logits.reshape(-1, num_ops), y_act.view(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # convert STOP to probs for logging
    stop_probs = torch.sigmoid(stop_logits).detach().cpu().numpy().mean(axis=0)

    return {
        "loss": float(loss.item()),
        "tok": float(loss_tok.item()),
        "seq": float(loss_seq.item()),
        "vq": float(vq_loss.item()),
        "stop": float(stop_loss.item()),
        "stop_probs": stop_probs.tolist(),  # avg prob per step
    }


def train_epoch_wake_sleep(
    model: CLModel,
    tok: HFTokenizerWrapper,
    optimizer: torch.optim.Optimizer,
    steps: int = 100,
    batch_size: int = 8,
    T: int = 24,
    sleep_every: int = 20,
    wake_micro: bool = False,
):
    """Wake/Sleep/Dream training loop (minimal yet complete).

    - Wake: run standard training steps; Reasoner logs traces; optionally micro mode during wake.
    - Sleep-Abstraction: mine candidates from traces and install/prune in OpFunction.
    - Sleep-Dreaming: build a dataset of replays (from wake samples) and train controller with MAP (trace CE).
    """
    # Enable Reasoner wake/sleep and optional micro mode
    reasoner: ReasonerV2 = model.reasoner
    reasoner.wake_sleep = True
    if wake_micro:
        reasoner.exec_mode = "micro"
    device = next(model.parameters()).device
    log = {}
    # Simple replay buffer over wake traces
    rb = ReplayBuffer(max_items=256)

    for it in range(1, steps + 1):
        C = model.decoder.seq_head.out_features if hasattr(model.decoder, "seq_head") else 3
        ids, mask, y_tok, y_seq, y_stop = make_synth_batch(
            tok, B=batch_size, T=T, C=C, max_steps=reasoner.max_steps, device=device
        )
        out = train_step(
            model, (ids, mask, y_tok, y_seq, y_stop), optimizer, lambda_vq=0.1, lambda_stop=0.1
        )
        # collect wake traces into replay buffer
        if reasoner.wake_sleep and getattr(reasoner, "_last_traces", None) is not None:
            traces = reasoner._last_traces
            # convert variable-length action sequences to fixed-length with -1 padding
            total_actions = model.reasoner.to_action.out_features
            for bi, tr in enumerate(traces):
                # keep only action ids (exclude prim/func tag); map to global action id
                seq_ids = []
                for kind, idx in tr:
                    if kind == "prim":
                        seq_ids.append(int(idx))
                    else:
                        seq_ids.append(len(model.reasoner.prims) + int(idx))
                rb.add(ids[bi].detach().cpu(), mask[bi].detach().cpu(), seq_ids)
        if it % sleep_every == 0:
            installed = reasoner.sleep_abstraction()
            # Optional: switch back to macro after installing for stability
            if not wake_micro:
                reasoner.exec_mode = "macro"
            # Sleep-B: Dreaming (MAP over traces)
            if len(rb) > 0:
                # build a dream batch from replays only (no synthetic fantasies)
                dream_bs = batch_size
                replays = rb.sample(dream_bs)
                if replays:
                    ids_rep = torch.stack([x.ids for x in replays], dim=0).to(device)
                    mask_rep = torch.stack([x.mask for x in replays], dim=0).to(device)
                    traces_rep = [x.trace for x in replays]
                    # compress traces using current library (canonicalize to macro calls)
                    patterns = extract_primitive_patterns_from_library(reasoner)
                    traces_canon = [compress_trace_with_patterns(tr, patterns, len(reasoner.prims)) for tr in traces_rep]
                    # fabricate labels for token/seq heads to keep loss well-formed
                    y_tok = torch.zeros(ids_rep.size(0), T, dtype=torch.long, device=device)
                    y_seq = torch.zeros(ids_rep.size(0), dtype=torch.long, device=device)
                    # STOP target from trace lengths
                    y_stop = make_stop_targets_from_traces(traces_canon, max_steps=reasoner.max_steps, device=device)
                    # one optimization step with CE to target traces (MAP)
                    out_map = train_step_with_trace_supervision(
                        model, (ids_rep, mask_rep, y_tok, y_seq, y_stop), traces_canon, optimizer
                    )
                    out.update({"map": out_map.get("map", 0.0)})
                    # Optionally: structured combinations of wake samples to diversify traces without external data
                    combos = build_composite_from_replays(replays, target_count=max(1, dream_bs // 2), T=T, device=device)
                    if combos is not None and combos[0].size(0) > 0:
                        ids_comb, mask_comb, traces_comb = combos
                        # compress combined traces again to exploit cross-boundary patterns
                        traces_comb_canon = [compress_trace_with_patterns(tr, patterns, len(reasoner.prims)) for tr in traces_comb]
                        y_tok_c = torch.zeros(ids_comb.size(0), T, dtype=torch.long, device=device)
                        y_seq_c = torch.zeros(ids_comb.size(0), dtype=torch.long, device=device)
                        y_stop_c = make_stop_targets_from_traces(traces_comb_canon, max_steps=reasoner.max_steps, device=device)
                        out_map2 = train_step_with_trace_supervision(
                            model, (ids_comb, mask_comb, y_tok_c, y_seq_c, y_stop_c), traces_comb_canon, optimizer
                        )
                        out.update({"map_combo": out_map2.get("map", 0.0)})
            log = {**out, "installed": installed, "telem": getattr(reasoner, "_telemetry", {})}
    return log


# ------------------------------- DREAMING (REPLAY-ONLY) --------------------


@dataclass
class RBItem:
    ids: torch.Tensor  # (T,)
    mask: torch.Tensor  # (T,)
    trace: list[int]    # list of action ids


class ReplayBuffer:
    def __init__(self, max_items: int = 512):
        self.max_items = int(max_items)
        self.items: list[RBItem] = []

    def __len__(self):
        return len(self.items)

    def add(self, ids: torch.Tensor, mask: torch.Tensor, trace: list[int]):
        self.items.append(RBItem(ids=ids, mask=mask, trace=trace))
        if len(self.items) > self.max_items:
            self.items.pop(0)

    def sample(self, k: int) -> list[RBItem]:
        if len(self.items) == 0 or k <= 0:
            return []
        k = min(k, len(self.items))
        idx = np.random.choice(len(self.items), size=k, replace=False)
        return [self.items[i] for i in idx]


def make_stop_targets_from_traces(traces: list[list[int]], max_steps: int, device: str | torch.device) -> torch.Tensor:
    B = len(traces)
    y = torch.zeros(B, max_steps, dtype=torch.float32, device=device)
    for i, tr in enumerate(traces):
        stop_idx = min(len(tr) - 1, max_steps - 1) if len(tr) > 0 else 0
        y[i, stop_idx] = 1.0
    return y


def train_step_with_trace_supervision(
    model: CLModel,
    batch,
    traces: list[list[int]],
    optimizer: torch.optim.Optimizer,
    lambda_vq: float = 0.1,
    lambda_stop: float = 0.1,
):
    """Single optimization step with MAP objective over traces.

    Computes standard token/seq/vq/stop losses plus action cross-entropy against
    the provided target trace (per example). Traces are action-id sequences of
    length <= max_steps. Unused steps are ignored in the action loss.
    """
    ids, mask, y_tok, y_seq, y_stop = batch
    model.train()
    logits_tok, logits_seq, vq_loss, _, stop_logits, action_logits = model(ids, mask)

    # token loss (mask-aware)
    B, T, C = logits_tok.shape
    loss_tok_per_pos = F.cross_entropy(
        logits_tok.view(-1, C), y_tok.view(-1), reduction="none"
    ).view(B, T)
    mask_f = mask.float()
    loss_tok = (loss_tok_per_pos * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    # sequence loss
    loss_seq = F.cross_entropy(logits_seq, y_seq)

    # STOP loss
    stop_loss = F.binary_cross_entropy_with_logits(stop_logits, y_stop.float())

    # Action CE against target traces (MAP)
    B, S, A = action_logits.shape
    # Build targets with -100 for ignore positions beyond trace length
    y = torch.full((B, S), -100, dtype=torch.long, device=ids.device)
    for i, tr in enumerate(traces[:B]):
        for t, a in enumerate(tr[:S]):
            y[i, t] = int(a)
    action_loss = F.cross_entropy(action_logits.view(-1, A), y.view(-1), ignore_index=-100)

    loss = loss_tok + loss_seq + lambda_vq * vq_loss + lambda_stop * stop_loss + action_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "map": float(action_loss.item()),
        "tok": float(loss_tok.item()),
        "seq": float(loss_seq.item()),
        "vq": float(vq_loss.item()),
        "stop": float(stop_loss.item()),
    }


# --------------------------- TRACE REWRITES --------------------------------


def extract_primitive_patterns_from_library(reasoner: ReasonerV2) -> list[tuple[int, list[int]]]:
    """Return [(sid, [prim_idxs...])] for slots whose bodies are primitive-only.

    prim_idxs are indices in the primitives set (not global action ids).
    """
    pats: list[tuple[int, list[int]]] = []
    if not reasoner.use_functions or reasoner.op_function is None:
        return pats
    for sid, slot in enumerate(reasoner.op_function.slots):
        if len(slot.steps) == 0:
            continue
        prim_seq: list[int] = []
        ok = True
        for st in slot.steps:
            if st.kind == "primitive" and st.idx is not None:
                prim_seq.append(int(st.idx))
            elif st.kind == "return":
                break
            else:
                ok = False
                break
        if ok and len(prim_seq) >= 2:
            pats.append((sid, prim_seq))
    # Sort by decreasing length to enable greedy longest-match replacement
    pats.sort(key=lambda x: len(x[1]), reverse=True)
    return pats


def compress_trace_with_patterns(trace: list[int], patterns: list[tuple[int, list[int]]], num_prims: int) -> list[int]:
    """Replace primitive n-grams with function calls using greedy longest-match.

    trace is over GLOBAL action ids; num_prims is used to map function slot sid to global id.
    Only matches contiguous sequences of primitive actions.
    """
    if not trace:
        return []
    out: list[int] = []
    i = 0
    while i < len(trace):
        a = trace[i]
        # Only consider matches starting at primitive action
        if a < num_prims and patterns:
            matched = False
            for sid, prim_seq in patterns:
                L = len(prim_seq)
                if i + L <= len(trace) and all(trace[i + k] == prim_seq[k] for k in range(L)):
                    # replace with function action id
                    out.append(num_prims + sid)
                    i += L
                    matched = True
                    break
            if matched:
                continue
        # otherwise keep the original action
        out.append(a)
        i += 1
    return out


def build_composite_from_replays(replays: list[RBItem], target_count: int, T: int, device) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]] | None:
    """Create structured, random combinations of wake samples by concatenating two traces and inputs.

    - Inputs: concatenate token ids/masks of two replays (truncated to length T).
    - Traces: concatenate action sequences from both (global ids), truncated to max_steps later by CE trainer.
    """
    if not replays or target_count <= 0:
        return None
    # Randomly form pairs (with replacement if needed)
    items = []
    for _ in range(target_count):
        a = replays[random.randrange(0, len(replays))]
        b = replays[random.randrange(0, len(replays))]
        ids_c, mask_c = concat_ids_masks(a.ids.to(device), a.mask.to(device), b.ids.to(device), b.mask.to(device), T=T)
        trace_c = list(a.trace) + list(b.trace)
        items.append((ids_c, mask_c, trace_c))
    ids_batch = torch.stack([x[0] for x in items], dim=0)
    mask_batch = torch.stack([x[1] for x in items], dim=0)
    traces = [x[2] for x in items]
    return ids_batch, mask_batch, traces


def concat_ids_masks(ids1: torch.Tensor, mask1: torch.Tensor, ids2: torch.Tensor, mask2: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate two token sequences with masks and truncate/pad to length T.

    Assumes ids shape (T,) and mask shape (T,) with pad=0 entries at tail.
    """
    # Extract actual (non-pad) lengths
    L1 = int(mask1.sum().item())
    L2 = int(mask2.sum().item())
    take1 = min(L1, T)
    take2 = min(L2, max(0, T - take1))
    out_ids = torch.zeros(T, dtype=ids1.dtype, device=ids1.device)
    out_mask = torch.zeros(T, dtype=mask1.dtype, device=mask1.device)
    if take1 > 0:
        out_ids[:take1] = ids1[:take1]
        out_mask[:take1] = 1
    if take2 > 0:
        out_ids[take1 : take1 + take2] = ids2[:take2]
        out_mask[take1 : take1 + take2] = 1
    return out_ids, out_mask
