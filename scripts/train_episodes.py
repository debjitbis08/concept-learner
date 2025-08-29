import argparse
import os
import math
import torch
from contextlib import nullcontext
from typing import List
from concept_learner.episodes import EpisodeConfig, EpisodeGenerator
from concept_learner.compositions import (
    make_numeric_compositions,
    make_equality_compositions,
)
from concept_learner.model import CLModel
from concept_learner.proto_router import ProtoRouter
from concept_learner.tokenizer import HFTokenizerWrapper
from concept_learner.trainer import (
    ReplayBuffer,
    extract_primitive_patterns_from_library,
    compress_trace_with_patterns,
    build_composite_from_replays,
    make_stop_targets_from_traces,
    train_step_with_trace_supervision,
)


def _pack_pairs(a_desc, a_mask, b_desc, b_mask, device):
    # Shift digit tokens by +2 to reserve: 0=PAD, 1=CLS, 2=SEP
    a = a_desc.clone().to(device)
    b = b_desc.clone().to(device)
    am = a_mask.to(device)
    bm = b_mask.to(device)
    a[am] = a[am] + 2
    b[bm] = b[bm] + 2
    B, L = a.shape
    total = 1 + L + 1 + L
    ids = torch.zeros(B, total, dtype=torch.long, device=device)
    mask = torch.zeros(B, total, dtype=torch.long, device=device)
    # [CLS]
    ids[:, 0] = 1
    mask[:, 0] = 1
    # A
    ids[:, 1 : 1 + L] = a
    mask[:, 1 : 1 + L] = am.long()
    # [SEP]
    ids[:, 1 + L] = 2
    mask[:, 1 + L] = 1
    # B
    ids[:, 1 + L + 1 :] = b
    mask[:, 1 + L + 1 :] = bm.long()
    return ids, mask


def save_checkpoint(
    path, model, opt, step, best_acc: float | None = None, ema_state: dict | None = None, proto_state: dict | None = None, knn_state: dict | None = None
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}
    if best_acc is not None:
        payload["best_acc"] = float(best_acc)
    if ema_state is not None:
        payload["ema"] = ema_state
    if proto_state is not None:
        payload["proto"] = proto_state
    if knn_state is not None:
        payload["knn"] = knn_state
    torch.save(payload, path)


def load_checkpoint(path, model, opt, map_location=None, ema=None, proto: ProtoRouter | None = None, knn=None, expected_model_hash: str | None = None):
    ckpt = torch.load(path, map_location=map_location)
    sd = ckpt["model"]
    # expose best_acc to callers for resume logic
    load_checkpoint.last_best_acc = ckpt.get("best_acc", None)
    load_checkpoint.loaded_ema = False
    try:
        model.load_state_dict(sd)
    except RuntimeError as e:
        msg = str(e)

        # Backward-compat: expand/trim embedding and classifier heads
        def _resize_like(key: str, ref_param: torch.nn.Parameter):
            old = sd.get(key)
            if old is None:
                return
            new = ref_param.data
            if tuple(old.shape) == tuple(new.shape):
                return
            # handle linear weight [out, in] and bias [out]
            if old.ndim == 2 and new.ndim == 2 and old.shape[1] == new.shape[1]:
                out = new.detach().clone()
                out.zero_()
                rows = min(old.shape[0], new.shape[0])
                out[:rows].copy_(old[:rows])
                sd[key] = out
            elif old.ndim == 1 and new.ndim == 1:
                out = new.detach().clone()
                out.zero_()
                rows = min(old.shape[0], new.shape[0])
                out[:rows].copy_(old[:rows])
                sd[key] = out

        if "enc.tok_emb.weight" in msg:
            _resize_like("enc.tok_emb.weight", model.enc.tok_emb.weight)
        if any(
            s in msg
            for s in [
                "head.weight",
                "decoder.token_head.weight",
                "decoder.seq_head.weight",
            ]
        ):
            # adjust all known heads to current shapes
            _resize_like("head.weight", model.head.weight)
            _resize_like("head.bias", model.head.bias)
            _resize_like("decoder.token_head.weight", model.decoder.token_head.weight)
            _resize_like("decoder.token_head.bias", model.decoder.token_head.bias)
            _resize_like("decoder.seq_head.weight", model.decoder.seq_head.weight)
            _resize_like("decoder.seq_head.bias", model.decoder.seq_head.bias)
        # try loading with relaxed strictness after adjustments
        model.load_state_dict(sd, strict=False)
    if opt is not None and "opt" in ckpt:
        try:
            opt.load_state_dict(ckpt["opt"])
            # repair optimizer state if embedding size changed
            try:
                tok_param = model.enc.tok_emb.weight
                st = opt.state.get(tok_param, None)
                if isinstance(st, dict):
                    for k in ("exp_avg", "exp_avg_sq"):
                        if k in st:
                            buf = st[k]
                            if (
                                isinstance(buf, torch.Tensor)
                                and buf.shape != tok_param.data.shape
                            ):
                                new_buf = torch.zeros_like(tok_param.data)
                                rows = min(buf.shape[0], new_buf.shape[0])
                                new_buf[:rows].copy_(buf[:rows])
                                st[k] = new_buf
                # also repair classifier head states if shapes changed
                for p in [
                    model.head.weight,
                    model.head.bias,
                    model.decoder.token_head.weight,
                    model.decoder.token_head.bias,
                    model.decoder.seq_head.weight,
                    model.decoder.seq_head.bias,
                ]:
                    stp = opt.state.get(p, None)
                    if isinstance(stp, dict):
                        for k in ("exp_avg", "exp_avg_sq"):
                            if (
                                k in stp
                                and isinstance(stp[k], torch.Tensor)
                                and stp[k].shape != p.data.shape
                            ):
                                newb = torch.zeros_like(p.data)
                                rows = min(stp[k].shape[0], newb.shape[0])
                                if stp[k].ndim == 2 and newb.ndim == 2:
                                    cols = min(stp[k].shape[1], newb.shape[1])
                                    newb[:rows, :cols].copy_(stp[k][:rows, :cols])
                                else:
                                    newb[:rows].copy_(stp[k][:rows])
                                stp[k] = newb
            except Exception:
                # if anything goes wrong, we proceed without repairing
                pass
        except Exception:
            pass
    # optionally restore EMA state
    try:
        if ema is not None and isinstance(ckpt.get("ema"), dict):
            ema.load_state_dict(ckpt["ema"])
            load_checkpoint.loaded_ema = True
    except Exception:
        pass
    try:
        if proto is not None and isinstance(ckpt.get("proto"), dict):
            proto.load_state_dict(ckpt["proto"])  # type: ignore[arg-type]
    except Exception:
        pass
    # optionally restore kNN memory snapshot if hash matches
    try:
        if knn is not None and isinstance(ckpt.get("knn"), dict):
            state = ckpt["knn"]
            mh = state.get("model_hash", None)
            if expected_model_hash is None or (mh is not None and mh == expected_model_hash):
                knn.load_snapshot(state, expected_hash=expected_model_hash)
    except Exception:
        pass
    return int(ckpt.get("step", 0))


@torch.no_grad()
def _quick_eval(
    model: CLModel,
    gen: EpisodeGenerator,
    tok: HFTokenizerWrapper,
    device: str,
    batch_size: int,
    eval_batches: int,
    max_len: int,
    num_numbers: int,
    relations: str | None = None,
    idx_range: tuple[int, int] | None = None,
    template_filter: dict | None = None,
    probe=None,
    probe_alpha: float | None = None,
    proto: ProtoRouter | None = None,
    proto_space: str = "typed",
    knn=None,
    knn_space: str = "typed",
    knn_alpha: float | None = None,
    knn_alpha_end: float | None = None,
) -> float:
    model.eval()
    total, correct = 0, 0
    for _ in range(eval_batches):
        rel_ids = _parse_relations_arg(relations)
        batch = gen.sample_posneg_pairs(batch=batch_size, allowed_relations=rel_ids, idx_range=idx_range)
        ids, mask = _pack_pair_questions_text(batch, tok, max_len, template_filter=template_filter)
        y = batch["label"].to(device)
        logits_tok, logits_seq, vq_loss, indices, stop_logits, action_logits = model(ids, mask)
        # Optional prototypical routing bias on YES/NO for pair relations
        if proto is not None:
            with torch.no_grad():
                # Build normalized features per requested space
                h_pool, _ = model.enc(ids, mask)
                # s_final is exposed on reasoner last forward as return value
                # Model.forward returns s_final in slot 2; keep consistent with above
                # We recompute it via last call: we already captured return values.
                # Use reasoner's _last_val? Stick to returned tensors to avoid ambiguity.
                # logits_tok, logits_seq, vq_loss, indices, stop_logits, action_logits = ...
                # We need s_final but model.forward returns it at index 2; to avoid interface changes,
                # we read it from local variable where available via closure — not possible.
                # Instead, reuse model.reasoner._last_s if present, else skip.
                s_typed = getattr(model.reasoner, "_last_s", None)
                if s_typed is None:
                    # Fallback: approximate with h_pool if typed state not exposed
                    s_typed = h_pool
                feat = ProtoRouter.build_feature(h_pool, s_typed, proto_space)
                YES_IDX = num_numbers
                NO_IDX = num_numbers + 1
                # Apply bias; do not update in eval
                logits_seq = proto.apply_bias(logits_seq, feat, batch["rel"], YES_IDX, NO_IDX, train_mode=False)
        NO_IDX = num_numbers + 1
        YES_IDX = num_numbers
        # Optional kNN posterior mixing on full classes
        if knn is not None:
            with torch.no_grad():
                h_pool, _ = model.enc(ids, mask)
                s_typed = getattr(model.reasoner, "_last_s", None)
                if s_typed is None:
                    s_typed = h_pool
                feat = ProtoRouter.build_feature(h_pool, s_typed, knn_space)
                p_model = torch.softmax(logits_seq, dim=-1)
                p_knn = knn.posterior(feat)
                a = float(knn_alpha if knn_alpha is not None else 0.2)
                a_end = float(knn_alpha_end if knn_alpha_end is not None else a)
                # Use provided alpha (caller may pass decayed value)
                p = (a * p_model + (1.0 - a) * p_knn).clamp_min(1e-8)
                logits_seq = torch.log(p)
                # safety: sanitize logits before downstream use
                logits_seq = torch.nan_to_num(logits_seq, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        if probe is not None and getattr(probe, "w", None) is not None:
            # Blend learned head with ridge probe
            with torch.no_grad():
                h, _ = model.enc(ids, mask)
                ridge_logit = probe.predict_logit(h)
                from concept_learner.utils import select_pair_columns_safe
                base_pair = select_pair_columns_safe(logits_seq, NO_IDX, YES_IDX, order="ab")
                learned_logit = base_pair[:, 1] - base_pair[:, 0]
                a = float(probe_alpha if probe_alpha is not None else getattr(probe, "alpha", 0.7))
                blended = a * learned_logit + (1.0 - a) * ridge_logit
                # decision by sign of blended logit
                pred_bin = (blended > 0).long()
        else:
            from concept_learner.utils import select_pair_columns_safe
            pred_bin = select_pair_columns_safe(logits_seq, NO_IDX, YES_IDX, order="ab").argmax(dim=-1)
        correct += (pred_bin == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(1, total)


# --------------------------- Ridge Probe (Few-Shot) -------------------------


class RidgeProbe:
    def __init__(self, d: int, max_items: int = 2048, lam: float = 1e-3, alpha: float = 0.7, device: str | torch.device = "cpu"):
        self.max_items = int(max_items)
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.d = int(d)
        self.device = device
        self.X: list[torch.Tensor] = []  # list of (n_i, d)
        self.Y: list[torch.Tensor] = []  # list of (n_i,)
        self.w: torch.Tensor | None = None  # (d,)

    def __len__(self):
        return sum(x.size(0) for x in self.X)

    def add(self, x: torch.Tensor, y: torch.Tensor):
        if x.numel() == 0:
            return
        # detach to CPU for storage
        self.X.append(x.detach().to("cpu").float())
        self.Y.append(y.detach().to("cpu").float())
        # trim to ring size
        while len(self) > self.max_items and len(self.X) > 0:
            # pop oldest chunk
            drop = self.X[0].size(0)
            self.X.pop(0)
            self.Y.pop(0)

    def fit(self, device: str | torch.device):
        if len(self.X) == 0:
            self.w = None
            return False
        X = torch.cat(self.X, dim=0)  # (N,d)
        Y = torch.cat(self.Y, dim=0)  # (N,)
        if X.size(0) < self.d:
            self.w = None
            return False
        # Closed-form ridge: w = (X^T X + lam I)^{-1} X^T y
        Xt = X.t()  # (d,N)
        XtX = Xt @ X  # (d,d)
        lamI = self.lam * torch.eye(self.d, dtype=X.dtype, device=X.device)
        Xty = Xt @ Y  # (d,)
        try:
            w = torch.linalg.solve(XtX + lamI, Xty)
        except RuntimeError:
            w = torch.linalg.pinv(XtX + lamI) @ Xty
        self.w = w.to(device)
        self.device = device
        return True

    @torch.no_grad()
    def predict_logit(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B,d) on any device; returns (B,) ridge logit
        if self.w is None:
            return torch.zeros(h.size(0), device=h.device)
        w = self.w.to(h.device)
        return (h @ w)


def _pack_text_batch(
    texts: List[str], tok: HFTokenizerWrapper, max_len: int, device: str
):
    B = len(texts)
    ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
    mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, t in enumerate(texts):
        enc = tok.encode(t, max_len=max_len)
        ids[i] = torch.tensor(enc.ids[:max_len], dtype=torch.long, device=device)
        mask[i] = torch.tensor(enc.mask[:max_len], dtype=torch.long, device=device)
    return ids, mask


def _pair_templates(a: int, b: int, r: int) -> List[str]:
    """Return a mix of natural-language and symbolic templates for pair relations.
    Keeps variety to better match eval phrasing and symbolic forms.
    """
    if r == 0:  # same_parity
        return [
            f"Do {a} and {b} have the same parity?",
            f"Is parity({a}) equal to parity({b})?",
            f"Is {a}%2 == {b}%2?",
        ]
    if r == 1:  # successor
        return [
            f"Is {b} the successor of {a}?",
            f"Is {b} equal to {a} + 1?",
            f"Is {b} == {a}+1?",
            f"{b} = {a} + 1?",
        ]
    if r == 2:  # predecessor
        return [
            f"Is {b} the predecessor of {a}?",
            f"Is {b} equal to {a} - 1?",
            f"Is {b} == {a}-1?",
            f"{b} = {a} - 1?",
        ]
    if r == 3:  # add_2
        return [
            f"Is {b} equal to {a} + 2?",
            f"Is {b} == {a}+2?",
            f"{b} = {a} + 2?",
        ]
    if r == 4:  # same_tens
        return [
            f"Do {a} and {b} have the same tens digit?",
            f"Is tens({a}) equal to tens({b})?",
            f"tens({a}) == tens({b})?",
        ]
    if r == 5:  # same_ones
        return [
            f"Do {a} and {b} have the same ones digit?",
            f"Is ones({a}) equal to ones({b})?",
            f"{a}%10 == {b}%10?",
        ]
    if r == 6:  # makes_ten (ones digits sum to 10)
        return [
            f"Do the ones digits of {a} and {b} make ten?",
            f"Is ones({a}) + ones({b}) equal to 10?",
            f"({a}%10 + {b}%10) == 10?",
        ]
    if r == 7:  # greater
        return [
            f"Is {a} greater than {b}?",
            f"Is {a} larger than {b}?",
            f"Is {a} > {b}?",
            f"{a}>{b}?",
            f"{a} > {b}?",
        ]
    if r == 8:  # smaller
        return [
            f"Is {a} smaller than {b}?",
            f"Is {a} less than {b}?",
            f"Is {a} < {b}?",
            f"{a}<{b}?",
            f"{a} < {b}?",
        ]
    return [f"Do {a} and {b} satisfy the relation?"]


def _pack_pair_questions_text(batch, tok: HFTokenizerWrapper, max_len: int, template_filter: dict | None = None):
    import random

    a = batch["a_idx"].tolist()
    b = batch["b_idx"].tolist()
    r = batch["rel"].tolist()
    texts = []
    tf = template_filter
    for ai, bi, ri in zip(a, b, r):
        tpls = _pair_templates(ai, bi, ri)
        if isinstance(tf, dict) and ri in tf and len(tf[ri]) > 0:
            idxs = [i for i in tf[ri] if 0 <= i < len(tpls)]
            if len(idxs) == 0:
                choice = random.choice(tpls)
            else:
                choice = tpls[random.choice(idxs)]
        else:
            choice = random.choice(tpls)
        texts.append(choice)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _pack_text_batch(texts, tok, max_len, device)


def _pack_pair_questions_text_with_tpl(batch, tok: HFTokenizerWrapper, max_len: int, template_filter: dict | None = None):
    """Same as _pack_pair_questions_text but also returns the chosen template id per item."""
    import random
    a = batch["a_idx"].tolist()
    b = batch["b_idx"].tolist()
    r = batch["rel"].tolist()
    texts = []
    tpl_ids = []
    tf = template_filter
    for ai, bi, ri in zip(a, b, r):
        tpls = _pair_templates(ai, bi, ri)
        if isinstance(tf, dict) and ri in tf and len(tf[ri]) > 0:
            idxs = [i for i in tf[ri] if 0 <= i < len(tpls)]
            if len(idxs) == 0:
                tidx = random.randrange(len(tpls))
            else:
                tidx = random.choice(idxs)
        else:
            tidx = random.randrange(len(tpls))
        texts.append(tpls[tidx])
        tpl_ids.append(int(tidx))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids, mask = _pack_text_batch(texts, tok, max_len, device)
    return ids, mask, torch.tensor(tpl_ids, dtype=torch.long, device=device)


def _pack_parity_text(a: torch.Tensor, tok: HFTokenizerWrapper, max_len: int):
    import random
    texts = []
    for ai in a.tolist():
        texts.append(
            random.choice(
                [
                    f"Is {ai} even or odd?",
                    f"Parity of {ai}?",
                    f"Is the number {ai} even or odd?",
                ]
            )
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _pack_text_batch(texts, tok, max_len, device)


def _pack_offset_text(a: torch.Tensor, off: torch.Tensor, tok: HFTokenizerWrapper, max_len: int):
    import random
    texts = []
    for ai, oi in zip(a.tolist(), off.tolist()):
        if int(oi) >= 0:
            k = int(oi)
            choices = [
                f"What is {ai}+{k}?",
                f"{ai} + {k} = ?",
                f"Compute {ai} + {k}.",
            ]
        else:
            k = abs(int(oi))
            choices = [
                f"What is {ai}-{k}?",
                f"{ai} - {k} = ?",
                f"Compute {ai} - {k}.",
            ]
        texts.append(random.choice(choices))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _pack_text_batch(texts, tok, max_len, device)


def _pack_count_examples_text(kind, a, c, tok: HFTokenizerWrapper, max_len: int):
    import random

    kind_l = kind.tolist()
    a_l = a.tolist()
    c_l = c.tolist()
    texts = []
    for k, ai, ci in zip(kind_l, a_l, c_l):
        if k == 0:
            texts.append(
                random.choice(
                    [
                        f"What comes after {ai}?",
                        f"What is the successor of {ai}?",
                        f"Next number after {ai}?",
                    ]
                )
            )
        elif k == 1:
            texts.append(
                random.choice(
                    [
                        f"What comes before {ai}?",
                        f"What is the predecessor of {ai}?",
                        f"Previous number before {ai}?",
                    ]
                )
            )
        else:
            texts.append(
                random.choice(
                    [
                        f"What number comes between {ai} and {ci}?",
                        f"Which number is between {ai} and {ci}?",
                    ]
                )
            )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _pack_text_batch(texts, tok, max_len, device)


def _pack_place_value_text(kind, a, tok: HFTokenizerWrapper, max_len: int, face_place=None):
    import random
    kind_l = kind.tolist()
    a_l = a.tolist()
    texts = []
    fp_l = None
    if face_place is not None:
        try:
            fp_l = face_place.tolist()
        except Exception:
            fp_l = None
    for idx, (k, ai) in enumerate(zip(kind_l, a_l)):
        if k == 0:
            texts.append(random.choice([
                f"What is the ones digit of {ai}?",
                f"Ones digit of {ai}?",
                f"Which digit is in the ones place of {ai}?",
            ]))
        elif k == 1:
            texts.append(random.choice([
                f"What is the tens digit of {ai}?",
                f"Tens digit of {ai}?",
                f"Which digit is in the tens place of {ai}?",
            ]))
        elif k == 2:
            texts.append(random.choice([
                f"What is the place value of the tens digit of {ai}?",
                f"Place value of tens digit in {ai}?",
                f"What value does the tens place contribute in {ai}?",
            ]))
        else:
            # face value of a specific place (ones/tens)
            place = None
            if fp_l is not None:
                v = fp_l[idx]
                place = "ones" if v == 0 else ("tens" if v == 1 else None)
            if place is None:
                place = random.choice(["ones", "tens"])
            texts.append(random.choice([
                f"What is the face value of the {place} digit of {ai}?",
                f"Face value of the {place} digit in {ai}?",
                f"What is the face value for the {place} place in {ai}?",
            ]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _pack_text_batch(texts, tok, max_len, device)


def _pack_num_equality_text(a: torch.Tensor, b: torch.Tensor, tok: HFTokenizerWrapper, max_len: int):
    import random
    a_l = a.tolist()
    b_l = b.tolist()
    texts = []
    for ai, bi in zip(a_l, b_l):
        texts.append(
            random.choice(
                [
                    f"Is {ai} equal to {bi}?",
                    f"Are {ai} and {bi} equal?",
                    f"{ai} = {bi}?",
                    f"{ai} == {bi}?",
                    f"Does {ai} equal {bi}?",
                ]
            )
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _pack_text_batch(texts, tok, max_len, device)


def _pack_symbolic_compare_text(a: torch.Tensor, b: torch.Tensor, op: torch.Tensor, tok: HFTokenizerWrapper, max_len: int):
    import random
    a_l = a.tolist(); b_l = b.tolist(); op_l = op.tolist()
    texts = []
    for ai, bi, oi in zip(a_l, b_l, op_l):
        if oi == 0:  # '>'
            candidates = [
                # natural-language
                f"Is {ai} greater than {bi}?",
                f"Is {ai} larger than {bi}?",
                f"Is {ai} more than {bi}?",
                f"Does {ai} exceed {bi}?",
                # symbolic
                f"{ai}>{bi}?",
                f"{ai} > {bi}?",
                f"Is {ai} > {bi}?",
            ]
        else:  # '<'
            candidates = [
                # natural-language
                f"Is {ai} smaller than {bi}?",
                f"Is {ai} less than {bi}?",
                f"Is {ai} below {bi}?",
                f"Is {ai} under {bi}?",
                # symbolic
                f"{ai}<{bi}?",
                f"{ai} < {bi}?",
                f"Is {ai} < {bi}?",
            ]
        texts.append(random.choice(candidates))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _pack_text_batch(texts, tok, max_len, device)


def _pack_addition_text(a: torch.Tensor, b: torch.Tensor, tok: HFTokenizerWrapper, max_len: int):
    import random
    a_l = a.tolist(); b_l = b.tolist()
    texts = []
    for ai, bi in zip(a_l, b_l):
        candidates = [
            f"{ai}+{bi}=?",
            f"{ai} + {bi} = ?",
            f"What is {ai}+{bi}?",
            f"What is {ai} + {bi}?",
            f"What is the sum of {ai} and {bi}?",
            f"Compute {ai} + {bi}.",
            f"{ai} plus {bi} equals ?",
        ]
        texts.append(random.choice(candidates))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _pack_text_batch(texts, tok, max_len, device)


def _parse_relations_arg(rel_arg: str | None):
    """Parse --relations for pair relations (0..8). Returns list of ids or None (all)."""
    if not rel_arg or rel_arg.strip().lower() == "all":
        return None
    name_to_id = {
        "same_parity": 0,
        "successor": 1,
        "predecessor": 2,
        "add_2": 3,
        "same_tens": 4,
        "same_ones": 5,
        "makes_ten": 6,
        "greater": 7,
        "smaller": 8,
    }
    ids = []
    for part in rel_arg.split(","):
        s = part.strip().lower()
        if s == "":
            continue
        if s.isdigit():
            i = int(s)
            if 0 <= i <= 8:
                ids.append(i)
        elif s in name_to_id:
            ids.append(name_to_id[s])
        else:
            # ignore non-pair names here (e.g., place_value, face_value)
            pass
    ids = sorted(set(ids))
    return ids if len(ids) > 0 else None


def _pack_equality_examples_text(
    a: torch.Tensor, kind: torch.Tensor, tok: HFTokenizerWrapper, max_len: int
):
    import random

    a_l = a.tolist()
    k_l = kind.tolist()
    texts = []
    for ai, ki in zip(a_l, k_l):
        if ki == 0:
            rhs = ai + 1
            texts.append(
                random.choice(
                    [
                        f"Is the successor of {ai} equal to {rhs}?",
                        f"Is next of {ai} = {ai} + 1?",
                        f"Does {ai} + 1 equal the successor of {ai}?",
                    ]
                )
            )
        else:
            rhs = ai - 1
            texts.append(
                random.choice(
                    [
                        f"Is the predecessor of {ai} equal to {rhs}?",
                        f"Is previous of {ai} = {ai} - 1?",
                        f"Does {ai} - 1 equal the predecessor of {ai}?",
                    ]
                )
            )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids, mask = _pack_text_batch(texts, tok, max_len, device)
    y = torch.ones(len(texts), dtype=torch.long, device=device)  # all true
    return ids, mask, y


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
            new_avg = (1.0 - d) * param.detach() + d * self.shadow[name]
            self.shadow[name] = new_avg

    def store(self, model: torch.nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": {k: v.clone().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict):
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow", {})
        self.shadow = {k: v.clone() for k, v in shadow.items()}

    def reset(self, model: torch.nn.Module):
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def reconcile(self, model: torch.nn.Module):
        """
        Align EMA shadow tensors to current model parameter shapes/devices.
        If a shadow entry is missing or has a mismatched shape, reinitialize it
        from the current parameter value to avoid shape errors after architecture changes.
        Removes shadow entries that no longer correspond to any parameter.
        """
        new_shadow = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            cur = self.shadow.get(name)
            if cur is None or tuple(cur.shape) != tuple(param.shape):
                new_shadow[name] = param.detach().clone()
            else:
                # ensure device matches
                new_shadow[name] = cur.detach().to(param.device).clone()
        self.shadow = new_shadow


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    min_base = 10 if getattr(args, "base10", False) else args.min_base
    max_base = 10 if getattr(args, "base10", False) else args.max_base
    ecfg = EpisodeConfig(
        max_number=args.max_number,
        max_len=args.max_len,
        min_base=min_base,
        max_base=max_base,
        device=device,
    )
    gen = EpisodeGenerator(ecfg)

    tok = HFTokenizerWrapper("bert-base-cased")
    try:
        print(f"Tokenizer backend: {getattr(tok, 'backend', 'unknown')} ({getattr(tok, 'backend_name', 'n/a')}), vocab_size={tok.vocab_size}")
    except Exception:
        pass
    try:
        print(f"Tokenizer backend: {getattr(tok, 'backend', 'unknown')} ({getattr(tok, 'backend_name', 'n/a')}), vocab_size={tok.vocab_size}")
    except Exception:
        pass
    vocab_size = tok.vocab_size
    # unified head: numbers (0..N-1) + {YES, NO, EVEN, ODD}
    num_numbers = ecfg.max_number
    YES_IDX = num_numbers
    NO_IDX = num_numbers + 1
    EVEN_IDX = num_numbers + 2
    ODD_IDX = num_numbers + 3
    model = CLModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_classes=num_numbers + 4,
        pad_id=0,
        cls_id=1,
        max_len=args.max_len,
        vq_dim=args.vq_dim,
        vq_codebook_size=args.vq_codebook_size,
        vq_num_quantizers=(args.parallel_heads + args.serial_heads),
        vq_num_parallel_heads=args.parallel_heads,
        vq_serial_codebook_size=args.serial_codebook_size,
        vq_commitment_weight=args.vq_commitment_weight,
        vq_pre_vq_noise_std=args.pre_vq_noise_std,
        vq_orth_weight=args.vq_orth_weight,
        vq_entropy_weight=args.vq_entropy_weight,
    ).to(device)
    # Compute encoder/VQ hash for optional kNN snapshot validity
    import hashlib as _hashlib
    def _encoder_hash(m: CLModel) -> str:
        h = _hashlib.sha256()
        try:
            sd = m.state_dict()
            keys = [k for k in sd.keys() if k.startswith("enc.") or k.startswith("rvq.")]
            for k in sorted(keys):
                t = sd[k].detach().cpu().contiguous()
                h.update(k.encode()); h.update(bytes(t.shape)); h.update(t.numpy().tobytes())
        except Exception:
            for n, p in m.named_parameters():
                if n.startswith("enc.") or n.startswith("rvq."):
                    h.update(n.encode()); h.update(str(float(p.detach().sum().item())).encode())
        return h.hexdigest()
    model_hash = _encoder_hash(model)
    # Wire reasoner knobs
    try:
        r = model.reasoner
        r.max_steps = int(getattr(args, "max_steps", r.max_steps))
        r.epsilon = float(getattr(args, "epsilon", r.epsilon))
        r.topk_actions = int(getattr(args, "topk_actions", r.topk_actions))
        r.action_entropy_weight = float(getattr(args, "lambda_act_entropy", r.action_entropy_weight))
        r.stop_bias = float(getattr(args, "stop_bias", r.stop_bias))
        r.mine_min_support = int(getattr(args, "mine_min_support", r.mine_min_support))
        r.mine_max_len = int(getattr(args, "mine_max_len", r.mine_max_len))
        r.mine_min_body_len = int(getattr(args, "mine_min_body_len", r.mine_min_body_len))
        r.mine_min_effect = float(getattr(args, "mine_min_effect", r.mine_min_effect))
        r.mine_cosine_max = float(getattr(args, "mine_cosine_max", r.mine_cosine_max))
        r.mine_overlap_max = float(getattr(args, "mine_overlap_max", r.mine_overlap_max))
        r.fn_bias_init = float(getattr(args, "fn_bias_init", r.fn_bias_init))
        r.fn_bias_decay = float(getattr(args, "fn_bias_decay", r.fn_bias_decay))
    except Exception:
        pass
    # Enable wake/sleep by default, micro-exec as default training mode
    if int(getattr(args, "wake_sleep", 1)) > 0:
        try:
            model.reasoner.wake_sleep = True
            model.reasoner.exec_mode = getattr(args, "exec_mode", "micro")
        except Exception:
            pass
    # opt = torch.optim.AdamW(
    #     model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    # )

    # Optimizer with separate group for controller heads (action/STOP)
    head_decay, head_no_decay, decay, no_decay = [], [], [], []
    head_param_ids = set()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = n.startswith("reasoner.to_action") or n.startswith("reasoner.to_stop")
        if is_head:
            head_param_ids.add(id(p))
            if n.endswith(".bias"):
                head_no_decay.append(p)
            else:
                head_decay.append(p)
            continue
        if n.endswith(".bias") or "norm" in n.lower() or "codebook" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = []
    if len(head_decay) > 0:
        param_groups.append({"params": head_decay, "weight_decay": args.weight_decay, "is_head": True})
    if len(head_no_decay) > 0:
        param_groups.append({"params": head_no_decay, "weight_decay": 0.0, "is_head": True})
    if len(decay) > 0:
        param_groups.append({"params": decay, "weight_decay": args.weight_decay})
    if len(no_decay) > 0:
        param_groups.append({"params": no_decay, "weight_decay": 0.0})
    opt = torch.optim.AdamW(param_groups, lr=args.lr)

    # Optional AMP (mixed precision) for memory savings
    use_amp = bool(getattr(args, "amp", False)) and (str(device).startswith("cuda") and torch.cuda.is_available())
    amp_ctx = (lambda: torch.amp.autocast("cuda")) if use_amp else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Ensure knn is defined before potential checkpoint loading that may reference it
    knn = None

    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        start_step = load_checkpoint(
            args.resume, model, opt, map_location=device, ema=None, proto=None, knn=knn, expected_model_hash=model_hash
        )
        print(f"Resumed from {args.resume} at step {start_step}")
        # initialize best_acc from checkpoint if present; else from a quick eval
        ckpt_best = getattr(load_checkpoint, "last_best_acc", None)
        if ckpt_best is not None:
            best_acc = float(ckpt_best)
            print(f"Loaded best_acc from checkpoint: {best_acc:.3f}")
        else:
            def parse_range(s):
                if not s:
                    return None
                try:
                    lo, hi = s.split("-")
                    return (int(lo), int(hi))
                except Exception:
                    return None
            # Compute baseline components for selection metric
            tr_range = parse_range(getattr(args, "train_range", None))
            od_range = parse_range(getattr(args, "ood_range", None))
            val_pair0 = _quick_eval(
                model,
                gen,
                tok,
                device,
                batch_size=min(args.batch_size, 128),
                eval_batches=10,
                max_len=args.max_len,
                num_numbers=num_numbers,
                relations=getattr(args, "relations", None),
                idx_range=tr_range,
                template_filter=tmpl_train,
                proto=None,
                proto_space=str(getattr(args, "proto_space", "typed")),
            )
            # quick counting eval baseline
            model.eval()
            tot0, hit0 = 0, 0
            with torch.no_grad():
                for _ in range(5):
                    d2 = gen.sample_counting(batch=min(args.batch_size, 128), idx_range=tr_range)
                    i2, m2 = _pack_count_examples_text(
                        d2["kind"], d2["a"], d2["c"], tok, max_len=args.max_len
                    )
                    y2 = d2["target"].to(device)
                    _, ls2, _, _, _, _ = model(i2, m2)
                    pred2 = ls2[:, :num_numbers].argmax(dim=-1)
                    hit0 += (pred2 == y2).sum().item()
                    tot0 += y2.numel()
            model.train()
            val_cnt0 = hit0 / max(1, tot0)
            avg0 = 0.5 * (val_pair0 + val_cnt0)
            # optional OOD slice
            try:
                import math as _m
                if od_range is not None:
                    val_range0 = _quick_eval(
                        model,
                        gen,
                        tok,
                        device,
                        batch_size=min(args.batch_size, 128),
                        eval_batches=5,
                        max_len=args.max_len,
                        num_numbers=num_numbers,
                        relations=getattr(args, "relations", None),
                        idx_range=od_range,
                        template_filter=tmpl_train,
                        proto=None,
                        proto_space=str(getattr(args, "proto_space", "typed")),
                    )
                else:
                    val_range0 = float("nan")
            except Exception:
                val_range0 = float("nan")
            sel0 = avg0
            if od_range is not None and not (isinstance(val_range0, float) and (val_range0 != val_range0)):
                sel0 = (val_pair0 + val_cnt0 + float(val_range0)) / 3.0
            best_acc = float(sel0)
            try:
                msg = f"Computed baseline selection after resume: sel={best_acc:.3f} (avg={avg0:.3f}"
                if od_range is not None:
                    msg += f", range={float(val_range0):.3f})"
                else:
                    msg += ")"
                print(msg)
            except Exception:
                print(f"Computed baseline selection after resume: {best_acc:.3f}")

    # optional EMA of parameters
    ema = None
    if getattr(args, "ema_decay", 0.0) and args.ema_decay > 0.0:
        ema = ExponentialMovingAverage(model, decay=args.ema_decay)
        if args.resume and os.path.isfile(args.resume):
            try:
                ck = torch.load(args.resume, map_location=device)
                if isinstance(ck.get("ema"), dict):
                    ema.load_state_dict(ck["ema"])
                    # reconcile shadow shapes with current model (e.g., class count changed)
                    ema.reconcile(model)
                else:
                    ema.reset(model)
            except Exception:
                ema.reset(model)

    # schedulers
    def set_cosine_lr(step_idx: int):
        warmup = max(1, int(args.steps * args.warmup_ratio))
        if step_idx < warmup:
            lr = args.lr * float(step_idx + 1) / float(warmup)
        else:
            # optional single warm restart
            total = max(1, args.steps - warmup)
            if getattr(args, 'cosine_restart_step', 0) and args.cosine_restart_step > 0:
                period = int(args.cosine_restart_step)
                t_idx = (step_idx - warmup) % max(1, period)
                t = t_idx / max(1.0, period)
            else:
                t = (step_idx - warmup) / float(total)
                t = min(max(t, 0.0), 1.0)
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * t))
            lr = args.min_lr + (args.lr - args.min_lr) * cos_factor
        for g in opt.param_groups:
            g["lr"] = lr

    plateau_scheduler = None
    if args.sched == "plateau":
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
            threshold=1e-4,
        )

    best_acc = -1.0 if not (args.resume and os.path.isfile(args.resume)) else best_acc
    # Template holdout split (for template-OOD)
    def build_template_filters(holdout_ratio: float = 0.0):
        train_filt, ood_filt = {}, {}
        for r in range(9):
            tlist = _pair_templates(3, 5, r)
            k = len(tlist)
            if k <= 1:
                train_filt[r] = [0]
                ood_filt[r] = []
                continue
            k_tr = max(1, int(k * (1.0 - holdout_ratio)))
            train_filt[r] = list(range(0, k_tr))
            ood_filt[r] = list(range(k_tr, k))
        return train_filt, ood_filt
    tmpl_train, tmpl_ood = build_template_filters(getattr(args, "template_holdout", 0.0))
    model.train()
    seq_len = args.max_len
    # Wake/Sleep/Dream replay buffer
    rb = ReplayBuffer(max_items=getattr(args, "replay_max_items", 512))
    # Linear probe (ridge) for quick few-shot adaptation
    probe = RidgeProbe(d=args.d_model, max_items=getattr(args, "probe_max_items", 2048), lam=getattr(args, "probe_lambda", 1e-3), alpha=getattr(args, "probe_alpha", 0.7), device=device)
    # Eval history for on-demand sleep trigger
    eval_hist: list[float] = []
    prev_eval_avg: float | None = None
    # per-task dynamic weights
    lambda_bd = 1.0
    lambda_ood = 1.0
    # optional numeric-head EMA teacher
    num_ema = None
    try:
        import copy as _copy
        model.num_head_teacher = _copy.deepcopy(model.num_head).to(device)
        for p in model.num_head_teacher.parameters():
            p.requires_grad = False
        num_ema = {"decay": float(args.num_ema_decay)}
    except Exception:
        model.num_head_teacher = None
        num_ema = None

    # helpers for frozen mix
    def _digit_widths(nums: torch.Tensor) -> torch.Tensor:
        # base-10 digit count; treat 0 as width=1; allow negatives
        x = nums.abs().clamp_min(0)
        w = torch.ones_like(x)
        xnz = x.clone()
        xnz[xnz == 0] = 1
        w = torch.floor(torch.log10(xnz.float()) + 1).long()
        return w

    def _sample_by_width(n: int, widths: list[int], idx_range: tuple[int, int] | None):
        """Sample exactly n indices, roughly balanced by digit width.
        Falls back to uniform sampling within idx_range if a width has no candidates.
        """
        lo, hi = (0, gen.n_items - 1) if idx_range is None else (int(idx_range[0]), int(idx_range[1]))
        lo = max(0, lo); hi = min(gen.n_items - 1, hi)
        all_idx = torch.arange(lo, hi + 1, device=device)
        if all_idx.numel() <= 0:
            return torch.randint(0, max(1, gen.n_items), (n,), device=device)
        ws = _digit_widths(all_idx)
        # allocate counts evenly across given widths
        m = max(1, len(widths))
        counts = [n // m + (1 if i < (n % m) else 0) for i in range(m)]
        parts = []
        for w, c in zip(widths, counts):
            if c <= 0:
                continue
            cand = all_idx[ws == w]
            if cand.numel() == 0:
                continue
            sel = cand[torch.randint(0, cand.numel(), (c,), device=device)]
            parts.append(sel)
        if len(parts) == 0:
            # fallback: sample uniformly from the allowed range
            base = all_idx
            return base[torch.randint(0, base.numel(), (n,), device=device)]
        out = torch.cat(parts, dim=0)
        if out.numel() < n:
            # pad with uniform samples from range
            extra = all_idx[torch.randint(0, all_idx.numel(), (n - out.numel(),), device=device)]
            out = torch.cat([out, extra], dim=0)
        # shuffle and trim to n
        perm = torch.randperm(out.numel(), device=device)
        return out[perm][:n]

    def _build_compare_subset(n: int, kind: str, idx_range: tuple[int, int] | None):
        # kind: 'id' | 'bd' | 'ood' | 'cf'
        if n <= 0:
            return None
        # choose widths 2..7 evenly
        widths = [2, 3, 4, 5, 6, 7]
        if kind == 'ood' and args.ood_range:
            # sample entirely from OOD range if provided
            lo, hi = map(int, args.ood_range.split('-'))
            a = _sample_by_width(n, widths, (lo, hi))
            b = _sample_by_width(n, widths, (lo, hi))
        else:
            a = _sample_by_width(n, widths, idx_range)
            b = _sample_by_width(n, widths, idx_range)
        # boundary: force |a-b| in {0,1} and include a few special patterns
        if kind == 'bd':
            # Rebuild boundary set legitimately: avoid edges; balanced by construction; include counterfactual flips
            lo_safe = 1 if idx_range is None else max(1, int(idx_range[0]) + 1)
            hi_safe = (gen.n_items - 2) if idx_range is None else min(gen.n_items - 2, int(idx_range[1]) - 1)
            m = max(2, (n // 2) * 2)
            half = m // 2
            a0 = torch.randint(lo_safe, hi_safe + 1, (half,), device=device)
            # half positives, half negatives for '>'/'<' chosen later by op
            pos_mask = torch.zeros(half, dtype=torch.bool, device=device); pos_mask[: half // 2] = True
            pos_mask = pos_mask[torch.randperm(half, device=device)]
            # Randomly assign op per item
            op0 = torch.randint(0, 2, (half,), device=device)  # 0:'>', 1:'<'
            b0 = torch.empty_like(a0)
            yb0 = torch.empty_like(a0)
            for i in range(half):
                if op0[i] == 0:  # greater
                    if pos_mask[i]:
                        b0[i] = a0[i] - 1
                        yb0[i] = 1
                    else:
                        b0[i] = a0[i] if torch.rand(1, device=device).item() < 0.5 else a0[i] + 1
                        yb0[i] = 0
                else:  # smaller
                    if pos_mask[i]:
                        b0[i] = a0[i] + 1
                        yb0[i] = 1
                    else:
                        b0[i] = a0[i] if torch.rand(1, device=device).item() < 0.5 else a0[i] - 1
                        yb0[i] = 0
            # Counterfactual flips
            a1 = a0.clone(); op1 = op0.clone(); yb1 = 1 - yb0
            b1 = torch.empty_like(b0)
            for i in range(half):
                if op0[i] == 0:
                    b1[i] = a0[i] if yb0[i] == 1 else a0[i] - 1
                else:
                    b1[i] = a0[i] if yb0[i] == 1 else a0[i] + 1
            # combine and trim
            A = torch.cat([a0, a1], dim=0)
            B = torch.cat([b0, b1], dim=0)
            OO = torch.cat([op0, op1], dim=0)
            YY = torch.cat([yb0, yb1], dim=0)
            perm = torch.randperm(A.size(0), device=device)
            sel = perm[:n]
            a = A[sel]; b = B[sel]; op = OO[sel]; y_bin = YY[sel]
        # counterfactual flips: start from in-dist and tilt by ±1 to flip
        # For non-boundary kinds, (re)sample op and derive labels to match current lengths
        if kind != 'bd':
            n_eff = int(a.size(0))
            op = torch.randint(0, 2, (n_eff,), device=device)  # 0:'>', 1:'<'
            y_bin = torch.where(op == 0, (a > b).long(), (a < b).long())
        if kind == 'cf':
            delta = torch.where(y_bin > 0, -1, 1)
            b = (b + delta).clamp(0, gen.n_items - 1)
            y_bin = torch.where(op == 0, (a > b).long(), (a < b).long())
        # hard-negative mining for bd/ood: oversample pool and keep hardest n by small margin
        if kind in ('bd', 'ood') and n > 0 and args.hard_mine_pct > 0:
            pool = max(n, int(n / max(1e-6, min(0.99, args.hard_mine_pct))))
            pool = max(pool, n)
            # expand by sampling additional a,b
            aa = _sample_by_width(pool, widths, idx_range if kind != 'ood' or not args.ood_range else tuple(int(x) for x in args.ood_range.split('-')))
            bb = _sample_by_width(pool, widths, idx_range if kind != 'ood' or not args.ood_range else tuple(int(x) for x in args.ood_range.split('-')))
            if kind == 'bd':
                half = pool // 2
                bb[:half] = aa[:half]
                rem = pool - half
                if rem > 0:
                    delta = torch.randint(0, 2, (rem,), device=device)
                    delta = torch.where(delta == 0, torch.tensor(-1, device=device), torch.tensor(1, device=device))
                    bb[half:] = (aa[half:] + delta).clamp(0, gen.n_items - 1)
            oo = torch.randint(0, 2, (pool,), device=device)
            yb = torch.where(oo == 0, (aa > bb).long(), (aa < bb).long())
            ids_p, mask_p = _pack_symbolic_compare_text(aa, bb, oo, tok, max_len=seq_len)
            with torch.no_grad():
                _, logits_p, _, _, _, _ = model(ids_p, mask_p)
                from concept_learner.utils import select_pair_columns_safe
                lp = select_pair_columns_safe(logits_p, NO_IDX, YES_IDX, order="ab")
                margin = (lp[:, 1] - lp[:, 0]).abs()  # |logit_yes - logit_no|
                # smallest margins are hardest
                idx_sel = torch.topk(-margin, k=n).indices
            ids = ids_p[idx_sel]
            mask = mask_p[idx_sel]
            a = aa[idx_sel]
            b = bb[idx_sel]
            op = oo[idx_sel]
            y_bin = yb[idx_sel]
        # Pack texts if not mined above
        if 'ids' not in locals() or 'mask' not in locals():
            ids, mask = _pack_symbolic_compare_text(a, b, op, tok, max_len=seq_len)
        y = torch.where(
            y_bin > 0,
            torch.full((y_bin.size(0),), YES_IDX, dtype=torch.long, device=device),
            torch.full((y_bin.size(0),), NO_IDX, dtype=torch.long, device=device),
        )
        meta = {"a": a, "b": b, "op": op, "y_bin": y_bin}
        return ids, mask, y, meta
    # Initialize optional prototypical router
    proto = None
    from concept_learner.knn_memory import KNNSoftClassifier
    knn = None
    if int(getattr(args, "proto_enable", 0)) > 0:
        # feature dim is d_model regardless of space for (typed|pooled)
        feat_dim = int(args.d_model)
        proto = ProtoRouter(
            num_relations=9,
            feat_dim=feat_dim,
            beta=float(getattr(args, "proto_beta", 0.98)),
            weight=float(getattr(args, "proto_weight", 0.2)),
            weight_end=float(getattr(args, "proto_weight_end", 0.1)),
            decay_steps=int(getattr(args, "proto_decay_steps", 0)),
            space=str(getattr(args, "proto_space", "typed")),
            warmup=int(getattr(args, "proto_warmup", 50)),
            device=device,
        )
        # If resuming, try to load proto state as well
        if args.resume and os.path.isfile(args.resume):
            try:
                _ = load_checkpoint(args.resume, model, opt=None, map_location=device, proto=proto)
            except Exception:
                pass
    # ---- Composition scheduler state ----
    comp_enabled = int(getattr(args, "comp_enable", 1)) > 0
    comp_r = float(getattr(args, "comp_init", 0.15))
    comp_min = float(getattr(args, "comp_min", 0.10))
    comp_max = float(getattr(args, "comp_max", 0.50))
    comp_tick = int(getattr(args, "comp_tick", 500))
    comp_big = float(getattr(args, "comp_ramp_big", 0.10))
    comp_small = float(getattr(args, "comp_ramp_small", 0.05))
    comp_entry = float(getattr(args, "comp_entry", 0.30))
    comp_burst_every = int(getattr(args, "comp_burst_every", 400))
    comp_burst_groups = int(getattr(args, "comp_burst_groups", 2))
    comp_burst_k_min = int(getattr(args, "comp_burst_k_min", 12))
    comp_burst_k_max = int(getattr(args, "comp_burst_k_max", 16))
    comp_support_min = int(getattr(args, "comp_support_min", 8))
    comp_sleep_gate = int(getattr(args, "comp_sleep_gate", 0)) > 0
    comp_support: dict[int, int] = {}
    comp_stage = "A"  # 'A' basics, 'B' composition
    act_ent_hist: list[float] = []

    # Curriculum (phase-wise exclusive training)
    curr_enable = int(getattr(args, "curriculum_enable", 0)) > 0
    curr_groups = [s.strip() for s in str(getattr(args, "curr_groups", "BasicsA,BasicsB,Structure,Composition")).split(",") if s.strip()]
    if len(curr_groups) == 0:
        curr_groups = ["BasicsA", "BasicsB", "Structure", "Composition"]
    curr_idx = 0
    curr_phase = curr_groups[curr_idx]
    curr_phase_min = int(getattr(args, "curr_phase_min_steps", 800))
    thr_cnt_A = float(getattr(args, "curr_thr_cnt_A", 0.70))
    thr_in_A = float(getattr(args, "curr_thr_in_A", 0.60))
    thr_vq_A = float(getattr(args, "curr_thr_vq_A", 0.30))
    thr_stop_lo_A = float(getattr(args, "curr_thr_stop_lo_A", 0.18))
    thr_stop_hi_A = float(getattr(args, "curr_thr_stop_hi_A", 0.35))
    thr_bdry_B_lo = float(getattr(args, "curr_thr_bdry_B_lo", 0.45))
    thr_bdry_B_hi = float(getattr(args, "curr_thr_bdry_B_hi", 0.60))
    thr_tood_gap = float(getattr(args, "curr_thr_tood_gap", 0.05))

    for step in range(start_step, args.steps):
        # EMA decay schedule: increase after threshold
        if ema is not None:
            try:
                thr = int(getattr(args, "ema_decay_after", 10000))
                if step >= thr:
                    ema.decay = float(getattr(args, "ema_decay_final", ema.decay))
            except Exception:
                pass
        if args.sched == "cosine":
            set_cosine_lr(step)
        # Apply head LR multiplier during early steps
        def _apply_head_lr(step_idx: int):
            base_lr = opt.param_groups[0]["lr"] if len(opt.param_groups) > 0 else args.lr
            mult = 2.0 if step_idx < int(getattr(args, "head_boost_steps", 5000)) else 1.0
            for g in opt.param_groups:
                is_head = bool(g.get("is_head", False))
                g["lr"] = base_lr * (mult if is_head else 1.0)
        _apply_head_lr(step)
        rel_arg = getattr(args, "relations", None)
        rel_ids = _parse_relations_arg(rel_arg)
        # If no --relations provided, train on all pair relations (0..8).
        # If provided, use exactly the requested subset.
        include_pairs = True
        if rel_arg and rel_arg.strip().lower() != "all" and (rel_ids is None or len(rel_ids) == 0):
            include_pairs = False

        # Curriculum: compute per-phase task toggles
        if curr_enable:
            curr_phase = curr_groups[curr_idx]
            inc_pairs = (curr_phase in ("BasicsB", "Structure"))
            inc_count = (curr_phase == "BasicsA")
            inc_place = (curr_phase == "Structure")
            inc_parity = (curr_phase == "BasicsA")
            inc_offsets = (curr_phase == "BasicsA")
            inc_eqnum = (curr_phase == "BasicsA")
            inc_cmp = (curr_phase == "BasicsB")
            inc_edge = (curr_phase == "BasicsB")
            inc_add = (curr_phase == "BasicsB")
            # Only enable composition generation in Composition phase
            comp_enabled = comp_enabled and (curr_phase == "Composition")
            include_pairs = include_pairs and inc_pairs
        else:
            inc_pairs = inc_count = inc_place = inc_parity = inc_offsets = inc_eqnum = inc_cmp = inc_edge = inc_add = True

        if include_pairs and not args.frozen_mix:
            # optional restriction to training range for in-distribution sampling
            def parse_range(s):
                if not s:
                    return None
                try:
                    lo, hi = s.split("-")
                    return (int(lo), int(hi))
                except Exception:
                    return None
            train_idx_range = parse_range(getattr(args, "train_range", None))
            if args.overfit_batch is not None and step == start_step:
                cached = gen.sample_posneg_pairs(
                    batch=args.overfit_batch, allowed_relations=rel_ids, idx_range=train_idx_range
                )
            batch = (
                cached
                if args.overfit_batch is not None
                else gen.sample_posneg_pairs(
                    batch=args.batch_size, allowed_relations=rel_ids, idx_range=train_idx_range
                )
            )
        elif not include_pairs:
            # empty pairs batch
            B0 = 0
            device0 = device
            batch = {
                "a_idx": torch.empty(B0, dtype=torch.long, device=device0),
                "b_idx": torch.empty(B0, dtype=torch.long, device=device0),
                "rel": torch.empty(B0, dtype=torch.long, device=device0),
                "label": torch.empty(B0, dtype=torch.long, device=device0),
            }
        else:
            # Frozen mix batch for pair compares (robust exact partition with remainder)
            Bm = args.batch_size
            r_id = float(getattr(args, 'mix_ratio_id', 0.45))
            r_bd = float(getattr(args, 'mix_ratio_bd', 0.20))
            r_ood = float(getattr(args, 'mix_ratio_ood', 0.25))
            # If cf not provided (>0), derive it as the remainder; else use provided
            if float(getattr(args, 'mix_ratio_cf', 0.0)) > 0.0:
                r_cf = float(getattr(args, 'mix_ratio_cf', 0.0))
            else:
                r_cf = max(0.0, 1.0 - (r_id + r_bd + r_ood))
            # Normalize to sum to 1 (avoid drift)
            s = max(1e-8, r_id + r_bd + r_ood + r_cf)
            rr = [r_id / s, r_bd / s, r_ood / s, r_cf / s]
            # Floor counts then distribute remainder by largest fractional parts
            raw = [rr[0] * Bm, rr[1] * Bm, rr[2] * Bm, rr[3] * Bm]
            base = [int(x) for x in raw]
            rem = int(Bm - sum(base))
            if rem > 0:
                frac_idx = sorted(range(4), key=lambda i: (raw[i] - base[i]), reverse=True)
                for i in range(rem):
                    base[frac_idx[i % 4]] += 1
            # Assign
            n_id, n_bd, n_ood, n_cf = base
            # In-dist idx range
            def parse_range(s):
                if not s:
                    return None
                try:
                    lo, hi = s.split("-")
                    return (int(lo), int(hi))
                except Exception:
                    return None
            train_idx_range = parse_range(getattr(args, "train_range", None))
            comp_splits = {}
            comp_splits['id'] = _build_compare_subset(n_id, 'id', train_idx_range)
            comp_splits['bd'] = _build_compare_subset(n_bd, 'bd', train_idx_range)
            comp_splits['ood'] = _build_compare_subset(n_ood, 'ood', train_idx_range)
            comp_splits['cf'] = _build_compare_subset(n_cf, 'cf', train_idx_range)
            ids_cmp_list, mask_cmp_list, y_cmp_list = [], [], []
            for key in ['id', 'bd', 'ood', 'cf']:
                if comp_splits[key] is None:
                    continue
                ids_k, mask_k, y_k, _ = comp_splits[key]
                ids_cmp_list.append(ids_k)
                mask_cmp_list.append(mask_k)
                y_cmp_list.append(y_k)
            ids_pairs = torch.cat(ids_cmp_list, dim=0) if len(ids_cmp_list) > 0 else torch.empty(0, seq_len, dtype=torch.long, device=device)
            mask_pairs = torch.cat(mask_cmp_list, dim=0) if len(mask_cmp_list) > 0 else torch.empty(0, seq_len, dtype=torch.long, device=device)
            y_pairs = torch.cat(y_cmp_list, dim=0) if len(y_cmp_list) > 0 else torch.empty(0, dtype=torch.long, device=device)
        # natural-language pair questions
        if not args.frozen_mix:
            ids_pairs, mask_pairs, tpl_ids_pairs = _pack_pair_questions_text_with_tpl(
                batch, tok, max_len=seq_len, template_filter=tmpl_train
            )
            y_pairs_bin = batch["label"].to(device)
            y_pairs = torch.where(
                y_pairs_bin > 0,
                torch.full_like(y_pairs_bin, YES_IDX),
                torch.full_like(y_pairs_bin, NO_IDX),
            )
        # mix in equality statements (succ/pred == a±1)
        eq_bsz = max(1, args.batch_size // 4)
        a_eq = torch.randint(0, gen.n_items, (eq_bsz,), device=device)
        kind_eq = torch.randint(0, 2, (eq_bsz,), device=device)
        ids_eq, mask_eq, _y_eq_bin = _pack_equality_examples_text(
            a_eq, kind_eq, tok, max_len=seq_len
        )
        y_eq = torch.full((eq_bsz,), YES_IDX, dtype=torch.long, device=device)

        # counting batch (BasicsA)
        if (not curr_enable) or inc_count:
            cnt_bsz = args.batch_size
            data_cnt = gen.sample_counting(batch=cnt_bsz, idx_range=train_idx_range)
            ids_cnt, mask_cnt = _pack_count_examples_text(
                data_cnt["kind"], data_cnt["a"], data_cnt["c"], tok, max_len=seq_len
            )
            y_cnt = data_cnt["target"].to(device)  # 0..N-1
        else:
            ids_cnt = torch.empty(0, seq_len, dtype=torch.long, device=device)
            mask_cnt = torch.empty(0, seq_len, dtype=torch.long, device=device)
            y_cnt = torch.empty(0, dtype=torch.long, device=device)

        # place-value / face-value inclusion controlled by --relations
        include_place, include_face = True, True
        rel_arg = getattr(args, "relations", None)
        if rel_arg and rel_arg.strip().lower() != "all":
            # determine inclusion from names
            rel_arg_l = [s.strip().lower() for s in rel_arg.split(",") if s.strip()]
            include_place = any(s in ("place_value",) for s in rel_arg_l)
            include_face = any(s in ("face_value",) for s in rel_arg_l)
        allowed_kinds = []
        if include_face:
            allowed_kinds += [0, 1, 3]
        # Guard: place value (kind=2) produces targets up to 90; require N>=100
        if include_place and ecfg.max_number >= 100:
            allowed_kinds += [2]
        use_pv = len(allowed_kinds) > 0
        if use_pv:
            pv_bsz = args.batch_size
            data_pv = gen.sample_place_value(batch=pv_bsz, allowed_kinds=allowed_kinds, idx_range=train_idx_range)
            ids_pv, mask_pv = _pack_place_value_text(
                data_pv["kind"], data_pv["a"], tok, max_len=seq_len, face_place=data_pv.get("face_place")
            )
            y_pv = data_pv["target"].to(device)
            ids = torch.cat([ids_pairs, ids_eq, ids_cnt, ids_pv], dim=0)
            mask = torch.cat([mask_pairs, mask_eq, mask_cnt, mask_pv], dim=0)
            y = torch.cat([y_pairs, y_eq, y_cnt, y_pv], dim=0)
        else:
            ids = torch.cat([ids_pairs, ids_eq, ids_cnt], dim=0)
            mask = torch.cat([mask_pairs, mask_eq, mask_cnt], dim=0)
            y = torch.cat([y_pairs, y_eq, y_cnt], dim=0)

        # numeric equality batch (true/false on a == b)
        eqnum_bsz = args.batch_size
        data_eqn = gen.sample_number_equality(batch=eqnum_bsz, idx_range=train_idx_range)
        ids_eqn, mask_eqn = _pack_num_equality_text(
            data_eqn["a"], data_eqn["b"], tok, max_len=seq_len
        )
        y_eqn = torch.where(
            data_eqn["label"].to(device) > 0,
            torch.full((eqnum_bsz,), YES_IDX, dtype=torch.long, device=device),
            torch.full((eqnum_bsz,), NO_IDX, dtype=torch.long, device=device),
        )

        ids = torch.cat([ids, ids_eqn], dim=0)
        mask = torch.cat([mask, mask_eqn], dim=0)
        y = torch.cat([y, y_eqn], dim=0)

        # symbolic comparison batch (BasicsB)
        if (not curr_enable) or inc_cmp:
            cmp_bsz = args.batch_size
            data_cmp = gen.sample_symbolic_compare(batch=cmp_bsz, idx_range=train_idx_range)
            ids_cmp, mask_cmp = _pack_symbolic_compare_text(
                data_cmp["a"], data_cmp["b"], data_cmp["op"], tok, max_len=seq_len
            )
            y_cmp = torch.where(
                data_cmp["label"].to(device) > 0,
                torch.full((cmp_bsz,), YES_IDX, dtype=torch.long, device=device),
                torch.full((cmp_bsz,), NO_IDX, dtype=torch.long, device=device),
            )
            ids = torch.cat([ids, ids_cmp], dim=0)
            mask = torch.cat([mask, mask_cmp], dim=0)
            y = torch.cat([y, y_cmp], dim=0)

        # Edge-case oversampling for comparisons (BasicsB)
        try:
            if (not curr_enable) or inc_edge:
                B_edge = max(1, int(0.15 * cmp_bsz))
                a_edge = torch.randint(0, gen.n_items, (B_edge,), device=device)
                # choose pattern among equal, +1, -1
                mode = torch.randint(0, 3, (B_edge,), device=device)
            b_edge = a_edge.clone()
            b_edge = torch.where(mode == 0, a_edge, b_edge)
            b_edge = torch.where(mode == 1, (a_edge + 1).clamp(max=gen.n_items - 1), b_edge)
            b_edge = torch.where(mode == 2, (a_edge - 1).clamp(min=0), b_edge)
            # build a>b? and a<b? variants evenly
            op_edge = torch.randint(0, 2, (B_edge,), device=device)
            # labels
            y_bin_edge = torch.where(op_edge == 0, (a_edge > b_edge).long(), (a_edge < b_edge).long())
            ids_cmp2, mask_cmp2 = _pack_symbolic_compare_text(a_edge, b_edge, op_edge, tok, max_len=seq_len)
            y_cmp2 = torch.where(
                y_bin_edge > 0,
                torch.full((B_edge,), YES_IDX, dtype=torch.long, device=device),
                torch.full((B_edge,), NO_IDX, dtype=torch.long, device=device),
            )
            if (not curr_enable) or inc_edge:
                ids = torch.cat([ids, ids_cmp2], dim=0)
                mask = torch.cat([mask, mask_cmp2], dim=0)
                y = torch.cat([y, y_cmp2], dim=0)
        except Exception:
            pass

        # addition batch (BasicsB)
        if (not curr_enable) or inc_add:
            add_bsz = args.batch_size
            data_add = gen.sample_addition(batch=add_bsz, idx_range=train_idx_range)
            ids_add, mask_add = _pack_addition_text(
                data_add["a"], data_add["b"], tok, max_len=seq_len
            )
            y_add = data_add["target"].to(device)
            ids = torch.cat([ids, ids_add], dim=0)
            mask = torch.cat([mask, mask_add], dim=0)
            y = torch.cat([y, y_add], dim=0)

        # Oversample carry cases for addition (…9 + 1)
        try:
            if (not curr_enable) or inc_add:
                Bc = max(1, args.batch_size // 8)
                a_carry = torch.randint(0, gen.n_items, (Bc,), device=device)
                a_carry = (a_carry // 10) * 10 + 9
                a_carry = a_carry.clamp(max=gen.n_items - 1)
                b_carry = torch.ones_like(a_carry)
                y_carry = (a_carry + b_carry).clamp(max=gen.n_items - 1)
                ids_add2, mask_add2 = _pack_addition_text(a_carry, b_carry, tok, max_len=seq_len)
            if (not curr_enable) or inc_add:
                ids = torch.cat([ids, ids_add2], dim=0)
                mask = torch.cat([mask, mask_add2], dim=0)
                y = torch.cat([y, y_carry], dim=0)
        except Exception:
            pass

        # parity batch (BasicsA)
        if (not curr_enable) or inc_parity:
            par_bsz = args.batch_size
            data_par = gen.sample_parity(batch=par_bsz, idx_range=train_idx_range)
            ids_par, mask_par = _pack_parity_text(data_par["a"], tok, max_len=seq_len)
            y_par = torch.where(
                (data_par["target"].to(device) % 2) > 0,
                torch.full((par_bsz,), ODD_IDX, dtype=torch.long, device=device),
                torch.full((par_bsz,), EVEN_IDX, dtype=torch.long, device=device),
            )
            ids = torch.cat([ids, ids_par], dim=0)
            mask = torch.cat([mask, mask_par], dim=0)
            y = torch.cat([y, y_par], dim=0)

        # simple +/- k tasks with k in {1,2,3} (BasicsA)
        if (not curr_enable) or inc_offsets:
            off_bsz = args.batch_size
            data_off = gen.sample_offset(batch=off_bsz, idx_range=train_idx_range)
            ids_off, mask_off = _pack_offset_text(
                data_off["a"], data_off["offset"], tok, max_len=seq_len
            )
            y_off = data_off["target"].to(device)
            ids = torch.cat([ids, ids_off], dim=0)
            mask = torch.cat([mask, mask_off], dim=0)
            y = torch.cat([y, y_off], dim=0)

        # Dynamic composition sizing based on target share r
        if comp_enabled:
            try:
                T_before = int(ids.size(0))
                Bsp = max(1, args.batch_size // 8)
                future_extras = (2 * Bsp) + 9  # succ/pred oversample + near-zero set
                target_comp = int(max(0, round((comp_r / max(1e-6, (1.0 - comp_r))) * (T_before + future_extras))))
                comp_num_bsz = max(0, target_comp // 2)
                comp_yn_bsz = max(0, target_comp - comp_num_bsz)
                if comp_num_bsz > 0:
                    comp_texts_num, comp_y_num, kinds_num = make_numeric_compositions(
                        n_items=ecfg.max_number, batch=comp_num_bsz, device=device
                    )
                    ids_cnum, mask_cnum = _pack_text_batch(comp_texts_num, tok, seq_len, device)
                    y = torch.cat([y, comp_y_num.to(device)], dim=0)
                    ids = torch.cat([ids, ids_cnum], dim=0)
                    mask = torch.cat([mask, mask_cnum], dim=0)
                    for k in kinds_num:
                        comp_support[int(k)] = comp_support.get(int(k), 0) + 1
                if comp_yn_bsz > 0:
                    comp_texts_yn, comp_y_yn_bin, kinds_yn = make_equality_compositions(
                        n_items=ecfg.max_number, batch=comp_yn_bsz, device=device
                    )
                    ids_cyn, mask_cyn = _pack_text_batch(comp_texts_yn, tok, seq_len, device)
                    y_cyn = torch.where(
                        comp_y_yn_bin.to(device) > 0,
                        torch.full((comp_yn_bsz,), YES_IDX, dtype=torch.long, device=device),
                        torch.full((comp_yn_bsz,), NO_IDX, dtype=torch.long, device=device),
                    )
                    ids = torch.cat([ids, ids_cyn], dim=0)
                    mask = torch.cat([mask, mask_cyn], dim=0)
                    y = torch.cat([y, y_cyn], dim=0)
                    for k in kinds_yn:
                        comp_support[int(k)] = comp_support.get(int(k), 0) + 1
            except Exception:
                pass

        # Oversample successor/pred around carries/borrows (BasicsA)
        try:
            if curr_enable and not inc_count:
                raise Exception("skip carry oversample outside BasicsA")
            Bsp = max(1, args.batch_size // 8)
            # successor with ones=9
            a_succ = torch.randint(0, gen.n_items, (Bsp,), device=device)
            a_succ = (a_succ // 10) * 10 + 9
            kind_succ = torch.zeros(Bsp, dtype=torch.long, device=device)  # successor
            c_dummy = torch.full_like(a_succ, -1)
            y_succ = (a_succ + 1).clamp(max=gen.n_items - 1)
            ids_succ, mask_succ = _pack_count_examples_text(kind_succ, a_succ, c_dummy, tok, max_len=seq_len)
            # predecessor with ones=0
            a_pred = torch.randint(0, gen.n_items, (Bsp,), device=device)
            a_pred = (a_pred // 10) * 10 + 0
            kind_pred = torch.ones(Bsp, dtype=torch.long, device=device)  # predecessor
            y_pred = (a_pred - 1).clamp(min=0)
            ids_pred, mask_pred = _pack_count_examples_text(kind_pred, a_pred, c_dummy, tok, max_len=seq_len)
            ids = torch.cat([ids, ids_succ, ids_pred], dim=0)
            mask = torch.cat([mask, mask_succ, mask_pred], dim=0)
            y = torch.cat([y, y_succ, y_pred], dim=0)
        except Exception:
            pass

        # Negatives around zero for comparisons (BasicsB)
        try:
            if curr_enable and not inc_cmp:
                raise Exception("skip near-zero compare outside BasicsB")
            vals = torch.tensor([-1, 0, 1], device=device)
            combos = []
            for ai in vals:
                for bi in vals:
                    combos.append((ai.item(), bi.item()))
            a_n = torch.tensor([c[0] for c in combos], device=device)
            b_n = torch.tensor([c[1] for c in combos], device=device)
            # alternate operators > and <
            op_n = torch.tensor([i % 2 for i in range(len(combos))], device=device)
            y_bin_n = torch.where(op_n == 0, (a_n > b_n).long(), (a_n < b_n).long())
            ids_ncmp, mask_ncmp = _pack_symbolic_compare_text(a_n, b_n, op_n, tok, max_len=seq_len)
            y_ncmp = torch.where(
                y_bin_n > 0,
                torch.full((len(combos),), YES_IDX, dtype=torch.long, device=device),
                torch.full((len(combos),), NO_IDX, dtype=torch.long, device=device),
            )
            ids = torch.cat([ids, ids_ncmp], dim=0)
            mask = torch.cat([mask, mask_ncmp], dim=0)
            y = torch.cat([y, y_ncmp], dim=0)
        except Exception:
            pass
        # Optional compositional template-block bursts for support
        if comp_enabled and (not curr_enable or curr_phase == "Composition") and comp_burst_every > 0:
            try:
                if (step + 1) % comp_burst_every == 0:
                    import random as _r
                    groups = max(1, comp_burst_groups)
                    for _ in range(groups):
                        K = _r.randint(max(1, comp_burst_k_min), max(comp_burst_k_min, comp_burst_k_max))
                        kind = _r.randrange(0, 8)
                        texts_b, y_b, kinds_b = make_numeric_compositions(
                            n_items=ecfg.max_number, batch=K, device=device, force_kind=kind
                        )
                        ids_b, mask_b = _pack_text_batch(texts_b, tok, seq_len, device)
                        ids = torch.cat([ids, ids_b], dim=0)
                        mask = torch.cat([mask, mask_b], dim=0)
                        y = torch.cat([y, y_b.to(device)], dim=0)
                        for k in kinds_b:
                            comp_support[int(k)] = comp_support.get(int(k), 0) + 1
            except Exception:
                pass
        # Reset fast-weights per episode (batch) before forward
        try:
            if int(getattr(args, "fast_enable", 1)) > 0:
                model.decoder.reset_fast()
        except Exception:
            pass
        with amp_ctx():
            logits_tok, logits_seq, vq_loss, indices, stop_logits, action_logits = model(ids, mask)
            # One-shot Hebbian update and recompute logits (before applying proto bias)
            try:
                if int(getattr(args, "fast_enable", 1)) > 0:
                    model.decoder.hebbian_update(y)
                    logits_seq2 = model.decoder.recompute_seq_logits()
                    if logits_seq2 is not None:
                        logits_seq = logits_seq2
            except Exception:
                pass
        # Sanitize logits_seq after any fast-weight recompute, before optional biases/mixing
        logits_seq = torch.nan_to_num(logits_seq, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        # kNN posterior mixing (train-time optional) and memory update
        try:
            if knn is None and int(getattr(args, "knn_enable", 1)) > 0:
                # determine feature dim from space
                d = args.d_model
                feat_dim = d if getattr(args, 'knn_space', 'typed') != 'concat' else (2 * d)
                knn = KNNSoftClassifier(
                    num_classes=num_numbers + 4,
                    feat_dim=feat_dim,
                    capacity=int(getattr(args, 'knn_capacity', 10000)),
                    k=int(getattr(args, 'knn_k', 32)),
                    tau=float(getattr(args, 'knn_tau', 0.2)),
                    device=device,
                ).to(device)
            if knn is not None:
                # build features
                with torch.no_grad():
                    h_pool, _H = model.enc(ids, mask)
                    s_typed = getattr(model.reasoner, "_last_s", None)
                    if s_typed is None:
                        s_typed = h_pool
                    from concept_learner.proto_router import ProtoRouter as _PR
                    feat = _PR.build_feature(h_pool, s_typed, getattr(args, 'knn_space', 'typed'))
                # update memory with current batch
                knn.add(feat, y)
                # apply mixing if not eval-only
                if int(getattr(args, 'knn_eval_only', 1)) == 0:
                    alpha = float(getattr(args, 'knn_alpha', 0.2))
                    dec = int(getattr(args, 'knn_alpha_decay_steps', 0))
                    a_end = float(getattr(args, 'knn_alpha_end', 0.1))
                    if dec > 0:
                        t = min(1.0, max(0.0, (step - start_step) / float(max(1, dec))))
                        alpha = (1.0 - t) * alpha + t * a_end
                    with torch.no_grad():
                        p_model = torch.softmax(logits_seq, dim=-1)
                        p_knn = knn.posterior(feat)
                        p_mix = (alpha * p_model + (1.0 - alpha) * p_knn).clamp_min(1e-8)
                        # replace logits_seq by log of mixed prob for CE
                        logits_seq = torch.log(p_mix)
        except Exception:
            pass
        # Apply prototypical routing on the pair slice at the front (if enabled)
        if proto is not None and ids_pairs.size(0) > 0:
            with torch.no_grad():
                h_all, _ = model.enc(ids, mask)
                s_typed = getattr(model.reasoner, "_last_s", None)
                if s_typed is None:
                    s_typed = h_all
                feat_all = ProtoRouter.build_feature(h_all, s_typed, getattr(args, "proto_space", "typed"))
            YES_IDX = num_numbers
            NO_IDX = num_numbers + 1
            # update + bias only for the pair segment at the front
            if not args.frozen_mix:
                rel_ids_pairs = batch["rel"]
            else:
                rel_list = []
                for k in ["id", "bd", "ood", "cf"]:
                    if comp_splits.get(k) is None:
                        continue
                    meta_k = comp_splits[k][3]
                    # op: 0-> '>' (greater:7), 1-> '<' (smaller:8)
                    op_k = meta_k.get("op").to(device)
                    rel_k = torch.where(op_k == 0, torch.full_like(op_k, 7), torch.full_like(op_k, 8))
                    rel_list.append(rel_k)
                rel_ids_pairs = torch.cat(rel_list, dim=0) if len(rel_list) > 0 else torch.full((ids_pairs.size(0),), -1, device=device, dtype=torch.long)
            # guard shape
            rel_ids_pairs = rel_ids_pairs.to(device) if rel_ids_pairs is not None else torch.full((ids_pairs.size(0),), -1, device=device, dtype=torch.long)
            logits_seq[: ids_pairs.size(0)] = proto.apply_bias(
                logits_seq[: ids_pairs.size(0)],
                feat_all[: ids_pairs.size(0)],
                rel_ids_pairs[: ids_pairs.size(0)],
                YES_IDX,
                NO_IDX,
                train_mode=True,
            )
            # sanitize after bias
            logits_seq = torch.nan_to_num(logits_seq, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        # If frozen mix is enabled, compute separate CE losses for each compare split and reweight
        # Label smoothing schedule: start -> final after step threshold
        ls_sched_start = float(getattr(args, "label_smoothing_start", 0.05))
        ls_sched_final = float(getattr(args, "label_smoothing_final", 0.02))
        ls_sched_after = int(getattr(args, "label_smoothing_after", 8000))
        cur_ls = (
            ls_sched_start if (getattr(args, "label_smoothing", 0.0) <= 0.0) else float(getattr(args, "label_smoothing"))
        )
        if getattr(args, "label_smoothing", 0.0) <= 0.0:
            cur_ls = ls_sched_start if step < ls_sched_after else ls_sched_final
        # Final safety: sanitize logits before CE and build a safe mask for valid labels
        logits_seq = torch.nan_to_num(logits_seq, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        C = logits_seq.size(-1)
        ignore_index = -100
        valid_y = (y >= 0) & (y < C)
        y_safe = y.clone()
        y_safe = torch.where(valid_y, y_safe, torch.full_like(y_safe, ignore_index))
        seq_mask = valid_y
        # Diagnostics: track label coverage and range
        try:
            n_seq_valid = int(seq_mask.sum().item())
            y_min = int(y.min().item()) if y.numel() > 0 else 0
            y_max = int(y.max().item()) if y.numel() > 0 else -1
        except Exception:
            n_seq_valid, y_min, y_max = 0, 0, -1
        if seq_mask.any():
            # CE on masked rows
            logits_m = logits_seq[seq_mask]
            y_m = y_safe[seq_mask]
            # quick diagnostic: check finiteness pre-CE
            try:
                nf_all = int((~torch.isfinite(logits_m)).sum().item())
                if nf_all > 0:
                    print(f"[warn] logits contained {nf_all} non-finite values before CE (masked subset)")
            except Exception:
                pass
            # Compute CE in FP32 even under AMP to avoid half-precision instability
            loss_seq = torch.nn.functional.cross_entropy(
                logits_m.float(), y_m, label_smoothing=cur_ls
            )
            # If CE still returns non-finite, attempt manual smoothed CE
            if not torch.isfinite(loss_seq):
                try:
                    eps = float(cur_ls)
                    lsm = torch.log_softmax(logits_m.float(), dim=-1)
                    nll = -lsm.gather(1, y_m.view(-1, 1)).squeeze(1)
                    if eps > 0.0:
                        smooth = -lsm.mean(dim=-1)
                        loss_seq = ((1.0 - eps) * nll + eps * smooth).mean()
                    else:
                        loss_seq = nll.mean()
                    if not torch.isfinite(loss_seq):
                        print("[warn] non-finite seq CE even after manual; skipping seq loss this step")
                        loss_seq = logits_seq.new_zeros(())
                    else:
                        print("[info] used manual smoothed CE due to non-finite from F.cross_entropy")
                except Exception:
                    print("[warn] CE fallback failed; skipping seq loss this step")
                    loss_seq = logits_seq.new_zeros(())
        else:
            fb = str(getattr(args, "seq_fallback", "skip")).lower()
            w_fb = float(getattr(args, "seq_fallback_weight", 0.1))
            if fb == "uniform":
                # Cross-entropy to uniform target ≈ mean(-log_softmax)
                loss_u = (-torch.log_softmax(logits_seq, dim=-1)).mean()
                loss_seq = w_fb * torch.nan_to_num(loss_u, nan=0.0, posinf=0.0, neginf=0.0)
                try:
                    print("[info] zero seq labels; using uniform fallback")
                except Exception:
                    pass
            elif fb == "self":
                with torch.no_grad():
                    y_pseudo = logits_seq.argmax(dim=-1).clamp(0, C - 1)
                loss_p = torch.nn.functional.cross_entropy(logits_seq, y_pseudo)
                loss_seq = w_fb * torch.nan_to_num(loss_p, nan=0.0, posinf=0.0, neginf=0.0)
                try:
                    print("[info] zero seq labels; using self-training fallback")
                except Exception:
                    pass
            else:
                loss_seq = logits_seq.new_zeros(())
                try:
                    print("[warn] zero seq labels in batch; skipping seq CE")
                except Exception:
                    pass
        if args.frozen_mix:
            # recompute logits for compare-only batch and split by sizes
            offs = []
            for key in ['id', 'bd', 'ood', 'cf']:
                if key in locals() and comp_splits.get(key) is not None and comp_splits[key][0].size(0) > 0:
                    offs.append((key, comp_splits[key][0].size(0)))
            start = 0
            loss_cmp_total = torch.tensor(0.0, device=device)
            for key, sz in offs:
                ids_k, mask_k, y_k, _m = comp_splits[key]
                _, logits_k, _, _, _, _ = model(ids_k, mask_k)
                logits_k = torch.nan_to_num(logits_k, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                Ck = logits_k.size(-1)
                yk_valid = (y_k >= 0) & (y_k < Ck)
                if yk_valid.any():
                    ce_k = torch.nn.functional.cross_entropy(logits_k[yk_valid].float(), y_k[yk_valid], label_smoothing=cur_ls)
                else:
                    ce_k = logits_k.new_zeros(())
                w = 1.0
                if key == 'bd':
                    w = lambda_bd
                elif key == 'ood':
                    w = lambda_ood
                loss_cmp_total = loss_cmp_total + w * ce_k
            if len(offs) > 0:
                # replace default CE contribution with the reweighted compare CE averaged with other tasks
                loss_seq = loss_seq * 0.5 + loss_cmp_total * 0.5

        # Collect binary compare examples into probe buffer
        try:
            YES_IDX = num_numbers
            NO_IDX = num_numbers + 1
            with torch.no_grad():
                h_all, _H = model.enc(ids, mask)
            sel = (y == YES_IDX) | (y == NO_IDX)
            if sel.any() and int(getattr(args, "probe_enable", 1)) > 0:
                x = h_all[sel]
                y_bin = (y[sel] == YES_IDX).float()
                probe.add(x, y_bin)
        except Exception:
            pass

        # ---- Auxiliary: margin-aware BCE for YES/NO near boundary ----
        # apply only on symbolic compares where |a-b|<=1 or a==b
        try:
            a_cmp = data_cmp["a"].to(device)
            b_cmp = data_cmp["b"].to(device)
            y_cmp_bin = data_cmp["label"].float().to(device)
            with torch.no_grad():
                ids_c, mask_c = ids_cmp, mask_cmp
            logits_cmp = model(ids_c, mask_c)[1]
            logits_cmp = torch.nan_to_num(logits_cmp, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            from concept_learner.utils import select_pair_columns_safe
            yes_no = select_pair_columns_safe(logits_cmp, YES_IDX, NO_IDX, order="ab")
            logit_yes = yes_no[:, 0] - yes_no[:, 1]
            w = torch.where((a_cmp == b_cmp) | ((a_cmp - b_cmp).abs() <= 1), 1.3, 1.0)
            # focal-like weighting if skewed
            p = torch.sigmoid(logit_yes).clamp(1e-4, 1 - 1e-4)
            y = y_cmp_bin
            p_t = torch.where(y > 0, p, 1 - p)
            pos_ratio = y.mean().item()
            gamma = 0.5 if (pos_ratio < 0.4 or pos_ratio > 0.6) else 0.0
            focal = (1 - p_t) ** gamma if gamma > 0 else 1.0
            bce_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                logit_yes, y, reduction="none"
            )
            bce_margin = ((bce_raw * w * focal).mean())
        except Exception:
            bce_margin = torch.tensor(0.0, device=device)

        # ---- Auxiliary: ranking loss to encourage monotone number line ----
        # Use symbolic compare pairs for supervision
        try:
            # pack numbers alone so the numeric head reads only digits
            def _pack_numbers(nums: torch.Tensor):
                texts = [str(int(v.item())) for v in nums]
                return _pack_text_batch(texts, tok, max_len=seq_len, device=device)

            a_rank = data_cmp["a"]
            b_rank = data_cmp["b"]
            # Counterfactual pairing near boundary: flip by +/-1 when possible
            try:
                a_cf = a_rank.clone()
                b_cf = b_rank.clone()
                # move the right-hand side by +1 or -1 to cross boundary occasionally
                delta = torch.randint_like(b_cf, low=0, high=2)
                delta = torch.where(delta == 0, torch.tensor(-1, device=device), torch.tensor(1, device=device))
                b_cf = (b_cf + delta).clamp_min(0)
                a_rank = torch.cat([a_rank, a_cf], dim=0)
                b_rank = torch.cat([b_rank, b_cf], dim=0)
            except Exception:
                pass
            ids_a, mask_a = _pack_numbers(a_rank)
            ids_b, mask_b = _pack_numbers(b_rank)
            v_a = model.predict_scalar(ids_a, mask_a).squeeze(-1)
            v_b = model.predict_scalar(ids_b, mask_b).squeeze(-1)
            # sign = +1 when a<b, -1 when a>b
            sign = torch.where(a_rank < b_rank, 1.0, -1.0).to(device)
            margin = 1.0
            rank_loss = torch.nn.functional.relu(margin - sign * (v_b - v_a)).mean()
        except Exception:
            rank_loss = torch.tensor(0.0, device=device)

        # ---- Auxiliary: numeric regression on digits (20% of steps) ----
        try:
            if step % 5 == 0:
                nums = []
                for t in [
                    data_add.get("a"), data_add.get("b"), data_add.get("target"),
                    data_off.get("a"), data_off.get("target"),
                    data_cnt.get("a"), data_cnt.get("target"),
                ]:
                    if t is not None:
                        nums.append(t.to(device))
                if len(nums) > 0:
                    nums_cat = torch.unique(torch.cat(nums))
                    ids_n, mask_n = _pack_numbers(nums_cat)
                    v_pred = model.predict_scalar(ids_n, mask_n).squeeze(-1)
                    v_tgt = torch.log1p(nums_cat.float())
                    num_loss = torch.nn.functional.mse_loss(v_pred, v_tgt)
                else:
                    num_loss = torch.tensor(0.0, device=device)
            else:
                num_loss = torch.tensor(0.0, device=device)
        except Exception:
            num_loss = torch.tensor(0.0, device=device)

        # Build stop targets per-example using simple heuristics from episode metadata
        max_steps = getattr(model.reasoner, "max_steps", 1)
        Btot = ids.size(0)
        stop_targets = torch.zeros(Btot, max_steps, device=device)
        stop_mask = torch.zeros(Btot, max_steps, device=device)

        # mapping for relations (pairs)
        rel_steps_map = {
            0: 2,  # same_parity -> compare parity of two numbers
            1: 1,  # successor
            2: 1,  # predecessor
            3: 1,  # add_2
            4: 2,  # same_tens
            5: 2,  # same_ones
            6: 2,  # makes_ten
            7: 2,  # greater
            8: 2,  # smaller
        }
        # Determine steps for compare/pair items.
        # When using natural-language pair batches (no frozen_mix), we have `batch['rel']`.
        # Under frozen_mix, we don't have `batch`; treat all compare prompts as 2 steps (gt/lt).
        if 'batch' in locals() and isinstance(batch, dict):
            rel = batch.get("rel")
        else:
            rel = None
        if rel is not None and rel.numel() > 0:
            steps_pairs = torch.tensor([rel_steps_map[int(r.item())] for r in rel], device=device)
        else:
            steps_pairs = torch.full((ids_pairs.size(0),), 2, dtype=torch.long, device=device) if ids_pairs is not None else torch.zeros(0, dtype=torch.long, device=device)
        # equality statements -> 1 step
        steps_eq = torch.ones(ids_eq.size(0), dtype=torch.long, device=device)
        # counting mapping: successor/pred=1, between=2
        steps_cnt = torch.ones(ids_cnt.size(0), dtype=torch.long, device=device)
        if "kind" in data_cnt:
            steps_cnt = torch.where(
                data_cnt["kind"].to(device) == 2,
                torch.tensor(2, device=device),
                torch.tensor(1, device=device),
            )

        # place-value tasks -> assume 1 step
        steps_pv = torch.ones(0, dtype=torch.long, device=device)

        # assign into targets/mask for concatenated batch
        off_pairs = 0
        off_eq = ids_pairs.size(0)
        off_cnt = ids_pairs.size(0) + ids_eq.size(0)
        off_pv = ids_pairs.size(0) + ids_eq.size(0) + ids_cnt.size(0)
        off_eqn = off_pv + (ids_pv.size(0) if 'ids_pv' in locals() and use_pv else 0)
        off_cmp = off_eqn + ids_eqn.size(0)
        off_add = off_cmp + ids_cmp.size(0)
        off_par = off_add + ids_add.size(0)
        off_off = off_par + ids_par.size(0)
        for i, s in enumerate(steps_pairs.tolist()):
            s = int(max(1, min(s, max_steps)))
            stop_targets[off_pairs + i, s - 1] = 1.0
            stop_mask[off_pairs + i, :s] = 1.0
        for i, s in enumerate(steps_eq.tolist()):
            s = int(max(1, min(s, max_steps)))
            stop_targets[off_eq + i, s - 1] = 1.0
            stop_mask[off_eq + i, :s] = 1.0
        for i, s in enumerate(steps_cnt.tolist()):
            s = int(max(1, min(s, max_steps)))
            stop_targets[off_cnt + i, s - 1] = 1.0
            stop_mask[off_cnt + i, :s] = 1.0
        if use_pv:
            steps_pv = torch.ones(ids_pv.size(0), dtype=torch.long, device=device)
            for i, s in enumerate(steps_pv.tolist()):
                s = int(max(1, min(s, max_steps)))
                stop_targets[off_pv + i, s - 1] = 1.0
                stop_mask[off_pv + i, :s] = 1.0
        # equality of numbers: treat as compare -> 2 steps
        steps_eqn = torch.full((ids_eqn.size(0),), 2, dtype=torch.long, device=device)
        for i, s in enumerate(steps_eqn.tolist()):
            s = int(max(1, min(s, max_steps)))
            stop_targets[off_eqn + i, s - 1] = 1.0
            stop_mask[off_eqn + i, :s] = 1.0
        # symbolic compare: also 2 steps
        steps_cmp = torch.full((ids_cmp.size(0),), 2, dtype=torch.long, device=device)
        for i, s in enumerate(steps_cmp.tolist()):
            s = int(max(1, min(s, max_steps)))
            stop_targets[off_cmp + i, s - 1] = 1.0
            stop_mask[off_cmp + i, :s] = 1.0
        # addition: 2 steps (parse + add)
        steps_add = torch.full((ids_add.size(0),), 2, dtype=torch.long, device=device)
        for i, s in enumerate(steps_add.tolist()):
            s = int(max(1, min(s, max_steps)))
            stop_targets[off_add + i, s - 1] = 1.0
            stop_mask[off_add + i, :s] = 1.0
        # parity and +/-k are 1-step
        for i in range(ids_par.size(0)):
            stop_targets[off_par + i, 0] = 1.0
            stop_mask[off_par + i, 0] = 1.0
        for i in range(ids_off.size(0)):
            stop_targets[off_off + i, 0] = 1.0
            stop_mask[off_off + i, 0] = 1.0

        # masked BCE across valid steps only
        stop_per = torch.nn.functional.binary_cross_entropy_with_logits(
            stop_logits, stop_targets, reduction="none"
        )
        stop_loss = (stop_per * stop_mask).sum() / stop_mask.sum().clamp_min(1.0)

        # tiny OpAdd stabilization regularizer (encourage integer-like k)
        reg_k = 0.0
        try:
            for op in getattr(model.reasoner, "ops", []):
                if getattr(op, "name", "") == "add":
                    reg_k = reg_k + op.regularization()
            reg_k = float(reg_k) if isinstance(reg_k, (int, float)) else reg_k
        except Exception:
            reg_k = torch.tensor(0.0, device=device)

        lambda_rank = 0.1
        lambda_num = 0.05
        lambda_margin = 1.0
        lambda_k = 1e-3

        # include auxiliary sparsity and halting penalties from reasoner if present
        aux = getattr(model.reasoner, "_aux_loss", {}) or {}
        telem = getattr(model.reasoner, "_telemetry", {}) or {}
        # aux reward for using functions (percent of episodes using >=1 fn)
        fn_use = float(telem.get("pct_used_fn", 0.0))
        fn_use_term = torch.tensor(fn_use, device=device)
        loss = (
            loss_seq
            + args.lambda_vq * vq_loss
            + args.lambda_stop * stop_loss
            + lambda_rank * rank_loss
            + lambda_num * num_loss
            + lambda_margin * bce_margin
            + lambda_k * reg_k
            + 1e-3 * aux.get("sparse", torch.tensor(0.0, device=device))
            + 1e-3 * aux.get("halt_over", torch.tensor(0.0, device=device))
            + float(getattr(args, "lambda_act_entropy", 1e-3)) * aux.get("act_entropy", torch.tensor(0.0, device=device))
            - float(getattr(args, "lambda_fn_adv", 1e-4)) * aux.get("fn_adv", torch.tensor(0.0, device=device))
            - float(getattr(args, "lambda_fn_use", 0.05)) * fn_use_term
        )

        # Final guard: sanitize total loss to avoid inf/nan propagating into backward
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)

        # Consistency loss with EMA teacher for numeric head (digit-dropout view)
        if getattr(model, 'num_head_teacher', None) is not None and args.lambda_consistency > 0:
            try:
                # unlabeled pool: union of numbers in current sub-batches
                nums_uni = []
                for t in [
                    data_add.get("a"), data_add.get("b"), data_add.get("target"),
                    data_off.get("a"), data_off.get("target"),
                    data_cnt.get("a"), data_cnt.get("target"),
                ]:
                    if t is not None:
                        nums_uni.append(t.to(device))
                if len(nums_uni) > 0:
                    Nset = torch.unique(torch.cat(nums_uni))
                    # pack clean texts
                    clean_texts = [str(int(v.item())) for v in Nset]
                    # apply digit-dropout to create a noisy view (mask a non-leading digit)
                    def _digit_dropout_str(s: str) -> str:
                        if len(s) <= 1 or not s.isdigit():
                            return s
                        import random as _r
                        idxs = [i for i in range(1, len(s)) if s[i].isdigit()]
                        if not idxs:
                            return s
                        j = _r.choice(idxs)
                        # replace with same digit to keep magnitude? here we use identity most times; occasionally flip
                        if _r.random() < 0.5:
                            return s[:j] + s[j] + s[j + 1:]
                        return s[:j] + str((int(s[j]) + 1) % 10) + s[j + 1:]
                    noisy_texts = [_digit_dropout_str(t) for t in clean_texts]
                    ids_clean, mask_clean = _pack_text_batch(clean_texts, tok, max_len=seq_len, device=device)
                    ids_noisy, mask_noisy = _pack_text_batch(noisy_texts, tok, max_len=seq_len, device=device)
                    with torch.no_grad():
                        v_teacher = model.num_head_teacher(model.enc(ids_clean, mask_clean)[0]).squeeze(-1)
                    v_student = model.predict_scalar(ids_noisy, mask_noisy).squeeze(-1)
                    cons_loss = torch.nn.functional.mse_loss(v_student, v_teacher)
                    loss = loss + args.lambda_consistency * cons_loss
            except Exception:
                pass

        opt.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ema is not None:
            ema.update(model)
        # update EMA teacher for numeric head
        if getattr(model, 'num_head_teacher', None) is not None and num_ema is not None:
            with torch.no_grad():
                m = num_ema["decay"]
                for p_t, p_s in zip(model.num_head_teacher.parameters(), model.num_head.parameters()):
                    p_t.copy_(m * p_t + (1.0 - m) * p_s)

        # VQ hygiene: periodic dead-code reinit
        try:
            if (step + 1) % int(getattr(args, "vq_reinit_every", 2000)) == 0:
                nreset = model.rvq.reinit_dead_codes(step + 1, util_floor=float(getattr(args, "vq_util_floor", 0.02)), min_interval=int(getattr(args, "vq_reinit_every", 2000)))
                if nreset > 0:
                    print(f"[vq] step={step + 1} reinitialized {nreset} dead codes")
        except Exception:
            pass

        # TTL tick for drafted temporary functions
        try:
            model.reasoner.ttl_tick(1)
        except Exception:
            pass

        # ---------------- Draft-by-demonstration (1-shot function drafting) ------
        try:
            if int(getattr(args, "draft_enable", 1)) > 0 and 'ids_pairs' in locals() and ids_pairs.size(0) > 0 and not args.frozen_mix:
                # Group by (rel, template) among the pair-front segment
                rel_pairs = batch.get("rel")
                if rel_pairs is not None:
                    from collections import defaultdict
                    groups = defaultdict(list)
                    tpl_list = tpl_ids_pairs.detach().cpu().tolist()
                    rel_list = rel_pairs.detach().cpu().tolist()
                    for i in range(ids_pairs.size(0)):
                        groups[(int(rel_list[i]), int(tpl_list[i]))].append(i)
                    # pick up to N groups with support>=K
                    K = int(getattr(args, "draft_min_support", 8))
                    beam = int(getattr(args, "draft_beam", 3))
                    max_len_d = int(getattr(args, "draft_max_len", 3))
                    min_len_d = int(getattr(args, "draft_min_len", 2))
                    min_margin = float(getattr(args, "draft_min_margin", 0.5))
                    ttl = int(getattr(args, "draft_ttl", 1000))
                    bias = float(getattr(args, "draft_bias", 0.7))
                    todo = [k for k, v in groups.items() if len(v) >= K]
                    todo = todo[: int(getattr(args, "draft_groups_per_step", 1))]
                    # Prepare baseline margins for candidate evaluation
                    YES_IDX = num_numbers; NO_IDX = num_numbers + 1
                    def yes_no_margin(logits):
                        return (logits[:, YES_IDX] - logits[:, NO_IDX])
                    for gk in todo:
                        idxs = groups[gk]
                        ids_g = ids_pairs[idxs]
                        mask_g = mask_pairs[idxs]
                        with torch.no_grad():
                            _, logits_g0, _, _, _, _ = model(ids_g, mask_g)
                            base_margin = yes_no_margin(logits_g0).mean().item()
                        # Build candidate sequences via simple beam over primitive indices
                        P = len(getattr(model.reasoner, 'prims', []))
                        # start with all length-2 candidates scored by improvement
                        from concept_learner.reasoning_ops import StepSpec, FnCandidate
                        def eval_seq(seq: list[int]) -> float:
                            # install temp, run, measure improvement, then clear
                            steps = [StepSpec(kind='primitive', idx=int(j)) for j in seq] + [StepSpec(kind='return')]
                            cand = FnCandidate(steps=steps)
                            sids = model.reasoner.install_temp_functions([cand], ttl=ttl, bias=bias)
                            try:
                                with torch.no_grad():
                                    _, lg, _, _, _, _ = model(ids_g, mask_g)
                                    marg = yes_no_margin(lg).mean().item()
                                return float(marg - base_margin)
                            finally:
                                if sids:
                                    model.reasoner.clear_slots(sids)
                        scores2 = []
                        for a0 in range(P):
                            for a1 in range(P):
                                sc = eval_seq([a0, a1])
                                scores2.append(((a0, a1), sc))
                        scores2.sort(key=lambda x: x[1], reverse=True)
                        beamset = scores2[:beam]
                        best_seq = list(beamset[0][0]) if beamset else []
                        best_gain = beamset[0][1] if beamset else -1e9
                        # try extending to length-3
                        if max_len_d >= 3 and beamset:
                            scores3 = []
                            for (s2, _) in beamset:
                                for a2 in range(P):
                                    sc = eval_seq(list(s2) + [a2])
                                    scores3.append((list(s2) + [a2], sc))
                            scores3.sort(key=lambda x: x[1], reverse=True)
                            if scores3 and scores3[0][1] > best_gain:
                                best_seq, best_gain = list(scores3[0][0]), float(scores3[0][1])
                        # decide install
                        if len(best_seq) >= min_len_d and best_gain >= min_margin:
                            steps_best = [StepSpec(kind='primitive', idx=int(j)) for j in best_seq] + [StepSpec(kind='return')]
                            sid_new = model.reasoner.install_temp_functions([FnCandidate(steps=steps_best)], ttl=ttl, bias=bias)
                            if sid_new:
                                print(f"[draft] installed slot={sid_new[0]} seq={best_seq} gain={best_gain:.3f} support={len(idxs)} key={gk}")
        except Exception:
            pass

        # ---------------- Hard-negative mining (templates & steps) ---------------
        try:
            if int(getattr(args, "hnm_enable", 1)) > 0 and (step + 1) % int(getattr(args, "hnm_every", 200)) == 0:
                rel_ids = _parse_relations_arg(getattr(args, "relations", None))
                # in-distribution train range
                def parse_range(s):
                    if not s:
                        return None
                    try:
                        lo, hi = s.split("-")
                        return (int(lo), int(hi))
                    except Exception:
                        return None
                train_idx_range = parse_range(getattr(args, "train_range", None))
                pool_n = max(args.batch_size, int(getattr(args, "hnm_pool_mult", 2.0) * args.batch_size))
                batch_pool = gen.sample_posneg_pairs(batch=pool_n, allowed_relations=rel_ids, idx_range=train_idx_range)
                # Choose a template id for each example and build texts deterministically per (rel, tpl)
                a_l = batch_pool["a_idx"].tolist(); b_l = batch_pool["b_idx"].tolist(); r_l = batch_pool["rel"].tolist()
                tpl_ids: list[int] = []
                texts: list[str] = []
                for ai, bi, ri in zip(a_l, b_l, r_l):
                    tpls = _pair_templates(ai, bi, int(ri))
                    if len(tpls) == 0:
                        tpls = [f"rel({ai},{bi}) ?"]
                    # deterministic but diverse: use (ai+bi) mod len
                    tidx = int((ai + bi) % len(tpls))
                    tpl_ids.append(tidx)
                    texts.append(tpls[tidx])
                # pack
                ids_hnm, mask_hnm = _pack_text_batch(texts, tok, max_len=seq_len, device=device)
                y_hnm = batch_pool["label"].to(device)
                # forward
                with torch.no_grad():
                    _, logits_seq_hnm, _, _, stop_logits_hnm, _ = model(ids_hnm, mask_hnm)
                    logits_seq_hnm = torch.nan_to_num(logits_seq_hnm, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                    Ch = logits_seq_hnm.size(-1)
                    y_h_safe = y_hnm.clone()
                    y_h_safe[(y_h_safe < 0) | (y_h_safe >= Ch)] = -100
                    loss_items = torch.nn.functional.cross_entropy(logits_seq_hnm.float(), y_h_safe, reduction='none', ignore_index=-100)
                    # predicted stop step (0-based)
                    p_stop = torch.sigmoid(stop_logits_hnm)  # (B,S)
                    step_pred = p_stop.argmax(dim=-1)
                    multi_mask = step_pred >= max(0, int(getattr(args, "hnm_min_steps", 2)) - 1)
                # group by (rel, tpl)
                from collections import defaultdict
                groups = defaultdict(list)
                for i, (ri, ti) in enumerate(zip(r_l, tpl_ids)):
                    groups[(int(ri), int(ti))].append(i)
                cap = int(getattr(args, "hnm_cap_per_template", 32))
                frac = float(getattr(args, "hnm_frac", 0.25))
                target_total = max(1, int(frac * ids_hnm.size(0)))
                # rank within groups
                selected = []
                for key, idxs in groups.items():
                    li = loss_items[idxs]
                    k = min(len(idxs), cap, max(1, int(frac * len(idxs))))
                    topk = torch.topk(li, k=k).indices.tolist()
                    selected.extend([idxs[j] for j in topk])
                # trim globally to target_total by highest loss
                if len(selected) > target_total:
                    li_sel = loss_items[selected]
                    ord_idx = torch.topk(li_sel, k=target_total).indices.tolist()
                    selected = [selected[j] for j in ord_idx]
                # enforce multi-step quota
                target_multi = int(float(getattr(args, "hnm_multi_frac", 0.35)) * max(1, len(selected)))
                sel_set = set(selected)
                cur_multi = int(multi_mask[selected].sum().item()) if len(selected) > 0 else 0
                if cur_multi < target_multi:
                    # add more multi-step from remaining pool by loss
                    rem = [i for i in range(ids_hnm.size(0)) if (i not in sel_set) and bool(multi_mask[i].item())]
                    if len(rem) > 0:
                        li_rem = loss_items[rem]
                        need = target_multi - cur_multi
                        add_idx = torch.topk(li_rem, k=min(need, len(rem))).indices.tolist()
                        selected.extend([rem[j] for j in add_idx])
                # build final tensors
                if len(selected) > 0:
                    ids_sel = ids_hnm[selected]
                    mask_sel = mask_hnm[selected]
                    y_sel = y_hnm[selected]
                    # one extra optimization step (sequence CE + VQ regularizer only)
                    with amp_ctx():
                        _lt, logits_seq_e, vq_loss_e, _, _sl, _ = model(ids_sel, mask_sel)
                        logits_seq_e = torch.nan_to_num(logits_seq_e, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                        Ce = logits_seq_e.size(-1)
                        y_e_valid = (y_sel >= 0) & (y_sel < Ce)
                        if y_e_valid.any():
                            ce_e = torch.nn.functional.cross_entropy(logits_seq_e[y_e_valid].float(), y_sel[y_e_valid])
                        else:
                            ce_e = logits_seq_e.new_zeros(())
                        loss_e = ce_e + args.lambda_vq * vq_loss_e
                    opt.zero_grad(set_to_none=True)
                    if use_amp:
                        scaler.scale(loss_e).backward()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss_e.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                    if ema is not None:
                        ema.update(model)
        except Exception:
            pass

        # ---------------- Wake: collect traces into replay buffer ----------------
        try:
            if int(getattr(args, "wake_sleep", 1)) > 0 and getattr(model.reasoner, "_last_traces", None) is not None:
                traces = model.reasoner._last_traces
                total_prims = len(getattr(model.reasoner, "prims", []))
                for bi, tr in enumerate(traces):
                    seq_ids = []
                    for kind, idx in tr:
                        if kind == "prim":
                            seq_ids.append(int(idx))
                        else:
                            seq_ids.append(total_prims + int(idx))
                    if bi < ids.size(0):
                        rb.add(ids[bi].detach().cpu(), mask[bi].detach().cpu(), seq_ids)
        except Exception:
            pass

        # ---------------- Sleep: abstraction + Dreaming (MAP over traces) --------
        try:
            if int(getattr(args, "wake_sleep", 1)) > 0 and (step + 1) % int(getattr(args, "sleep_every", 300)) == 0:
                installed = []
                try:
                    installed = model.reasoner.sleep_abstraction()
                except Exception:
                    installed = []
                # Telemetry each sleep cycle
                telem = getattr(model.reasoner, "_telemetry", {}) or {}
                # Build a compact telemetry summary
                fn_nonempty = telem.get("fn_nonempty", None)
                avg_len = telem.get("avg_flat_len", None)
                max_stack = telem.get("max_stack_depth", None)
                usage = telem.get("slot_usage_ema", None)
                pct_used_fn = telem.get("pct_used_fn", None)
                pct_depth_ge2 = telem.get("pct_depth_ge2", None)
                dlogp = telem.get("delta_logp_fn_vs_prim", None)
                if hasattr(usage, "tolist"):
                    usage = usage.tolist()
                usage_mean = (
                    float(sum(usage) / max(1, len(usage))) if isinstance(usage, (list, tuple)) and len(usage) > 0 else None
                )
                usage_max = (
                    float(max(usage)) if isinstance(usage, (list, tuple)) and len(usage) > 0 else None
                )
                def _r(x):
                    return None if x is None else round(float(x), 3)
                compact = {
                    "fn_nonempty": fn_nonempty,
                    "avg_len": _r(avg_len),
                    "max_stack": max_stack,
                    "usage_mean": _r(usage_mean),
                    "usage_max": _r(usage_max),
                    "%fn_used": _r(pct_used_fn),
                    "%depth>=2": _r(pct_depth_ge2),
                    "Δlogp_fn": _r(dlogp),
                }
                try:
                    rb_len = len(rb)
                except Exception:
                    rb_len = 0
                print(f"[sleep] step={step + 1} installed={len(installed)} rb={rb_len} telem={compact}")
                # Dream-MAP from replay buffer (replayed experiences)
                if len(rb) > 0:
                    dream_bs = min(args.batch_size, len(rb))
                    replays = rb.sample(dream_bs)
                    if replays:
                        ids_rep = torch.stack([x.ids for x in replays], dim=0).to(device)
                        mask_rep = torch.stack([x.mask for x in replays], dim=0).to(device)
                        traces_rep = [x.trace for x in replays]
                        # compress to macro using current library
                        patterns = extract_primitive_patterns_from_library(model.reasoner)
                        traces_canon = [
                            compress_trace_with_patterns(tr, patterns, len(getattr(model.reasoner, 'prims', [])))
                            for tr in traces_rep
                        ]
                        y_tok = torch.zeros(ids_rep.size(0), seq_len, dtype=torch.long, device=device)
                        y_seq = torch.zeros(ids_rep.size(0), dtype=torch.long, device=device)
                        y_stop = make_stop_targets_from_traces(traces_canon, max_steps=getattr(model.reasoner, 'max_steps', 1), device=device)
                        _ = train_step_with_trace_supervision(
                            model, (ids_rep, mask_rep, y_tok, y_seq, y_stop), traces_canon, opt
                        )
                        # structured combos to diversify
                        combos = build_composite_from_replays(
                            replays, target_count=max(1, dream_bs // 2), T=seq_len, device=device
                        )
                        if combos is not None and combos[0].size(0) > 0:
                            ids_c, mask_c, traces_c = combos
                            traces_c_canon = [
                                compress_trace_with_patterns(tr, patterns, len(getattr(model.reasoner, 'prims', [])))
                                for tr in traces_c
                            ]
                            y_tok_c = torch.zeros(ids_c.size(0), seq_len, dtype=torch.long, device=device)
                            y_seq_c = torch.zeros(ids_c.size(0), dtype=torch.long, device=device)
                            y_stop_c = make_stop_targets_from_traces(traces_c_canon, max_steps=getattr(model.reasoner, 'max_steps', 1), device=device)
                            _ = train_step_with_trace_supervision(
                                model, (ids_c, mask_c, y_tok_c, y_seq_c, y_stop_c), traces_c_canon, opt
                            )
                        # Fantasies: sample random programs from the learned library (functions + primitives)
                        try:
                            def _sample_fantasy_traces(reasoner, k: int, max_steps: int):
                                total = []
                                num_prims = len(getattr(reasoner, 'prims', []))
                                num_fn = 0
                                nonempty_fns = []
                                if getattr(reasoner, 'use_functions', False) and getattr(reasoner, 'op_function', None) is not None:
                                    slots = getattr(reasoner.op_function, 'slots', [])
                                    for sid, sl in enumerate(slots):
                                        if hasattr(sl, 'steps') and len(sl.steps) > 0:
                                            nonempty_fns.append(sid)
                                    num_fn = len(nonempty_fns)
                                import random as _r
                                for _ in range(k):
                                    L = max(1, min(max_steps, _r.randint(1, max_steps)))
                                    trace = []
                                    for __ in range(L):
                                        # mix: 50% primitives, 50% functions if available
                                        pick_fn = (num_fn > 0) and (_r.random() < 0.5)
                                        if pick_fn:
                                            sid = nonempty_fns[_r.randrange(num_fn)]
                                            trace.append(num_prims + sid)
                                        else:
                                            pid = _r.randrange(max(1, num_prims))
                                            trace.append(pid)
                                    total.append(trace)
                                return total
                            # pair fantasies with observable data (ids/mask) from replay
                            k = min(len(replays), max(1, dream_bs // 2))
                            traces_f = _sample_fantasy_traces(model.reasoner, k, getattr(model.reasoner, 'max_steps', 1))
                            if k > 0 and len(traces_f) == k:
                                ids_f = ids_rep[:k]
                                mask_f = mask_rep[:k]
                                y_tok_f = torch.zeros(k, seq_len, dtype=torch.long, device=device)
                                y_seq_f = torch.zeros(k, dtype=torch.long, device=device)
                                y_stop_f = make_stop_targets_from_traces(traces_f, max_steps=getattr(model.reasoner, 'max_steps', 1), device=device)
                                _ = train_step_with_trace_supervision(
                                    model, (ids_f, mask_f, y_tok_f, y_seq_f, y_stop_f), traces_f, opt
                                )
                        except Exception:
                            # fantasy generation is best-effort; ignore failures
                            pass
        except Exception:
            # keep training even if dreaming fails
            pass

        if (step + 1) % args.log_every == 0:
            with torch.no_grad():
                from concept_learner.utils import select_pair_columns_safe
                lp = select_pair_columns_safe(logits_seq, NO_IDX, YES_IDX, order="ab")
                probs = torch.softmax(lp, dim=-1)[:, 1].mean().item()
                # action entropy snapshot
                try:
                    import torch.nn.functional as _F
                    ap = _F.softmax(action_logits, dim=-1).clamp_min(1e-8)
                    ent = (-(ap * ap.log()).sum(dim=-1)).mean().item()
                    act_ent_hist.append(float(ent))
                except Exception:
                    pass
            # quick validation for best checkpoint tracking
            # VQ usage (unique codes per quantizer / codebook_size)
            vq_util = None
            try:
                ks = [model.rvq.codebook_size] * int(args.parallel_heads) + [
                    model.rvq.serial_codebook_size
                ] * int(args.serial_heads)
                utils = []
                for idx_t, K in zip(indices, ks):
                    u = len(torch.unique(idx_t)) / max(1, K)
                    utils.append(float(u))
                if len(utils) > 0:
                    vq_util = float(sum(utils) / len(utils))
            except Exception:
                vq_util = None

            # if EMA is enabled, evaluate with EMA weights
            _using_ema = ema is not None
            if _using_ema:
                ema.store(model)
                ema.copy_to(model)
            # Fit/update probe right before eval if enabled
            if int(getattr(args, "probe_enable", 1)) > 0:
                probe.fit(device)
            # compute decayed alpha for knn mixing
            knn_alpha_val = float(getattr(args, 'knn_alpha', 0.2))
            if int(getattr(args, 'knn_alpha_decay_steps', 0)) > 0:
                t = min(1.0, max(0.0, (step - start_step) / float(max(1, int(getattr(args, 'knn_alpha_decay_steps', 0))))) )
                knn_alpha_val = (1.0 - t) * float(getattr(args, 'knn_alpha', 0.2)) + t * float(getattr(args, 'knn_alpha_end', 0.1))
            val_acc_pair = _quick_eval(
                model,
                gen,
                tok,
                device,
                batch_size=min(args.batch_size, 128),
                eval_batches=10,
                max_len=seq_len,
                num_numbers=num_numbers,
                relations=getattr(args, "relations", None),
                probe=(probe if int(getattr(args, "probe_enable", 1)) > 0 else None),
                probe_alpha=getattr(args, "probe_alpha", 0.7),
                proto=proto,
                proto_space=str(getattr(args, "proto_space", "typed")),
                knn=(knn if int(getattr(args, 'knn_enable', 1)) > 0 else None),
                knn_space=str(getattr(args, 'knn_space', 'typed')),
                knn_alpha=knn_alpha_val,
                knn_alpha_end=float(getattr(args, 'knn_alpha_end', 0.1)),
            )
            # quick counting eval
            model.eval()
            tot, hit = 0, 0
            with torch.no_grad():
                for _ in range(5):
                    d2 = gen.sample_counting(batch=min(args.batch_size, 128))
                    i2, m2 = _pack_count_examples_text(
                        d2["kind"], d2["a"], d2["c"], tok, max_len=seq_len
                    )
                    y2 = d2["target"].to(device)
                    _, ls2, _, _, _, _ = model(i2, m2)
                    pred2 = ls2[:, :num_numbers].argmax(dim=-1)
                    hit += (pred2 == y2).sum().item()
                    tot += y2.numel()
            model.train()
            if _using_ema:
                ema.restore(model)
            val_acc_cnt = hit / max(1, tot)
            # avg used for telemetry and LR scheduling
            val_acc = 0.5 * (val_acc_pair + val_acc_cnt)
            if plateau_scheduler is not None:
                plateau_scheduler.step(val_acc)
                # Reapply head LR multiplier after scheduler update
                _apply_head_lr(step)
            cur_lr = opt.param_groups[0]["lr"]
            # VQ diagnostics
            vq_diag_str = ""
            avg_vq_chg = None
            try:
                ks = [model.rvq.codebook_size] * int(args.parallel_heads) + [
                    model.rvq.serial_codebook_size
                ] * int(args.serial_heads)
                parts = []
                changes_for_avg = []
                for h, (idx_t, K) in enumerate(zip(indices, ks)):
                    u = len(torch.unique(idx_t)) / max(1, K)
                    counts = torch.bincount(idx_t.view(-1), minlength=K).float()
                    p = counts / counts.sum().clamp_min(1.0)
                    H = -(p * (p + 1e-8).log()).sum().item()
                    ppl = float(torch.exp(torch.tensor(H)).item() / max(1, K))
                    parts.append(f"h{h}:util={u:.2f},ppl={ppl:.2f}")
                vq_diag_str = " ".join(parts)
                # stability under stochastic training noise: run twice in TRAIN mode
                prev_mode = model.training
                model.train()
                probe_ids = ids[: min(128, ids.size(0))]
                probe_mask = mask[: min(128, mask.size(0))]
                with torch.no_grad():
                    _, _, _, ind1, _, _ = model(probe_ids, probe_mask)
                    _, _, _, ind2, _, _ = model(probe_ids, probe_mask)
                stab = []
                for h in range(min(len(ind1), len(ks))):
                    a1 = ind1[h].view(-1)
                    a2 = ind2[h].view(-1)
                    changed = (a1 != a2).float().mean().item()
                    stab.append(f"h{h}:chg={changed:.3f}")
                    changes_for_avg.append(float(changed))
                vq_diag_str += " | stab " + " ".join(stab)
                if len(changes_for_avg) > 0:
                    avg_vq_chg = float(sum(changes_for_avg) / len(changes_for_avg))
                if not prev_mode:
                    model.eval()
            except Exception:
                pass

            # On-demand sleep trigger: if eval avg drops or VQ change spikes
            try:
                if int(getattr(args, "on_demand_sleep", 1)) > 0:
                    # maintain moving average over last N evals
                    N = max(1, int(getattr(args, "ods_window", 5)))
                    eval_hist.append(float(val_acc))
                    if len(eval_hist) > N:
                        eval_hist = eval_hist[-N:]
                    cur_avg = float(sum(eval_hist) / max(1, len(eval_hist)))
                    drop_thr = float(getattr(args, "ods_avg_drop", 0.0175))
                    vq_thr = float(getattr(args, "ods_vq_chg", 0.20))
                    avg_drop = (prev_eval_avg - cur_avg) if (prev_eval_avg is not None) else 0.0
                    drop_flag = prev_eval_avg is not None and (avg_drop > drop_thr)
                    vq_flag = (avg_vq_chg is not None) and (avg_vq_chg > vq_thr)
                    prev_eval_avg = cur_avg
                    support_ok = True
                    if comp_sleep_gate and comp_enabled:
                        support_ok = (max(comp_support.values()) if len(comp_support) > 0 else 0) >= comp_support_min
                    if (drop_flag or vq_flag) and support_ok and len(rb) > 0 and int(getattr(args, "wake_sleep", 1)) > 0:
                        print(
                            f"[on-demand sleep] step={step + 1} reason={'drop' if drop_flag else ''}{'+' if drop_flag and vq_flag else ''}{'vq' if vq_flag else ''} avg_drop={avg_drop:.4f} avg_vq_chg={avg_vq_chg if avg_vq_chg is not None else float('nan'):.3f}"
                        )
                        try:
                            _ = model.reasoner.sleep_abstraction()
                        except Exception:
                            pass
                        # Short dream burst
                        iters = max(1, int(getattr(args, "ods_sleep_iters", 50)))
                        dream_bs = max(1, min(int(getattr(args, "ods_dream_bs", max(8, args.batch_size // 2))), len(rb)))
                        for _it in range(iters):
                            replays = rb.sample(dream_bs)
                            if not replays:
                                break
                            ids_rep = torch.stack([x.ids for x in replays], dim=0).to(device)
                            mask_rep = torch.stack([x.mask for x in replays], dim=0).to(device)
                            traces_rep = [x.trace for x in replays]
                            patterns = extract_primitive_patterns_from_library(model.reasoner)
                            traces_canon = [
                                compress_trace_with_patterns(tr, patterns, len(getattr(model.reasoner, 'prims', [])))
                                for tr in traces_rep
                            ]
                            y_tok = torch.zeros(ids_rep.size(0), seq_len, dtype=torch.long, device=device)
                            y_seq = torch.zeros(ids_rep.size(0), dtype=torch.long, device=device)
                            y_stop = make_stop_targets_from_traces(traces_canon, max_steps=getattr(model.reasoner, 'max_steps', 1), device=device)
                            _ = train_step_with_trace_supervision(
                                model, (ids_rep, mask_rep, y_tok, y_seq, y_stop), traces_canon, opt
                            )
            except Exception:
                # On-demand sleeps are best-effort and must not break training
                pass

            # OOD evaluations
            def parse_range(s):
                if not s:
                    return None
                try:
                    lo, hi = s.split("-")
                    return (int(lo), int(hi))
                except Exception:
                    return None
            in_range = parse_range(getattr(args, "train_range", None))
            ood_range = parse_range(getattr(args, "ood_range", None))
            val_in = _quick_eval(
                model,
                gen,
                tok,
                device,
                batch_size=min(args.batch_size, 128),
                eval_batches=5,
                max_len=seq_len,
                num_numbers=num_numbers,
                relations=getattr(args, "relations", None),
                idx_range=in_range,
                template_filter=tmpl_train,
                probe=(probe if int(getattr(args, "probe_enable", 1)) > 0 else None),
                probe_alpha=getattr(args, "probe_alpha", 0.7),
                proto=proto,
                proto_space=str(getattr(args, "proto_space", "typed")),
            )
            val_range = (
                _quick_eval(
                    model,
                    gen,
                    tok,
                    device,
                    batch_size=min(args.batch_size, 128),
                    eval_batches=5,
                    max_len=seq_len,
                    num_numbers=num_numbers,
                    relations=getattr(args, "relations", None),
                    idx_range=ood_range,
                    template_filter=tmpl_train,
                    probe=(probe if int(getattr(args, "probe_enable", 1)) > 0 else None),
                    probe_alpha=getattr(args, "probe_alpha", 0.7),
                    proto=proto,
                    proto_space=str(getattr(args, "proto_space", "typed")),
                )
                if ood_range is not None
                else float("nan")
            )
            val_tood = _quick_eval(
                model,
                gen,
                tok,
                device,
                batch_size=min(args.batch_size, 128),
                eval_batches=5,
                max_len=seq_len,
                num_numbers=num_numbers,
                relations=getattr(args, "relations", None),
                idx_range=in_range,
                template_filter=tmpl_ood,
                probe=(probe if int(getattr(args, "probe_enable", 1)) > 0 else None),
                probe_alpha=getattr(args, "probe_alpha", 0.7),
                proto=proto,
                proto_space=str(getattr(args, "proto_space", "typed")),
            )
            # boundary-OOD
            b_batch = gen.sample_posneg_pairs_boundary(
                batch=min(args.batch_size, 128), idx_range=in_range
            )
            ids_b, mask_b = _pack_pair_questions_text(
                b_batch, tok, max_len=seq_len, template_filter=tmpl_train
            )
            with torch.no_grad():
                _, logits_b, _, _, _, _ = model(ids_b, mask_b)
                if proto is not None:
                    h_b, _ = model.enc(ids_b, mask_b)
                    s_b = getattr(model.reasoner, "_last_s", None)
                    if s_b is None:
                        s_b = h_b
                    feat_b = ProtoRouter.build_feature(h_b, s_b, getattr(args, "proto_space", "typed"))
                    logits_b = proto.apply_bias(logits_b, feat_b, b_batch.get("rel").to(device), YES_IDX, NO_IDX, train_mode=False)
                from concept_learner.utils import select_pair_columns_safe
                pred_b = select_pair_columns_safe(logits_b, NO_IDX, YES_IDX, order="ab").argmax(dim=-1)
                acc_b = (
                    (pred_b == b_batch["label"].to(device)).float().mean().item()
                )

            diag = ""
            try:
                if (step + 1) % int(getattr(args, "log_every", 50)) == 0 or n_seq_valid == 0:
                    diag = f" | n_seq={n_seq_valid} C={C} y=[{y_min},{y_max}]"
            except Exception:
                diag = ""
            print(
                f"step {step + 1}/{args.steps} loss={loss.item():.4f} seq={loss_seq.item():.4f} "
                f"vq={vq_loss.item():.4f} stop={stop_loss.item():.4f} p(yes)~{probs:.3f} val_pair={val_acc_pair:.3f} val_cnt={val_acc_cnt:.3f} avg={val_acc:.3f} lr={cur_lr:.2e}{diag}"
            )
            if vq_diag_str:
                print(f"  VQ: {vq_diag_str}")
            print(
                f"  eval in-dist={val_in:.3f} range-OOD={val_range:.3f} template-OOD={val_tood:.3f} boundary-OOD={acc_b:.3f}"
            )
            # Curriculum phase promotion (metric-gated)
            try:
                if curr_enable and (step + 1) >= curr_phase_min:
                    advance = False
                    if curr_phase == "BasicsA":
                        # gates: counting strong, in-dist baseline, VQ stable, STOP in range
                        stop_p = 0.0
                        try:
                            stop_p = float(torch.sigmoid(stop_logits).mean().item())
                        except Exception:
                            pass
                        vq_ok = (avg_vq_chg is not None and avg_vq_chg <= thr_vq_A)
                        if (val_acc_cnt >= thr_cnt_A) and (val_in >= thr_in_A) and vq_ok and (thr_stop_lo_A <= stop_p <= thr_stop_hi_A):
                            advance = True
                    elif curr_phase == "BasicsB":
                        # gates: boundary-OOD ~ mid, template-OOD close to in-dist
                        import math as _m
                        tood_gap = float('inf') if _m.isnan(float(val_tood)) else abs(val_tood - val_in)
                        if (acc_b >= thr_bdry_B_lo and acc_b <= thr_bdry_B_hi) and (tood_gap <= thr_tood_gap):
                            advance = True
                    elif curr_phase == "Structure":
                        telem = getattr(model.reasoner, "_telemetry", {})
                        pct_d2 = float(telem.get("pct_depth_ge2", 0.0) or 0.0)
                        import math as _m
                        tood_gap = float('inf') if _m.isnan(float(val_tood)) else abs(val_tood - val_in)
                        if (pct_d2 > 0.0) and (tood_gap <= thr_tood_gap):
                            advance = True
                    if advance and (curr_idx + 1 < len(curr_groups)):
                        curr_idx += 1
                        curr_phase = curr_groups[curr_idx]
                        print(f"  [curriculum] advancing to phase={curr_phase} (idx={curr_idx})")
            except Exception:
                pass
            # Composition scheduler tick (every comp_tick steps)
            try:
                if comp_enabled and comp_tick > 0 and ((step + 1) % comp_tick == 0):
                    # Gates for Module A → B
                    gates_ok = True
                    if vq_util is None:
                        pass
                    # VQ stability: avg_vq_chg computed below in VQ diag block; recompute quickly if None
                    avg_vq_chg_local = None
                    try:
                        ks = [model.rvq.codebook_size] * int(args.parallel_heads) + [model.rvq.serial_codebook_size] * int(args.serial_heads)
                        changes_for_avg = []
                        probe_ids = ids[: min(128, ids.size(0))]
                        probe_mask = mask[: min(128, mask.size(0))]
                        with torch.no_grad():
                            _, _, _, ind1, _, _ = model(probe_ids, probe_mask)
                            _, _, _, ind2, _, _ = model(probe_ids, probe_mask)
                        for h in range(min(len(ind1), len(ks))):
                            a1 = ind1[h].view(-1)
                            a2 = ind2[h].view(-1)
                            changed = (a1 != a2).float().mean().item()
                            changes_for_avg.append(float(changed))
                        if len(changes_for_avg) > 0:
                            avg_vq_chg_local = float(sum(changes_for_avg) / len(changes_for_avg))
                    except Exception:
                        avg_vq_chg_local = None
                    if avg_vq_chg_local is not None and avg_vq_chg_local > 0.30:
                        gates_ok = False
                    import math as _m
                    if val_in < 0.60:
                        gates_ok = False
                    if not _m.isnan(float(val_range)) and abs(val_range - val_in) > 0.05:
                        gates_ok = False
                    # STOP and action entropy gates
                    try:
                        stop_p = torch.sigmoid(stop_logits).detach().cpu().numpy().mean()
                        if not (0.18 <= stop_p <= 0.35):
                            gates_ok = False
                    except Exception:
                        pass
                    try:
                        if len(act_ent_hist) >= 500:
                            import numpy as _np
                            med500 = _np.median(_np.array(act_ent_hist[:500], dtype=float))
                            cur_ent = _np.median(_np.array(act_ent_hist[-min(200, len(act_ent_hist)):], dtype=float))
                            if cur_ent < med500:
                                gates_ok = False
                    except Exception:
                        pass
                    if comp_stage == "A" and gates_ok:
                        comp_stage = "B"
                        comp_r = max(comp_r, float(getattr(args, "comp_entry", 0.30)))
                    # Adaptive ramp in Module B
                    if comp_stage == "B":
                        telem = getattr(model.reasoner, "_telemetry", {})
                        pct_d2 = float(telem.get("pct_depth_ge2", 0.0) or 0.0)
                        avg_len = float(telem.get("avg_flat_len", 0.0) or 0.0)
                        delta = 0.0
                        if pct_d2 < 0.05 or avg_len < 1.5:
                            delta = float(getattr(args, "comp_ramp_big", 0.10))
                        elif 0.05 <= pct_d2 <= 0.15:
                            delta = float(getattr(args, "comp_ramp_small", 0.05))
                        else:
                            # hold; small rollback if eval dips
                            if abs(val_acc - prev_eval_avg) > float(getattr(args, "comp_eval_drop_thr", 0.03)) if (prev_eval_avg is not None) else False:
                                delta = -float(getattr(args, "comp_ramp_small", 0.05))
                        comp_r = float(min(float(getattr(args, "comp_max", 0.50)), max(float(getattr(args, "comp_min", 0.10)), comp_r + delta)))
                    comp_support = {}
                    print(f"  [comp] stage={comp_stage} r={comp_r:.2f} pct_d2={getattr(model.reasoner, '_telemetry', {}).get('pct_depth_ge2', 0.0)} avg_fn_len={getattr(model.reasoner, '_telemetry', {}).get('avg_flat_len', 0.0)}")
            except Exception:
                pass
            # Selection metric: if an OOD range is specified and evaluated, include it.
            sel_metric = val_acc
            try:
                import math as _m
                if ood_range is not None and not _m.isnan(float(val_range)):
                    sel_metric = (val_acc_pair + val_acc_cnt + float(val_range)) / 3.0
            except Exception:
                sel_metric = val_acc
            if args.save_dir and sel_metric > best_acc:
                best_acc = sel_metric
                best_path = os.path.join(args.save_dir, "best.pt")
                ema_state = ema.state_dict() if ema is not None else None
                save_checkpoint(
                    best_path,
                    model,
                    opt,
                    step + 1,
                    best_acc=best_acc,
                    ema_state=ema_state,
                    proto_state=(proto.state_dict() if proto is not None else None),
                )
                print(
                    f"  Saved best checkpoint: {best_path} (sel={best_acc:.3f}, avg={val_acc:.3f})"
                )

        if args.save_dir and (step + 1) % args.ckpt_every == 0:
            ema_state = ema.state_dict() if ema is not None else None
            if args.save_all_checkpoints:
                ckpt_path = os.path.join(args.save_dir, f"ckpt_step_{step + 1}.pt")
                save_checkpoint(ckpt_path, model, opt, step + 1, ema_state=ema_state, proto_state=(proto.state_dict() if proto is not None else None))
            latest = os.path.join(args.save_dir, "latest.pt")
            save_checkpoint(
                latest,
                model,
                opt,
                step + 1,
                best_acc=best_acc if best_acc >= 0 else None,
                ema_state=ema_state,
                proto_state=(proto.state_dict() if proto is not None else None),
            )


@torch.no_grad()
def evaluate(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    min_base = 10 if getattr(args, "base10", False) else args.min_base
    max_base = 10 if getattr(args, "base10", False) else args.max_base
    ecfg = EpisodeConfig(
        max_number=args.max_number,
        max_len=args.max_len,
        min_base=min_base,
        max_base=max_base,
        device=device,
    )
    gen = EpisodeGenerator(ecfg)
    tok = HFTokenizerWrapper("bert-base-cased")
    vocab_size = tok.vocab_size
    num_numbers = ecfg.max_number
    YES_IDX = num_numbers
    NO_IDX = num_numbers + 1
    EVEN_IDX = num_numbers + 2
    ODD_IDX = num_numbers + 3
    model = CLModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_classes=num_numbers + 4,
        pad_id=0,
        cls_id=1,
        max_len=args.max_len,
    ).to(device)
    # For evaluation we don't change wake/sleep or exec mode

    assert args.checkpoint and os.path.isfile(args.checkpoint), "--checkpoint required"
    load_checkpoint(args.checkpoint, model, opt=None, map_location=device)
    model.eval()

    total, correct = 0, 0
    # accumulators for diagnostics
    rel_tot = {i: 0 for i in range(9)}
    rel_hit = {i: 0 for i in range(9)}
    all_probs = []  # p(YES)
    all_labels = []
    all_a = []
    all_b = []
    all_rel = []
    # Optional temperature fitting on a small held-out sample
    T = 1.0
    if args.temp_fit:
        with torch.no_grad():
            logits_stack, labels_stack = [], []
            n_fit = max(1, args.eval_batches // 3)
            for _ in range(n_fit):
                batch = gen.sample_posneg_pairs(batch=args.batch_size)
                ids, mask = _pack_pair_questions_text(batch, tok, max_len=args.max_len)
                y = batch["label"].to(device)
                _, logits_seq_fit, _, _, _, _ = model(ids, mask)
                from concept_learner.utils import select_pair_columns_safe
                lp = select_pair_columns_safe(logits_seq_fit, NO_IDX, YES_IDX, order="ab")
                logits_stack.append(lp)
                labels_stack.append(y)
            if len(logits_stack) > 0:
                L = torch.cat(logits_stack, dim=0)
                Y = torch.cat(labels_stack, dim=0)
                temp = torch.tensor(1.0, device=device, requires_grad=True)
                opt_t = torch.optim.LBFGS([temp], lr=0.5, max_iter=50)

                def closure():
                    opt_t.zero_grad()
                    logits_scaled = L / temp.clamp_min(1e-2)
                    loss = torch.nn.functional.cross_entropy(logits_scaled, Y)
                    loss.backward()
                    return loss

                try:
                    opt_t.step(closure)
                    T = float(temp.detach().clamp(0.05, 10.0).item())
                except Exception:
                    T = 1.0

    _eval_knn = None  # local kNN memory for eval-only mixing
    for _ in range(args.eval_batches):
        batch = gen.sample_posneg_pairs(batch=args.batch_size)
        ids, mask = _pack_pair_questions_text(batch, tok, max_len=args.max_len)
        y = batch["label"].to(device)
        rel = batch.get("rel").to(device)
        a_idx = batch["a_idx"].to(device)
        b_idx = batch["b_idx"].to(device)
        _, logits_seq, _, _, _, _ = model(ids, mask)
        # Optional kNN mixing during eval
        if int(getattr(args, 'knn_enable', 0)) > 0:
            try:
                from concept_learner.knn_memory import KNNSoftClassifier as _K
                # Build (or reuse) a local kNN memory by sampling a larger pool once
                if _eval_knn is None:
                    _eval_knn = _K(num_classes=args.max_number + 4, feat_dim=args.d_model if getattr(args, 'knn_space', 'typed') != 'concat' else (2 * args.d_model), capacity=int(getattr(args, 'knn_capacity', 10000)), k=int(getattr(args, 'knn_k', 32)), tau=float(getattr(args, 'knn_tau', 0.2)), device=device)
                    # Populate from an offline pool
                    with torch.no_grad():
                        pool = gen.sample_posneg_pairs(batch=min(4096, args.batch_size * 16))
                        ids_p, mask_p = _pack_pair_questions_text(pool, tok, max_len=args.max_len)
                        _, lsp, _, _, _, _ = model(ids_p, mask_p)
                        h_pool_p, _ = model.enc(ids_p, mask_p)
                        s_typed_p = getattr(model.reasoner, "_last_s", None)
                        if s_typed_p is None:
                            s_typed_p = h_pool_p
                        feat_p = ProtoRouter.build_feature(h_pool_p, s_typed_p, getattr(args, 'knn_space', 'typed'))
                        y_p = pool["label"].to(device)
                        _eval_knn.add(feat_p, y_p)
                # mix
                h_pool_e, _ = model.enc(ids, mask)
                s_typed_e = getattr(model.reasoner, "_last_s", None)
                if s_typed_e is None:
                    s_typed_e = h_pool_e
                feat_e = ProtoRouter.build_feature(h_pool_e, s_typed_e, getattr(args, 'knn_space', 'typed'))
                p_m = torch.softmax(logits_seq, dim=-1)
                p_k = _eval_knn.posterior(feat_e)
                a = float(getattr(args, 'knn_alpha_end', 0.1))
                p_mix = (a * p_m + (1.0 - a) * p_k).clamp_min(1e-8)
                logits_seq = torch.log(p_mix)
            except Exception:
                pass
        from concept_learner.utils import select_pair_columns_safe
        logits_pair = select_pair_columns_safe(logits_seq, NO_IDX, YES_IDX, order="ab") / T
        pred = logits_pair.argmax(dim=-1)
        probs = torch.softmax(logits_pair, dim=-1)[:, 1]
        correct += (pred == y).sum().item()
        total += y.numel()
        # accumulate per-relation
        for r in range(9):
            sel = rel == r
            if sel.any():
                rel_tot[r] += int(sel.sum().item())
                rel_hit[r] += int((pred[sel] == y[sel]).sum().item())
        # store for calibration and boundary slices
        all_probs.append(probs.detach().cpu())
        all_labels.append(y.detach().cpu())
        all_a.append(a_idx.detach().cpu())
        all_b.append(b_idx.detach().cpu())
        all_rel.append(rel.detach().cpu())

    acc = correct / max(1, total)
    print(f"Eval accuracy over {total} examples: {acc:.4f}")
    # Build unified arrays
    import torch as _t
    probs_all = _t.cat(all_probs)
    labels_all = _t.cat(all_labels)
    a_all = _t.cat(all_a)
    b_all = _t.cat(all_b)
    rel_all = _t.cat(all_rel)

    # Threshold sweep: best by macro-F1 and by binarized Brier
    rel_names = [
        "same_parity",
        "successor",
        "predecessor",
        "add_2",
        "same_tens",
        "same_ones",
        "makes_ten",
        "greater",
        "smaller",
    ]

    # Optional: Template-OOD per relation (using simple holdout split)
    def build_template_filters(holdout_ratio: float = 0.0):
        train_filt, ood_filt = {}, {}
        for r in range(9):
            tlist = _pair_templates(3, 5, r)
            k = len(tlist)
            if k <= 1:
                train_filt[r] = [0]
                ood_filt[r] = []
                continue
            k_tr = max(1, int(k * (1.0 - holdout_ratio)))
            train_filt[r] = list(range(0, k_tr))
            ood_filt[r] = list(range(k_tr, k))
        return train_filt, ood_filt
    tmpl_train, tmpl_ood = build_template_filters(getattr(args, "template_holdout", 0.0))

    # Helper to compute macro-F1 at a threshold
    def macro_f1_at(thr: float) -> float:
        preds = (probs_all >= thr).long()
        f1s = []
        for ridx in range(9):
            sel = (rel_all == ridx)
            if not sel.any():
                continue
            y = labels_all[sel]
            yhat = preds[sel]
            tp = int(((y == 1) & (yhat == 1)).sum().item())
            fp = int(((y == 0) & (yhat == 1)).sum().item())
            fn = int(((y == 1) & (yhat == 0)).sum().item())
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
            f1s.append(f1)
        return float(sum(f1s) / max(1, len(f1s)))

    def brier_bin_at(thr: float) -> float:
        preds = (probs_all >= thr).float()
        return float(((preds - labels_all.float()) ** 2).mean().item())

    thrs = [i / 100.0 for i in range(20, 81)]  # 0.20 .. 0.80
    best_thr_f1 = max(thrs, key=macro_f1_at)
    best_thr_brier = min(thrs, key=brier_bin_at)
    # Nearby points
    def neighbors(center: float, step: float = 0.02):
        return [round(max(0.0, min(1.0, center - step)), 2), round(center, 2), round(min(1.0, center + step), 2)]

    nb_f1 = neighbors(best_thr_f1)
    nb_br = neighbors(best_thr_brier)
    print("Threshold sweep (macro-F1):")
    for t in nb_f1:
        print(f"  thr={t:.2f}  macro-F1={macro_f1_at(t):.3f}")
    print("Threshold sweep (Brier-bin):")
    for t in nb_br:
        print(f"  thr={t:.2f}  brier={brier_bin_at(t):.3f}")

    # Template-OOD accuracy per relation (if holdout > 0)
    if getattr(args, "template_holdout", 0.0) and args.template_holdout > 0.0:
        print("\nTemplate-OOD per relation (held-out templates):")
        with torch.no_grad():
            for ridx, rname in enumerate(rel_names):
                # skip if no OOD templates
                if ridx not in tmpl_ood or len(tmpl_ood[ridx]) == 0:
                    print(f"  {rname:12s}: (no OOD templates)")
                    continue
                batch = gen.sample_posneg_pairs(batch=min(256, args.batch_size), allowed_relations=[ridx])
                ids, mask = _pack_pair_questions_text(batch, tok, max_len=args.max_len, template_filter=tmpl_ood)
                y = batch["label"].to(device)
                _, logits_seq_r, _, _, _, _ = model(ids, mask)
                from concept_learner.utils import select_pair_columns_safe
                logits_pair_r = select_pair_columns_safe(logits_seq_r, NO_IDX, YES_IDX, order="ab") / T
                pred_r = logits_pair_r.argmax(dim=-1)
                acc_r = (pred_r == y).float().mean().item()
                print(f"  {rname:12s}: acc={acc_r:.3f}")

    # Build rows for eval report at best_thr_f1
    used_thr = float(round(best_thr_f1, 2))
    rows = []
    # precompute string rel names for speed
    rel_name_map = {i: n for i, n in enumerate(rel_names)}
    # canonical renderer
    def canonical(a: int, b: int, rname: str) -> str:
        if rname == "successor":
            return f"succ({a}) == {b} ?"
        if rname == "predecessor":
            return f"pred({a}) == {b} ?"
        if rname == "same_ones":
            return f"same_ones({a},{b}) ?"
        if rname == "same_tens":
            return f"same_tens({a},{b}) ?"
        if rname == "same_parity":
            return f"same_parity({a},{b}) ?"
        if rname == "makes_ten":
            return f"ones({a}) + ones({b}) == 10 ?"
        if rname == "add_2":
            return f"{a} + 2 == {b} ?"
        if rname == "greater":
            return f"greater({a},{b}) ?"
        if rname == "smaller":
            return f"smaller({a},{b}) ?"
        return f"rel({a},{b}) ?"

    preds_used = (probs_all >= used_thr).long()
    for i in range(probs_all.size(0)):
        a_i = int(a_all[i].item())
        b_i = int(b_all[i].item())
        ridx = int(rel_all[i].item())
        rname = rel_name_map.get(ridx, str(ridx))
        rows.append({
            "rel": rname,
            "gold": int(labels_all[i].item()),
            "pred": int(preds_used[i].item()),
            "p_yes": float(probs_all[i].item()),
            "a": a_i,
            "b": b_i,
            "canon": canonical(a_i, b_i, rname),
        })

    # Rich, compact evaluation report
    try:
        from concept_learner.eval_report import print_eval_report
        print_eval_report(rows, threshold_used=used_thr)
    except Exception as _e:
        # Fallback to old per-relation summary if report import fails
        print(f"(eval_report unavailable: {_e})")
        print("Per-relation accuracy:")
        for r, name in enumerate(rel_names):
            tot, hit = rel_tot[r], rel_hit[r]
            if tot > 0:
                print(f"  {name:12s}: {hit}/{tot} = {hit / max(1, tot):.3f}")
            else:
                print(f"  {name:12s}: (no samples)")

    # Error slices near boundary for greater/smaller (|a-b| <= 1)
    pred_all = (probs_all >= used_thr).long()
    eq_diff = (a_all - b_all).abs()
    for r, name in [(7, "greater"), (8, "smaller")]:
        sel = rel_all == r
        if sel.any():
            close = sel & (eq_diff <= 1)
            far = sel & (eq_diff > 1)
            if close.any():
                acc_close = (pred_all[close] == labels_all[close]).float().mean().item()
                print(
                    f"{name} close (|a-b|<=1): {int(close.sum())} ex, acc={acc_close:.3f}"
                )
            if far.any():
                acc_far = (pred_all[far] == labels_all[far]).float().mean().item()
                print(
                    f"{name} far   (|a-b|>1):  {int(far.sum())} ex, acc={acc_far:.3f}"
                )

    # Boundary-OOD confusion matrix for greater/smaller at |a-b|<=1
    def confusion(y_true, y_pred):
        tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
        tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
        fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
        fn = int(((y_true == 1) & (y_pred == 0)).sum().item())
        return tp, fp, fn, tn
    for r, name in [(7, "greater"), (8, "smaller")]:
        sel = rel_all == r
        close = sel & (eq_diff <= 1)
        if close.any():
            y_c = labels_all[close]
            p_c = pred_all[close]
            tp, fp, fn, tn = confusion(y_c, p_c)
            print(f"{name} boundary confusion (|a-b|<=1): TP={tp} FP={fp} FN={fn} TN={tn}")

    # Print a few sample predictions with canonical questions and answers
    batch = gen.sample_posneg_pairs(batch=4)
    ids, mask = _pack_pair_questions_text(batch, tok, max_len=args.max_len)
    y = batch["label"].tolist()
    a_idx = batch["a_idx"].tolist()
    b_idx = batch["b_idx"].tolist()
    rel = batch.get("rel")
    rel = rel.tolist() if rel is not None else [None] * len(y)
    _, logits_seq, _, _, _, _ = model(ids, mask)
    from concept_learner.utils import select_pair_columns_safe
    probs = torch.softmax(select_pair_columns_safe(logits_seq, NO_IDX, YES_IDX, order="ab"), dim=-1)[:, 1].tolist()

    def rel_name(r):
        names = [
            "same_parity",
            "successor",
            "predecessor",
            "add_2",
            "same_tens",
            "same_ones",
            "makes_ten",
            "greater",
            "smaller",
        ]
        try:
            return names[r] if r is not None else "(mixed)"
        except Exception:
            return str(r)

    def make_question(a, b, r):
        try:
            nm = rel_names[r]
        except Exception:
            nm = str(r)
        if nm == "successor":
            return f"succ({a}) == {b} ?"
        if nm == "predecessor":
            return f"pred({a}) == {b} ?"
        if nm == "same_ones":
            return f"same_ones({a},{b}) ?"
        if nm == "same_tens":
            return f"same_tens({a},{b}) ?"
        if nm == "same_parity":
            return f"same_parity({a},{b}) ?"
        if nm == "makes_ten":
            return f"ones({a}) + ones({b}) == 10 ?"
        if nm == "add_2":
            return f"{a} + 2 == {b} ?"
        if nm == "greater":
            return f"greater({a},{b}) ?"
        if nm == "smaller":
            return f"smaller({a},{b}) ?"
        return f"rel({a},{b}) ?"

    print("Sample QA predictions:")
    for i, p in enumerate(probs):
        q = make_question(a_idx[i], b_idx[i], rel[i])
        gold = "yes" if y[i] == 1 else "no"
        pred = "yes" if p >= 0.5 else "no"
        print(
            f"  Q: {q}\n     gold={gold} pred={pred} p(yes)={p:.3f} rel={rel_name(rel[i])}"
        )

    # Also show a few equality reformulations using different language
    print("\nSample equality QA (varied phrasing):")
    import random as _r

    with torch.no_grad():
        a_eq = torch.randint(0, gen.n_items, (4,), device=device)
        kind_eq = torch.randint(0, 2, (4,), device=device)
        ids_eq, mask_eq, y_eq = _pack_equality_examples_text(
            a_eq, kind_eq, tok, max_len=args.max_len
        )
        # use two-class slice {NO, YES}
        num_numbers = ecfg.max_number
        NO_IDX = num_numbers + 1
        YES_IDX = num_numbers
        _, logits_seq_eq, _, _, _, _ = model(ids_eq, mask_eq)
        from concept_learner.utils import select_pair_columns_safe
        probs_eq = torch.softmax(select_pair_columns_safe(logits_seq_eq, NO_IDX, YES_IDX, order="ab"), dim=-1)[
            :, 1
        ].tolist()
        y_eq = y_eq.tolist()
        a_eq_l = a_eq.tolist()
        k_eq_l = kind_eq.tolist()
        for i in range(len(y_eq)):
            ai = a_eq_l[i]
            if k_eq_l[i] == 0:
                # successor phrasing variety
                phr = [
                    f"Is the successor of {ai} equal to {ai} + 1?",
                    f"Is next of {ai} = {ai} + 1?",
                    f"Does {ai} + 1 equal the successor of {ai}?",
                    f"Is the successor of {ai} = {(ai + 1) % 1000}?",
                    f"Is next of {ai} equal to {(ai + 1) % 1000}?",
                ]
            else:
                phr = [
                    f"Is the predecessor of {ai} equal to {ai} - 1?",
                    f"Is previous of {ai} = {ai} - 1?",
                    f"Does {ai} - 1 equal the predecessor of {ai}?",
                    f"Is the predecessor of {ai} = {(ai - 1) % 1000}?",
                    f"Is previous of {ai} equal to {(ai - 1) % 1000}?",
                ]
            qtext = phr[_r.randrange(len(phr))]
            gold = "yes" if y_eq[i] == 1 else "no"
            pred = "yes" if probs_eq[i] >= 0.5 else "no"
            print(
                f"  Q: {qtext}\n     gold={gold} pred={pred} p(yes)={probs_eq[i]:.3f}"
            )

    # Counting diagnostics combined in eval
    print("\nCounting evaluation:")
    tot, hit = 0, 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            dc = gen.sample_counting(batch=min(args.batch_size, 128))
            ic, mc = _pack_count_examples_text(
                dc["kind"], dc["a"], dc["c"], tok, max_len=args.max_len
            )
            yc = dc["target"].to(device)
            _, lsc, _, _, _, _ = model(ic, mc)
            predc = lsc[:, : ecfg.max_number].argmax(dim=-1)
            hit += (predc == yc).sum().item()
            tot += yc.numel()
    print(f"Counting eval accuracy: {hit / max(1, tot):.3f}")
    # sample counting QA with variety
    import random as _r

    with torch.no_grad():
        dshow = gen.sample_counting(batch=4)
        ic, mc = _pack_count_examples_text(
            dshow["kind"], dshow["a"], dshow["c"], tok, max_len=args.max_len
        )
        yshow = dshow["target"].tolist()
        _, lss, _, _, _, _ = model(ic, mc)
        predshow = lss[:, : ecfg.max_number].argmax(dim=-1).tolist()
        kind = dshow["kind"].tolist()
        a = dshow["a"].tolist()
        c = dshow["c"].tolist()
        for i in range(len(kind)):
            if kind[i] == 0:
                phr = [
                    f"What is the successor of {a[i]}?",
                    f"What comes after {a[i]}?",
                    f"What number comes after {a[i]}?",
                ]
            elif kind[i] == 1:
                phr = [
                    f"What is the predecessor of {a[i]}?",
                    f"What comes before {a[i]}?",
                    f"What number comes before {a[i]}?",
                ]
            else:
                phr = [
                    f"What number comes between {a[i]} and {c[i]}?",
                    f"Which number is between {a[i]} and {c[i]}?",
                ]
            q = phr[_r.randrange(len(phr))]
            print(f"  Q: {q}\n     gold={yshow[i]} pred={predshow[i]}")


def main():
    p = argparse.ArgumentParser(description="Train/Eval on numeric episode pairs")
    sub = p.add_subparsers(dest="cmd", required=False)

    # shared args
    def add_shared(sp):
        sp.add_argument("--device", type=str, default=None)
        sp.add_argument("--d_model", type=int, default=128)
        sp.add_argument("--max_number", type=int, default=100)
        sp.add_argument("--max_len", type=int, default=32)
        sp.add_argument("--min_base", type=int, default=10)
        sp.add_argument("--max_base", type=int, default=10)
        sp.add_argument("--batch_size", type=int, default=64)
        # Fallback behavior when a batch has zero valid seq labels
        sp.add_argument("--seq_fallback", type=str, default="skip", help="What to do when no valid seq labels in a batch: skip|uniform|self")
        sp.add_argument("--seq_fallback_weight", type=float, default=0.1, help="Weight for fallback seq loss if used")

    pt = sub.add_parser("train", help="Train and save checkpoints")
    add_shared(pt)
    pt.add_argument("--steps", type=int, default=1000)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--weight_decay", type=float, default=0.0)
    pt.add_argument("--label_smoothing", type=float, default=0.0)
    pt.add_argument("--amp", action="store_true", help="Enable CUDA AMP mixed precision to save memory")
    # Schedules / stability tweaks
    pt.add_argument("--head_boost_steps", type=int, default=5000, help="Multiply LR for action/STOP heads during first N steps")
    pt.add_argument("--label_smoothing_start", type=float, default=0.05, help="Initial label smoothing before schedule kicks in")
    pt.add_argument("--label_smoothing_after", type=int, default=8000, help="Step after which to lower label smoothing")
    pt.add_argument("--label_smoothing_final", type=float, default=0.02, help="Final label smoothing after schedule step")
    pt.add_argument("--ema_decay_after", type=int, default=10000, help="Step after which to increase EMA decay")
    pt.add_argument("--ema_decay_final", type=float, default=0.997, help="Final EMA decay after schedule step")
    pt.add_argument("--lambda_fn_use", type=float, default=0.05, help="Aux reward weight for using functions (percent of episodes)")
    # kNN fallback (eval-time smoothing)
    pt.add_argument("--knn_enable", type=int, default=1, help="Enable kNN posterior mixing (1=on, 0=off)")
    pt.add_argument("--knn_capacity", type=int, default=10000, help="Memory capacity (N examples)")
    pt.add_argument("--knn_k", type=int, default=32, help="Number of neighbors")
    pt.add_argument("--knn_tau", type=float, default=0.2, help="Temperature for neighbor weights")
    pt.add_argument("--knn_alpha", type=float, default=0.2, help="Initial mix weight alpha")
    pt.add_argument("--knn_alpha_end", type=float, default=0.1, help="Final mix weight alpha_end")
    pt.add_argument("--knn_alpha_decay_steps", type=int, default=0, help="Linear decay steps from alpha to alpha_end (0 disables)")
    pt.add_argument("--knn_space", type=str, choices=["typed", "pooled", "concat"], default="typed", help="Feature space for kNN: typed-state, pooled encoder, or concat")
    pt.add_argument("--knn_eval_only", type=int, default=1, help="Apply kNN mixing only in eval paths (1=on, 0=also during training loss)")
    # Fast-weights (Hebbian) for sequence head
    pt.add_argument("--fast_enable", type=int, default=1, help="Enable per-episode fast-weights on classifier")
    pt.add_argument("--fast_eta", type=float, default=0.05, help="Hebbian learning rate eta")
    pt.add_argument("--fast_decay", type=float, default=0.90, help="Fast-weight decay per step")
    pt.add_argument("--fast_norm_max", type=float, default=1.0, help="Max Frobenius norm for fast weights")
    # Curriculum (phased training)
    pt.add_argument("--curriculum_enable", type=int, default=0, help="Enable phased curriculum (1=on, 0=off)")
    pt.add_argument("--curr_groups", type=str, default="BasicsA,BasicsB,Structure,Composition", help="Comma-separated phases")
    pt.add_argument("--curr_phase_min_steps", type=int, default=800, help="Min steps in a phase before promotion")
    pt.add_argument("--curr_thr_cnt_A", type=float, default=0.70, help="Counting accuracy gate for BasicsA→B")
    pt.add_argument("--curr_thr_in_A", type=float, default=0.60, help="In-dist gate for BasicsA→B")
    pt.add_argument("--curr_thr_vq_A", type=float, default=0.30, help="Avg VQ churn gate for BasicsA→B")
    pt.add_argument("--curr_thr_stop_lo_A", type=float, default=0.18, help="STOP lower bound for BasicsA→B")
    pt.add_argument("--curr_thr_stop_hi_A", type=float, default=0.35, help="STOP upper bound for BasicsA→B")
    pt.add_argument("--curr_thr_bdry_B_lo", type=float, default=0.45, help="Boundary-OOD lower bound for BasicsB→Structure")
    pt.add_argument("--curr_thr_bdry_B_hi", type=float, default=0.60, help="Boundary-OOD upper bound for BasicsB→Structure")
    pt.add_argument("--curr_thr_tood_gap", type=float, default=0.05, help="Max template-OOD vs in-dist gap")
    # Composition scheduler & bursts
    pt.add_argument("--comp_enable", type=int, default=1, help="Enable compositional data (1=on, 0=off)")
    pt.add_argument("--comp_init", type=float, default=0.15, help="Initial composition share r in Module A")
    pt.add_argument("--comp_entry", type=float, default=0.30, help="Composition share r when promoting to Module B")
    pt.add_argument("--comp_min", type=float, default=0.10)
    pt.add_argument("--comp_max", type=float, default=0.50)
    pt.add_argument("--comp_tick", type=int, default=500, help="Update interval (steps) for adaptive ramp")
    pt.add_argument("--comp_ramp_big", type=float, default=0.10, help="Increase to r when depth<5% or avg_len<1.5")
    pt.add_argument("--comp_ramp_small", type=float, default=0.05, help="Increase to r when depth in [5%,15%]")
    pt.add_argument("--comp_burst_every", type=int, default=400, help="Every N steps, inject compositional template bursts")
    pt.add_argument("--comp_burst_groups", type=int, default=2, help="Number of burst groups per event")
    pt.add_argument("--comp_burst_k_min", type=int, default=12)
    pt.add_argument("--comp_burst_k_max", type=int, default=16)
    pt.add_argument("--comp_support_min", type=int, default=8, help="Min same-template support to allow gated sleep")
    pt.add_argument("--comp_sleep_gate", type=int, default=0, help="Gate on-demand sleep on composition support (1=on)")
    pt.add_argument(
        "--sched",
        type=str,
        default="none",
        choices=["none", "plateau", "cosine"],
        help="LR scheduler to use",
    )
    pt.add_argument("--min_lr", type=float, default=1e-6)
    pt.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for cosine schedule",
    )
    pt.add_argument("--plateau_factor", type=float, default=0.5)
    pt.add_argument("--plateau_patience", type=int, default=800)
    pt.add_argument(
        "--ema_decay", type=float, default=0.0, help="EMA decay (0 to disable)"
    )
    # VQ architecture knobs
    pt.add_argument("--parallel_heads", type=int, default=2)
    pt.add_argument("--serial_heads", type=int, default=1)
    pt.add_argument("--vq_dim", type=int, default=None)
    pt.add_argument("--vq_codebook_size", type=int, default=None)
    pt.add_argument("--serial_codebook_size", type=int, default=None)
    pt.add_argument("--vq_commitment_weight", type=float, default=None)
    pt.add_argument("--pre_vq_noise_std", type=float, default=None)
    pt.add_argument("--vq_orth_weight", type=float, default=None)
    pt.add_argument("--vq_entropy_weight", type=float, default=None)
    pt.add_argument("--lambda_vq", type=float, default=0.1)
    pt.add_argument("--lambda_stop", type=float, default=0.1)
    pt.add_argument("--save_dir", type=str, default="runs/episodes")
    pt.add_argument("--ckpt_every", type=int, default=200)
    pt.add_argument("--log_every", type=int, default=50)
    # VQ hygiene
    pt.add_argument("--vq_util_floor", type=float, default=0.02, help="Utilization floor to consider a code dead (0..1)")
    pt.add_argument("--vq_reinit_every", type=int, default=2000, help="Reinitialize dead codes every N steps (cool-down)")
    pt.add_argument("--save_knn", type=int, default=0, help="Include kNN snapshot in checkpoints (0=off,1=on)")
    pt.add_argument("--save_knn_best", type=int, default=1, help="Save kNN snapshot with best.pt")
    pt.add_argument("--save_knn_every", type=int, default=0, help="Also save kNN snapshot every N steps (0 disables)")
    # Reasoner/action-selection knobs
    pt.add_argument("--max_steps", type=int, default=4, help="Reasoner max steps per episode")
    pt.add_argument("--epsilon", type=float, default=0.03, help="Epsilon-greedy exploration (train only)")
    pt.add_argument("--topk_actions", type=int, default=4, help="Top-K actions considered per step")
    pt.add_argument("--lambda_act_entropy", type=float, default=1e-3, help="Weight for action entropy bonus")
    pt.add_argument("--lambda_fn_adv", type=float, default=1e-4, help="Tiny credit when functions beat primitives on logits")
    pt.add_argument("--stop_bias", type=float, default=-0.25, help="Additive bias on STOP logits (negative lowers STOP)")
    # Function install filters & post-install bias
    pt.add_argument("--mine_min_support", type=int, default=16, help="Minimum support for mined functions")
    pt.add_argument("--mine_max_len", type=int, default=4, help="Maximum mined body length")
    pt.add_argument("--mine_min_body_len", type=int, default=2, help="Minimum mined body length")
    pt.add_argument("--mine_min_effect", type=float, default=1e-3, help="Minimum estimated effect for a function body")
    pt.add_argument("--mine_cosine_max", type=float, default=0.90, help="Max cosine similarity vs existing bodies")
    pt.add_argument("--mine_overlap_max", type=float, default=0.70, help="Max Jaccard overlap vs existing bodies")
    pt.add_argument("--fn_bias_init", type=float, default=1.0, help="Initial logit bias for newly installed functions")
    pt.add_argument("--fn_bias_decay", type=float, default=0.995, help="Exponential decay for post-install bias (per forward)")
    # Wake/Sleep/Dream (enabled by default)
    pt.add_argument(
        "--wake_sleep", type=int, default=1, help="Enable wake/sleep/dream cycle (1=on, 0=off)"
    )
    pt.add_argument(
        "--sleep_every", type=int, default=300, help="Run sleep-abstraction + dreaming every N steps"
    )
    # Linear probe (few-shot ridge) knobs
    pt.add_argument("--probe_enable", type=int, default=1, help="Enable ridge probe blending for pair eval (1=on, 0=off)")
    pt.add_argument("--probe_max_items", type=int, default=2048, help="Rolling buffer size for ridge probe")
    pt.add_argument("--probe_lambda", type=float, default=1e-3, help="Ridge regularization lambda")
    pt.add_argument("--probe_alpha", type=float, default=0.7, help="Blend: alpha*learned + (1-alpha)*ridge on logit scale")
    # On-demand sleep trigger knobs
    pt.add_argument("--on_demand_sleep", type=int, default=1, help="Enable on-demand short sleep when metrics regress (1=on, 0=off)")
    pt.add_argument("--ods_window", type=int, default=5, help="Window size (N evals) for moving-average drop detection")
    pt.add_argument("--ods_avg_drop", type=float, default=0.0175, help="Trigger if moving-average val acc drops by more than this absolute amount (e.g., 0.015=1.5 pts)")
    pt.add_argument("--ods_vq_chg", type=float, default=0.20, help="Trigger if average VQ assignment change exceeds this threshold (0..1)")
    pt.add_argument("--ods_sleep_iters", type=int, default=50, help="Number of short dream iterations to run on trigger")
    pt.add_argument("--ods_dream_bs", type=int, default=32, help="Batch size for each on-demand dream iteration")
    # Execution granularity for function calls
    pt.add_argument(
        "--exec_mode",
        type=str,
        choices=["micro", "macro"],
        default="micro",
        help="Reasoner execution mode during training (default: micro)",
    )
    pt.add_argument(
        "--replay_max_items", type=int, default=512, help="Max items in dream replay buffer"
    )
    pt.add_argument("--train_range", type=str, default=None, help="Train in-distribution range lo-hi (e.g., 0-79)")
    pt.add_argument("--ood_range", type=str, default=None, help="Range-OOD range lo-hi (e.g., 80-99)")
    pt.add_argument("--template_holdout", type=float, default=0.0, help="Fraction of NL templates per relation held out for template-OOD (0..1)")
    pt.add_argument("--resume", type=str, default=None)
    pt.add_argument(
        "--save_all_checkpoints",
        action="store_true",
        help="Also save step-indexed checkpoints (ckpt_step_*.pt). By default only best.pt and latest.pt are kept.",
    )
    pt.add_argument(
        "--same_base",
        action="store_true",
        help="Render both items with the same base per pair (easier task)",
    )
    pt.add_argument(
        "--overfit_batch",
        type=int,
        default=None,
        help="Overfit a single cached batch of this size for sanity",
    )
    pt.add_argument(
        "--base10",
        action="store_true",
        help="Force base-10 rendering only (min_base=max_base=10)",
    )
    pt.add_argument(
        "--relations",
        type=str,
        default="all",
        help=(
            "Comma-separated relation names or ids to train on (or 'all'). "
            "Pair relations: same_parity, successor, predecessor, add_2, same_tens, same_ones, makes_ten, greater, smaller. "
            "Special: place_value, face_value (controls inclusion of place/face value tasks)."
        ),
    )
    # Draft-by-demonstration (1-shot function drafting)
    pt.add_argument("--draft_enable", type=int, default=1, help="Enable draft-by-demonstration installs (1=on, 0=off)")
    pt.add_argument("--draft_min_support", type=int, default=8, help="Min batch support (same template) to trigger drafting")
    pt.add_argument("--draft_beam", type=int, default=3, help="Beam width over primitive sequences")
    pt.add_argument("--draft_max_len", type=int, default=3, help="Max primitive length for drafted function")
    pt.add_argument("--draft_min_len", type=int, default=2, help="Min primitive length for drafted function")
    pt.add_argument("--draft_min_margin", type=float, default=0.5, help="Required avg YES-NO margin improvement to accept draft")
    pt.add_argument("--draft_ttl", type=int, default=1000, help="TTL (steps) for drafted functions")
    pt.add_argument("--draft_bias", type=float, default=0.7, help="Initial action-logit bias for drafted functions")
    pt.add_argument("--draft_groups_per_step", type=int, default=1, help="Max (rel,template) groups to process per step")
    # Frozen sampler mix (fixed ratios) for pair comparisons
    pt.add_argument("--frozen_mix", action="store_true", help="Use fixed-ratio batch composition for compare tasks")
    pt.add_argument("--mix_ratio_id", type=float, default=0.60)
    pt.add_argument("--mix_ratio_bd", type=float, default=0.20)
    pt.add_argument("--mix_ratio_ood", type=float, default=0.15)
    pt.add_argument("--mix_ratio_cf", type=float, default=0.05)
    # Dynamic reweighting targets for boundary and OOD
    pt.add_argument("--target_bd_acc", type=float, default=0.85)
    pt.add_argument("--target_ood_acc", type=float, default=0.80)
    # EMA teacher for numeric head
    pt.add_argument("--num_ema_decay", type=float, default=0.995)
    pt.add_argument("--lambda_consistency", type=float, default=0.02)
    pt.add_argument("--hard_mine_pct", type=float, default=0.25, help="Fraction in boundary/OOD buckets to select as hardest based on margin")
    # Cosine warm restart step (0 to disable)
    pt.add_argument("--cosine_restart_step", type=int, default=0)
    # Prototypical routing knobs
    pt.add_argument("--proto_enable", type=int, default=1, help="Enable prototypical routing on pair relations (1=on, 0=off)")
    pt.add_argument("--proto_space", type=str, choices=["typed", "pooled"], default="typed", help="Feature space for prototypes: typed-state or encoder pooled")
    pt.add_argument("--proto_beta", type=float, default=0.98, help="EMA decay for prototypes (0..1)")
    pt.add_argument("--proto_weight", type=float, default=0.2, help="Initial bias weight w")
    pt.add_argument("--proto_weight_end", type=float, default=0.1, help="Final bias weight w_end for decay (set equal to keep constant)")
    pt.add_argument("--proto_decay_steps", type=int, default=0, help="Linear decay steps from w to w_end (0 disables decay)")
    pt.add_argument("--proto_warmup", type=int, default=50, help="Per-relation count warmup before applying bias")
    # Hard-negative mining (templates & steps curriculum)
    pt.add_argument("--hnm_enable", type=int, default=1, help="Enable hard-negative mining curriculum (1=on, 0=off)")
    pt.add_argument("--hnm_every", type=int, default=200, help="Mine hard negatives every N steps")
    pt.add_argument("--hnm_frac", type=float, default=0.25, help="Fraction of pool to keep as hard negatives")
    pt.add_argument("--hnm_pool_mult", type=float, default=2.0, help="Pool size multiplier relative to batch_size")
    pt.add_argument("--hnm_min_steps", type=int, default=2, help="Minimum predicted steps (via STOP head) for multi-step quota")
    pt.add_argument("--hnm_multi_frac", type=float, default=0.35, help="Target fraction of multi-step items in mined batch (0..1)")
    pt.add_argument("--hnm_cap_per_template", type=int, default=32, help="Max items selected per (relation, template) group to avoid overfocus")

    pe = sub.add_parser("eval", help="Evaluate from a checkpoint")
    add_shared(pe)
    pe.add_argument("--checkpoint", type=str, required=True)
    pe.add_argument("--eval_batches", type=int, default=50)
    pe.add_argument("--same_base", action="store_true")
    pe.add_argument(
        "--base10",
        action="store_true",
        help="Force base-10 rendering only (min_base=max_base=10)",
    )
    pe.add_argument("--temp_fit", action="store_true", help="Fit a scalar temperature on a held-out split for calibration")
    pe.add_argument("--template_holdout", type=float, default=0.0, help="Fraction of NL templates per relation held out for template-OOD (0..1)")

    # Counting subcommands (unified CLI)
    ct = sub.add_parser(
        "count-train", help="Train counting tasks (successor/pred/between)"
    )
    ct.add_argument("--device", type=str, default=None)
    ct.add_argument("--d_model", type=int, default=128)
    ct.add_argument("--steps", type=int, default=5000)
    ct.add_argument("--lr", type=float, default=3e-4)
    ct.add_argument("--batch_size", type=int, default=128)
    ct.add_argument("--max_number", type=int, default=100)
    ct.add_argument("--save_dir", type=str, default="runs/counting")
    ct.add_argument("--ckpt_every", type=int, default=500)
    ct.add_argument("--log_every", type=int, default=100)
    ct.add_argument("--resume", type=str, default=None)
    ct.add_argument("--max_len", type=int, default=18)
    ct.add_argument("--eval_batches", type=int, default=50)
    ct.add_argument("--weight_decay", type=float, default=0.0)
    ct.add_argument("--amp", action="store_true", help="Enable CUDA AMP mixed precision to save memory")
    ct.add_argument(
        "--save_all_checkpoints",
        action="store_true",
        help="Also save step-indexed checkpoints (ckpt_step_*.pt). By default only best.pt and latest.pt are kept.",
    )

    ce = sub.add_parser("count-eval", help="Evaluate a counting checkpoint")
    ce.add_argument("--device", type=str, default=None)
    ce.add_argument("--d_model", type=int, default=128)
    ce.add_argument("--max_number", type=int, default=100)
    ce.add_argument("--checkpoint", type=str, required=True)
    ce.add_argument("--max_len", type=int, default=18)
    ce.add_argument("--eval_batches", type=int, default=100)

    # Free-form question
    ask = sub.add_parser("ask", help="Ask a free-form question")
    ask.add_argument("--device", type=str, default=None)
    ask.add_argument("--d_model", type=int, default=128)
    ask.add_argument("--max_number", type=int, default=100, help="Number classes (0..N-1) + {YES,NO,EVEN,ODD}")
    ask.add_argument("--checkpoint", type=str, required=True)
    ask.add_argument("--max_len", type=int, default=32)
    ask.add_argument("--text", type=str, required=True, help="Question text to ask")

    args = p.parse_args()
    if args.cmd == "eval":
        evaluate(args)
        return
    if args.cmd == "ask":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = HFTokenizerWrapper("bert-base-cased")
        vocab_size = tok.vocab_size
        num_numbers = args.max_number
        YES_IDX = num_numbers
        NO_IDX = num_numbers + 1
        EVEN_IDX = num_numbers + 2
        ODD_IDX = num_numbers + 3
        model = CLModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_classes=num_numbers + 4,
            pad_id=0,
            cls_id=1,
            max_len=args.max_len,
        ).to(device)
        assert args.checkpoint and os.path.isfile(args.checkpoint), "--checkpoint required"
        load_checkpoint(args.checkpoint, model, opt=None, map_location=device)
        model.eval()
        ids, mask = _pack_text_batch([args.text], tok, args.max_len, device)
        with torch.no_grad():
            _, logits_seq, _, _, _, _ = model(ids, mask)
            pred = logits_seq.argmax(dim=-1)[0].item()
            if pred < num_numbers:
                ans = str(int(pred))
            elif pred == YES_IDX:
                ans = "yes"
            elif pred == NO_IDX:
                ans = "no"
            elif pred == EVEN_IDX:
                ans = "even"
            elif pred == ODD_IDX:
                ans = "odd"
            else:
                ans = str(int(pred))
            print(ans)
        return
    if args.cmd == "count-train":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        ecfg = EpisodeConfig(
            max_number=args.max_number,
            device=device,
            max_len=6,
            min_base=10,
            max_base=10,
        )
        gen = EpisodeGenerator(ecfg)
        tok = HFTokenizerWrapper("bert-base-cased")
        model = CLModel(
            vocab_size=tok.vocab_size,
            d_model=args.d_model,
            num_classes=ecfg.max_number,
            pad_id=0,
            cls_id=1,
            max_len=args.max_len,
        ).to(device)
        # opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.endswith(".bias") or "norm" in n.lower() or "codebook" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": args.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=args.lr,
        )
        # Optional AMP for memory savings
        use_amp_ct = bool(getattr(args, "amp", False)) and (str(device).startswith("cuda") and torch.cuda.is_available())
        amp_ctx_ct = (lambda: torch.amp.autocast("cuda")) if use_amp_ct else nullcontext
        scaler_ct = torch.amp.GradScaler("cuda", enabled=use_amp_ct)

        start = 0
        if args.resume and os.path.isfile(args.resume):
            start = load_checkpoint(args.resume, model, opt, map_location=device)
            print(f"Resumed from {args.resume} at step {start}")
        best = -1.0
        for step in range(start, args.steps):
            data = gen.sample_counting(batch=args.batch_size)
            ids, mask = _pack_count_examples_text(
                data["kind"], data["a"], data["c"], tok, max_len=args.max_len
            )
            y = data["target"].to(device)
            with amp_ctx_ct():
                _, logits_seq, vq_loss, _, stop_logits, _ = model(ids, mask)
                loss = torch.nn.functional.cross_entropy(logits_seq, y)
            opt.zero_grad(set_to_none=True)
            if use_amp_ct:
                scaler_ct.scale(loss).backward()
                scaler_ct.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_ct.step(opt)
                scaler_ct.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if (step + 1) % args.log_every == 0:
                # quick eval
                model.eval()
                tot, hit = 0, 0
                with torch.no_grad():
                    for _ in range(args.eval_batches):
                        data2 = gen.sample_counting(batch=min(args.batch_size, 128))
                        ids2, mask2 = _pack_count_examples_text(
                            data2["kind"],
                            data2["a"],
                            data2["c"],
                            tok,
                            max_len=args.max_len,
                        )
                        y2 = data2["target"].to(device)
                        _, logits_seq2, _, _, _, _ = model(ids2, mask2)
                        pred2 = logits_seq2.argmax(dim=-1)
                        hit += (pred2 == y2).sum().item()
                        tot += y2.numel()
                model.train()
                val_acc = hit / max(1, tot)
                print(
                    f"step {step + 1}/{args.steps} loss={loss.item():.4f} val_acc={val_acc:.3f}"
                )
                if val_acc > best and args.save_dir:
                    best = val_acc
                    os.makedirs(args.save_dir, exist_ok=True)
                    # Optional kNN snapshot with best
                    knn_state = None
                    try:
                        if int(getattr(args, "save_knn", 0)) > 0 and int(getattr(args, "save_knn_best", 1)) > 0 and 'knn' in globals() and knn is not None:
                            knn_state = knn.snapshot(model_hash)
                    except Exception:
                        knn_state = None
                    save_checkpoint(
                        os.path.join(args.save_dir, "best.pt"), model, opt, step + 1, knn_state=knn_state
                    )
                    print(
                        f"  Saved best checkpoint: {os.path.join(args.save_dir, 'best.pt')} (val_acc={val_acc:.3f})"
                    )
            if args.save_dir and (step + 1) % args.ckpt_every == 0:
                # Always maintain rolling latest.pt; optionally archive per-step checkpoint
                if args.save_all_checkpoints:
                    save_checkpoint(
                        os.path.join(args.save_dir, f"ckpt_step_{step + 1}.pt"),
                        model,
                        opt,
                        step + 1,
                        knn_state=(knn.snapshot(model_hash) if (int(getattr(args, 'save_knn', 0)) > 0 and int(getattr(args, 'save_knn_every', 0)) > 0 and ((step + 1) % int(getattr(args, 'save_knn_every', 0)) == 0) and 'knn' in globals() and knn is not None) else None),
                    )
                save_checkpoint(
                    os.path.join(args.save_dir, "latest.pt"), model, opt, step + 1,
                    knn_state=(knn.snapshot(model_hash) if (int(getattr(args, 'save_knn', 0)) > 0 and int(getattr(args, 'save_knn_every', 0)) > 0 and ((step + 1) % int(getattr(args, 'save_knn_every', 0)) == 0) and 'knn' in globals() and knn is not None) else None),
                )
        return
    if args.cmd == "count-eval":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        ecfg = EpisodeConfig(
            max_number=args.max_number,
            device=device,
            max_len=6,
            min_base=10,
            max_base=10,
        )
        gen = EpisodeGenerator(ecfg)
        tok = HFTokenizerWrapper("bert-base-cased")
        model = CLModel(
            vocab_size=tok.vocab_size,
            d_model=args.d_model,
            num_classes=ecfg.max_number,
            pad_id=0,
            cls_id=1,
            max_len=args.max_len,
        ).to(device)
        load_checkpoint(args.checkpoint, model, opt=None, map_location=device)
        model.eval()
        # eval
        tot, hit = 0, 0
        with torch.no_grad():
            for _ in range(args.eval_batches):
                data2 = gen.sample_counting(batch=min(256, 128))
                ids2, mask2 = _pack_count_examples_text(
                    data2["kind"], data2["a"], data2["c"], tok, max_len=args.max_len
                )
                y2 = data2["target"].to(device)
                _, logits_seq2, _, _, _, _ = model(ids2, mask2)
                pred2 = logits_seq2.argmax(dim=-1)
                hit += (pred2 == y2).sum().item()
                tot += y2.numel()
        print(f"Counting eval accuracy: {hit / max(1, tot):.3f}")
        # sample questions
        import random as _r

        with torch.no_grad():
            data = gen.sample_counting(batch=6)
            ids, mask = _pack_count_examples_text(
                data["kind"], data["a"], data["c"], tok, max_len=args.max_len
            )
            y = data["target"].tolist()
            _, logits_seq, _, _, _, _ = model(ids, mask)
            pred = logits_seq.argmax(dim=-1).tolist()
            kind = data["kind"].tolist()
            a = data["a"].tolist()
            c = data["c"].tolist()
            for i in range(len(kind)):
                if kind[i] == 0:
                    phr = [
                        f"What is the successor of {a[i]}?",
                        f"What comes after {a[i]}?",
                        f"What number comes after {a[i]}?",
                    ]
                elif kind[i] == 1:
                    phr = [
                        f"What is the predecessor of {a[i]}?",
                        f"What comes before {a[i]}?",
                        f"What number comes before {a[i]}?",
                    ]
                else:
                    phr = [
                        f"What number comes between {a[i]} and {c[i]}?",
                        f"Which number is between {a[i]} and {c[i]}?",
                    ]
                q = phr[_r.randrange(len(phr))]
                print(f"Q: {q}\n   gold={y[i]} pred={pred[i]}")
        return
    # default to pair training when no subcommand is provided
    train(args if args.cmd == "train" else p.parse_args(["train"]))


if __name__ == "__main__":
    main()
