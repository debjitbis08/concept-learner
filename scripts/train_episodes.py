import argparse
import os
import math
import torch
from typing import List
from concept_learner.episodes import EpisodeConfig, EpisodeGenerator
from concept_learner.model import CLModel
from concept_learner.tokenizer import HFTokenizerWrapper


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
    path, model, opt, step, best_acc: float | None = None, ema_state: dict | None = None
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}
    if best_acc is not None:
        payload["best_acc"] = float(best_acc)
    if ema_state is not None:
        payload["ema"] = ema_state
    torch.save(payload, path)


def load_checkpoint(path, model, opt, map_location=None, ema=None):
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
) -> float:
    model.eval()
    total, correct = 0, 0
    for _ in range(eval_batches):
        rel_ids = _parse_relations_arg(relations)
        batch = gen.sample_posneg_pairs(batch=batch_size, allowed_relations=rel_ids, idx_range=idx_range)
        ids, mask = _pack_pair_questions_text(batch, tok, max_len, template_filter=template_filter)
        y = batch["label"].to(device)
        _, logits_seq, _, _, _, _ = model(ids, mask)
        NO_IDX = num_numbers + 1
        YES_IDX = num_numbers
        pred_bin = logits_seq[:, [NO_IDX, YES_IDX]].argmax(dim=-1)
        correct += (pred_bin == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(1, total)


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
    # Simple natural templates; tokenizer handles subwords; we don't hardcode tokens.
    if r == 0:
        return [f"Do {a} and {b} have the same parity?"]
    if r == 1:
        return [f"Is {b} the successor of {a}?", f"Is {b} equal to {a} + 1?"]
    if r == 2:
        return [f"Is {b} the predecessor of {a}?", f"Is {b} equal to {a} - 1?"]
    if r == 3:
        return [f"Is {b} equal to {a} + 2?"]
    if r == 4:
        return [f"Do {a} and {b} have the same tens digit?"]
    if r == 5:
        return [f"Do {a} and {b} have the same ones digit?"]
    if r == 6:
        return [f"Do the ones digits of {a} and {b} make ten?"]
    if r == 7:
        return [f"Is {a} greater than {b}?"]
    if r == 8:
        return [f"Is {a} smaller than {b}?"]
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
                        f"Between {ai} and {ci}, which number comes in between?",
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
        sym = '>' if oi == 0 else '<'
        candidates = [
            f"{ai}{sym}{bi}?",
            f"{ai} {sym} {bi}?",
            f"Is {ai}{sym}{bi}?",
            f"Is {ai} {sym} {bi}?",
            f"Is {ai} {sym} {bi} true?",
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
    # opt = torch.optim.AdamW(
    #     model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    # )

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

    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        start_step = load_checkpoint(
            args.resume, model, opt, map_location=device, ema=None
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
            best_acc = _quick_eval(
                model,
                gen,
                tok,
                device,
                batch_size=min(args.batch_size, 128),
                eval_batches=10,
                max_len=args.max_len,
                num_numbers=num_numbers,
                relations=getattr(args, "relations", None),
                idx_range=parse_range(getattr(args, "train_range", None)),
                template_filter=tmpl_train,
            )
            print(f"Computed baseline val_acc after resume: {best_acc:.3f}")

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
            t = (step_idx - warmup) / max(1.0, (args.steps - warmup))
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
    for step in range(start_step, args.steps):
        if args.sched == "cosine":
            set_cosine_lr(step)
        rel_arg = getattr(args, "relations", None)
        rel_ids = _parse_relations_arg(rel_arg)
        # Remove training for same_parity (0), same_tens (4), same_ones (5), makes_ten (6)
        if rel_ids is None:
            rel_ids = [1, 2, 3, 7, 8]
        else:
            rel_ids = [r for r in rel_ids if r not in (0, 4, 5, 6)]
        include_pairs = True
        if rel_arg and rel_arg.strip().lower() != "all" and (rel_ids is None or len(rel_ids) == 0):
            include_pairs = False

        if include_pairs:
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
        else:
            # empty pairs batch
            B0 = 0
            device0 = device
            batch = {
                "a_idx": torch.empty(B0, dtype=torch.long, device=device0),
                "b_idx": torch.empty(B0, dtype=torch.long, device=device0),
                "rel": torch.empty(B0, dtype=torch.long, device=device0),
                "label": torch.empty(B0, dtype=torch.long, device=device0),
            }
        # natural-language pair questions
        ids_pairs, mask_pairs = _pack_pair_questions_text(
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

        # counting batch
        cnt_bsz = args.batch_size
        data_cnt = gen.sample_counting(batch=cnt_bsz, idx_range=train_idx_range)
        ids_cnt, mask_cnt = _pack_count_examples_text(
            data_cnt["kind"], data_cnt["a"], data_cnt["c"], tok, max_len=seq_len
        )
        y_cnt = data_cnt["target"].to(device)  # 0..N-1

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
        if include_place:
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

        # symbolic comparison batch (a>b? a<b?) -> YES/NO
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

        # addition batch (a+b=?) -> numeric class
        add_bsz = args.batch_size
        data_add = gen.sample_addition(batch=add_bsz, idx_range=train_idx_range)
        ids_add, mask_add = _pack_addition_text(
            data_add["a"], data_add["b"], tok, max_len=seq_len
        )
        y_add = data_add["target"].to(device)
        ids = torch.cat([ids, ids_add], dim=0)
        mask = torch.cat([mask, mask_add], dim=0)
        y = torch.cat([y, y_add], dim=0)

        # parity batch -> outputs EVEN/ODD classes
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

        # simple +/- k tasks with k in {1,2,3}
        off_bsz = args.batch_size
        data_off = gen.sample_offset(batch=off_bsz, idx_range=train_idx_range)
        ids_off, mask_off = _pack_offset_text(
            data_off["a"], data_off["offset"], tok, max_len=seq_len
        )
        y_off = data_off["target"].to(device)
        ids = torch.cat([ids, ids_off], dim=0)
        mask = torch.cat([mask, mask_off], dim=0)
        y = torch.cat([y, y_off], dim=0)
        logits_tok, logits_seq, vq_loss, indices, stop_logits, _ = model(ids, mask)

        loss_seq = torch.nn.functional.cross_entropy(
            logits_seq, y, label_smoothing=args.label_smoothing
        )

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
        rel = batch.get("rel")
        if rel is not None:
            steps_pairs = torch.tensor([rel_steps_map[int(r.item())] for r in rel], device=device)
        else:
            steps_pairs = torch.ones(ids_pairs.size(0), dtype=torch.long, device=device)
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
        loss = loss_seq + args.lambda_vq * vq_loss + args.lambda_stop * stop_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema is not None:
            ema.update(model)

        if (step + 1) % args.log_every == 0:
            with torch.no_grad():
                lp = logits_seq[:, [NO_IDX, YES_IDX]]
                probs = torch.softmax(lp, dim=-1)[:, 1].mean().item()
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
            val_acc = 0.5 * (val_acc_pair + val_acc_cnt)
            if plateau_scheduler is not None:
                plateau_scheduler.step(val_acc)
            cur_lr = opt.param_groups[0]["lr"]
            # VQ diagnostics
            vq_diag_str = ""
            try:
                ks = [model.rvq.codebook_size] * int(args.parallel_heads) + [
                    model.rvq.serial_codebook_size
                ] * int(args.serial_heads)
                parts = []
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
                vq_diag_str += " | stab " + " ".join(stab)
                if not prev_mode:
                    model.eval()
            except Exception:
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
                pred_b = logits_b[:, [NO_IDX, YES_IDX]].argmax(dim=-1)
                acc_b = (
                    (pred_b == b_batch["label"].to(device)).float().mean().item()
                )

            print(
                f"step {step + 1}/{args.steps} loss={loss.item():.4f} seq={loss_seq.item():.4f} "
                f"vq={vq_loss.item():.4f} stop={stop_loss.item():.4f} p(yes)~{probs:.3f} val_pair={val_acc_pair:.3f} val_cnt={val_acc_cnt:.3f} avg={val_acc:.3f} lr={cur_lr:.2e}"
            )
            if vq_diag_str:
                print(f"  VQ: {vq_diag_str}")
            print(
                f"  eval in-dist={val_in:.3f} range-OOD={val_range:.3f} template-OOD={val_tood:.3f} boundary-OOD={acc_b:.3f}"
            )
            if args.save_dir and val_acc > best_acc:
                best_acc = val_acc
                best_path = os.path.join(args.save_dir, "best.pt")
                ema_state = ema.state_dict() if ema is not None else None
                save_checkpoint(
                    best_path,
                    model,
                    opt,
                    step + 1,
                    best_acc=val_acc,
                    ema_state=ema_state,
                )
                print(f"  Saved best checkpoint: {best_path} (val_acc={val_acc:.3f})")

        if args.save_dir and (step + 1) % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"ckpt_step_{step + 1}.pt")
            ema_state = ema.state_dict() if ema is not None else None
            save_checkpoint(ckpt_path, model, opt, step + 1, ema_state=ema_state)
            latest = os.path.join(args.save_dir, "latest.pt")
            save_checkpoint(
                latest,
                model,
                opt,
                step + 1,
                best_acc=best_acc if best_acc >= 0 else None,
                ema_state=ema_state,
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

    assert args.checkpoint and os.path.isfile(args.checkpoint), "--checkpoint required"
    load_checkpoint(args.checkpoint, model, opt=None, map_location=device)
    model.eval()

    total, correct = 0, 0
    # accumulators for diagnostics
    rel_tot = {i: 0 for i in range(9)}
    rel_hit = {i: 0 for i in range(9)}
    all_probs = []
    all_labels = []
    all_a = []
    all_b = []
    all_rel = []
    for _ in range(args.eval_batches):
        batch = gen.sample_posneg_pairs(batch=args.batch_size)
        ids, mask = _pack_pair_questions_text(batch, tok, max_len=args.max_len)
        y = batch["label"].to(device)
        rel = batch.get("rel").to(device)
        a_idx = batch["a_idx"].to(device)
        b_idx = batch["b_idx"].to(device)
        _, logits_seq, _, _, _, _ = model(ids, mask)
        logits_pair = logits_seq[:, [NO_IDX, YES_IDX]]
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

    # Per-relation accuracy summary
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
    print("Per-relation accuracy:")
    for r, name in enumerate(names):
        tot, hit = rel_tot[r], rel_hit[r]
        if tot > 0:
            print(f"  {name:12s}: {hit}/{tot} = {hit / max(1, tot):.3f}")
        else:
            print(f"  {name:12s}: (no samples)")

    # Calibration / threshold sweep
    import torch as _t

    probs_all = _t.cat(all_probs)
    labels_all = _t.cat(all_labels)
    for thr in [0.4, 0.5, 0.6]:
        acc_thr = ((probs_all >= thr).long() == labels_all).float().mean().item()
        print(f"Acc@thr={thr:.1f}: {acc_thr:.3f}")
    # best among a small grid
    grid = [i / 100.0 for i in range(35, 66, 1)]
    best_thr, best_acc = 0.5, 0.0
    for thr in grid:
        acc_thr = ((probs_all >= thr).long() == labels_all).float().mean().item()
        if acc_thr > best_acc:
            best_acc, best_thr = acc_thr, thr
    print(
        f"Best fixed-threshold acc in [0.35,0.65]: {best_acc:.3f} at thr={best_thr:.2f}"
    )

    # Error slices near boundary for greater/smaller (|a-b| <= 1)
    a_all = _t.cat(all_a)
    b_all = _t.cat(all_b)
    rel_all = _t.cat(all_rel)
    pred_all = (probs_all >= 0.5).long()
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

    # Print a few sample predictions with questions and answers
    batch = gen.sample_posneg_pairs(batch=4)
    ids, mask = _pack_pair_questions_text(batch, tok, max_len=args.max_len)
    y = batch["label"].tolist()
    a_idx = batch["a_idx"].tolist()
    b_idx = batch["b_idx"].tolist()
    rel = batch.get("rel")
    rel = rel.tolist() if rel is not None else [None] * len(y)
    _, logits_seq, _, _, _, _ = model(ids, mask)
    probs = torch.softmax(logits_seq[:, [NO_IDX, YES_IDX]], dim=-1)[:, 1].tolist()

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
        if r == 0:
            return f"Do {a} and {b} have the same parity?"
        if r == 1:
            return f"Is {b} the successor of {a}?"
        if r == 2:
            return f"Is {b} the predecessor of {a}?"
        if r == 3:
            return f"Is {b} equal to {a} + 2?"
        if r == 4:
            return f"Do {a} and {b} have the same tens digit?"
        if r == 5:
            return f"Do {a} and {b} have the same ones digit?"
        if r == 6:
            return f"Do the ones digits of {a} and {b} make ten?"
        if r == 7:
            return f"Is {a} greater than {b}?"
        if r == 8:
            return f"Is {a} smaller than {b}?"
        return f"Do {a} and {b} satisfy the relation?"

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
        probs_eq = torch.softmax(logits_seq_eq[:, [NO_IDX, YES_IDX]], dim=-1)[
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
                    f"Between {a[i]} and {c[i]}, which number comes in between?",
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

    pt = sub.add_parser("train", help="Train and save checkpoints")
    add_shared(pt)
    pt.add_argument("--steps", type=int, default=1000)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--weight_decay", type=float, default=0.0)
    pt.add_argument("--label_smoothing", type=float, default=0.0)
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
    pt.add_argument("--train_range", type=str, default=None, help="Train in-distribution range lo-hi (e.g., 0-79)")
    pt.add_argument("--ood_range", type=str, default=None, help="Range-OOD range lo-hi (e.g., 80-99)")
    pt.add_argument("--template_holdout", type=float, default=0.0, help="Fraction of NL templates per relation held out for template-OOD (0..1)")
    pt.add_argument("--resume", type=str, default=None)
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
            _, logits_seq, vq_loss, _, stop_logits, _ = model(ids, mask)
            loss = torch.nn.functional.cross_entropy(logits_seq, y)
            opt.zero_grad(set_to_none=True)
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
                    save_checkpoint(
                        os.path.join(args.save_dir, "best.pt"), model, opt, step + 1
                    )
                    print(
                        f"  Saved best checkpoint: {os.path.join(args.save_dir, 'best.pt')} (val_acc={val_acc:.3f})"
                    )
            if args.save_dir and (step + 1) % args.ckpt_every == 0:
                save_checkpoint(
                    os.path.join(args.save_dir, f"ckpt_step_{step + 1}.pt"),
                    model,
                    opt,
                    step + 1,
                )
                save_checkpoint(
                    os.path.join(args.save_dir, "latest.pt"), model, opt, step + 1
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
                        f"Between {a[i]} and {c[i]}, which number comes in between?",
                    ]
                q = phr[_r.randrange(len(phr))]
                print(f"Q: {q}\n   gold={y[i]} pred={pred[i]}")
        return
    # default to pair training when no subcommand is provided
    train(args if args.cmd == "train" else p.parse_args(["train"]))


if __name__ == "__main__":
    main()
