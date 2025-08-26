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


def save_checkpoint(path, model, opt, step, best_acc: float | None = None, ema_state: dict | None = None):
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
        if any(s in msg for s in ["head.weight", "decoder.token_head.weight", "decoder.seq_head.weight"]):
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
                            if isinstance(buf, torch.Tensor) and buf.shape != tok_param.data.shape:
                                new_buf = torch.zeros_like(tok_param.data)
                                rows = min(buf.shape[0], new_buf.shape[0])
                                new_buf[:rows].copy_(buf[:rows])
                                st[k] = new_buf
                # also repair classifier head states if shapes changed
                for p in [model.head.weight, model.head.bias, model.decoder.token_head.weight, model.decoder.token_head.bias, model.decoder.seq_head.weight, model.decoder.seq_head.bias]:
                    stp = opt.state.get(p, None)
                    if isinstance(stp, dict):
                        for k in ("exp_avg", "exp_avg_sq"):
                            if k in stp and isinstance(stp[k], torch.Tensor) and stp[k].shape != p.data.shape:
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
def _quick_eval(model: CLModel, gen: EpisodeGenerator, tok: HFTokenizerWrapper, device: str, batch_size: int, eval_batches: int, max_len: int, num_numbers: int) -> float:
    model.eval()
    total, correct = 0, 0
    for _ in range(eval_batches):
        batch = gen.sample_posneg_pairs(batch=batch_size)
        ids, mask = _pack_pair_questions_text(batch, tok, max_len)
        y = batch["label"].to(device)
        _, logits_seq, _, _, _, _ = model(ids, mask)
        NO_IDX = num_numbers + 1
        YES_IDX = num_numbers
        pred_bin = logits_seq[:, [NO_IDX, YES_IDX]].argmax(dim=-1)
        correct += (pred_bin == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(1, total)




def _pack_text_batch(texts: List[str], tok: HFTokenizerWrapper, max_len: int, device: str):
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


def _pack_pair_questions_text(batch, tok: HFTokenizerWrapper, max_len: int):
    import random
    a = batch["a_idx"].tolist()
    b = batch["b_idx"].tolist()
    r = batch["rel"].tolist()
    texts = []
    for ai, bi, ri in zip(a, b, r):
        tpls = _pair_templates(ai, bi, ri)
        texts.append(random.choice(tpls))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _pack_text_batch(texts, tok, max_len, device)


def _pack_count_examples_text(kind, a, c, tok: HFTokenizerWrapper, max_len: int):
    import random
    kind_l = kind.tolist(); a_l = a.tolist(); c_l = c.tolist()
    texts = []
    for k, ai, ci in zip(kind_l, a_l, c_l):
        if k == 0:
            texts.append(random.choice([f"What comes after {ai}?", f"What is the successor of {ai}?", f"Next number after {ai}?"]))
        elif k == 1:
            texts.append(random.choice([f"What comes before {ai}?", f"What is the predecessor of {ai}?", f"Previous number before {ai}?",
            ]))
        else:
            texts.append(random.choice([f"What number comes between {ai} and {ci}?", f"Between {ai} and {ci}, which number comes in between?"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _pack_text_batch(texts, tok, max_len, device)


def _pack_equality_examples_text(a: torch.Tensor, kind: torch.Tensor, tok: HFTokenizerWrapper, max_len: int):
    import random
    a_l = a.tolist(); k_l = kind.tolist()
    texts = []
    for ai, ki in zip(a_l, k_l):
        if ki == 0:
            rhs = ai + 1
            texts.append(random.choice([f"Is the successor of {ai} equal to {rhs}?", f"Is next of {ai} = {ai} + 1?", f"Does {ai} + 1 equal the successor of {ai}?",
            ]))
        else:
            rhs = ai - 1
            texts.append(random.choice([f"Is the predecessor of {ai} equal to {rhs}?", f"Is previous of {ai} = {ai} - 1?", f"Does {ai} - 1 equal the predecessor of {ai}?",
            ]))
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
        return {"decay": self.decay, "shadow": {k: v.clone().cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict):
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow", {})
        self.shadow = {k: v.clone() for k, v in shadow.items()}

    def reset(self, model: torch.nn.Module):
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()


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
    vocab_size = tok.vocab_size
    # unified head: numbers (0..N-1) + {YES, NO}
    num_numbers = ecfg.max_number
    YES_IDX = num_numbers
    NO_IDX = num_numbers + 1
    model = CLModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_classes=num_numbers + 2,
        pad_id=0,
        cls_id=1,
        max_len=args.max_len,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        start_step = load_checkpoint(args.resume, model, opt, map_location=device, ema=None)
        print(f"Resumed from {args.resume} at step {start_step}")
        # initialize best_acc from checkpoint if present; else from a quick eval
        ckpt_best = getattr(load_checkpoint, "last_best_acc", None)
        if ckpt_best is not None:
            best_acc = float(ckpt_best)
            print(f"Loaded best_acc from checkpoint: {best_acc:.3f}")
        else:
            best_acc = _quick_eval(
                model,
                gen,
                tok,
                device,
                batch_size=min(args.batch_size, 128),
                eval_batches=10,
                max_len=args.max_len,
                num_numbers=num_numbers,
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
            opt, mode="max", factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.min_lr, threshold=1e-4
        )

    best_acc = -1.0 if not (args.resume and os.path.isfile(args.resume)) else best_acc
    model.train()
    seq_len = args.max_len
    for step in range(start_step, args.steps):
        if args.sched == "cosine":
            set_cosine_lr(step)
        if args.overfit_batch is not None and step == start_step:
            cached = gen.sample_posneg_pairs(batch=args.overfit_batch)
        batch = cached if args.overfit_batch is not None else gen.sample_posneg_pairs(batch=args.batch_size)
        # natural-language pair questions
        ids_pairs, mask_pairs = _pack_pair_questions_text(batch, tok, max_len=seq_len)
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
        data_cnt = gen.sample_counting(batch=cnt_bsz)
        ids_cnt, mask_cnt = _pack_count_examples_text(
            data_cnt["kind"], data_cnt["a"], data_cnt["c"], tok, max_len=seq_len
        )
        y_cnt = data_cnt["target"].to(device)  # 0..N-1

        ids = torch.cat([ids_pairs, ids_eq, ids_cnt], dim=0)
        mask = torch.cat([mask_pairs, mask_eq, mask_cnt], dim=0)
        y = torch.cat([y_pairs, y_eq, y_cnt], dim=0)
        logits_tok, logits_seq, vq_loss, indices, stop_logits, _ = model(ids, mask)

        loss_seq = torch.nn.functional.cross_entropy(logits_seq, y, label_smoothing=args.label_smoothing)
        stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            stop_logits, torch.zeros_like(stop_logits)
        )
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
                used = []
                for idx_t in indices:
                    used.append(len(torch.unique(idx_t)))
                used_avg = sum(used) / max(1, len(used))
                vq_util = used_avg / max(1, getattr(model.rvq, "codebook_size", used_avg))
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
            extra = f" vq_util={vq_util:.3f}" if vq_util is not None else ""
            print(
                f"step {step+1}/{args.steps} loss={loss.item():.4f} seq={loss_seq.item():.4f} "
                f"vq={vq_loss.item():.4f} stop={stop_loss.item():.4f} p(yes)~{probs:.3f} val_pair={val_acc_pair:.3f} val_cnt={val_acc_cnt:.3f} avg={val_acc:.3f} lr={cur_lr:.2e}{extra}"
            )
            if args.save_dir and val_acc > best_acc:
                best_acc = val_acc
                best_path = os.path.join(args.save_dir, "best.pt")
                ema_state = ema.state_dict() if ema is not None else None
                save_checkpoint(best_path, model, opt, step + 1, best_acc=val_acc, ema_state=ema_state)
                print(f"  Saved best checkpoint: {best_path} (val_acc={val_acc:.3f})")

        if args.save_dir and (step + 1) % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"ckpt_step_{step+1}.pt")
            ema_state = ema.state_dict() if ema is not None else None
            save_checkpoint(ckpt_path, model, opt, step + 1, ema_state=ema_state)
            latest = os.path.join(args.save_dir, "latest.pt")
            save_checkpoint(latest, model, opt, step + 1, best_acc=best_acc if best_acc >= 0 else None, ema_state=ema_state)


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
    model = CLModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_classes=num_numbers + 2,
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
            sel = (rel == r)
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
            print(f"  {name:12s}: {hit}/{tot} = {hit/max(1,tot):.3f}")
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
    print(f"Best fixed-threshold acc in [0.35,0.65]: {best_acc:.3f} at thr={best_thr:.2f}")

    # Error slices near boundary for greater/smaller (|a-b| <= 1)
    a_all = _t.cat(all_a)
    b_all = _t.cat(all_b)
    rel_all = _t.cat(all_rel)
    pred_all = (probs_all >= 0.5).long()
    eq_diff = (a_all - b_all).abs()
    for r, name in [(7, "greater"), (8, "smaller")]:
        sel = (rel_all == r)
        if sel.any():
            close = sel & (eq_diff <= 1)
            far = sel & (eq_diff > 1)
            if close.any():
                acc_close = (pred_all[close] == labels_all[close]).float().mean().item()
                print(f"{name} close (|a-b|<=1): {int(close.sum())} ex, acc={acc_close:.3f}")
            if far.any():
                acc_far = (pred_all[far] == labels_all[far]).float().mean().item()
                print(f"{name} far   (|a-b|>1):  {int(far.sum())} ex, acc={acc_far:.3f}")

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
        print(f"  Q: {q}\n     gold={gold} pred={pred} p(yes)={p:.3f} rel={rel_name(rel[i])}")

    # Also show a few equality reformulations using different language
    print("\nSample equality QA (varied phrasing):")
    import random as _r
    with torch.no_grad():
        a_eq = torch.randint(0, gen.n_items, (4,), device=device)
        kind_eq = torch.randint(0, 2, (4,), device=device)
        ids_eq, mask_eq, y_eq = _pack_equality_examples_text(a_eq, kind_eq, tok, max_len=args.max_len)
        # use two-class slice {NO, YES}
        num_numbers = ecfg.max_number
        NO_IDX = num_numbers + 1
        YES_IDX = num_numbers
        _, logits_seq_eq, _, _, _, _ = model(ids_eq, mask_eq)
        probs_eq = torch.softmax(logits_seq_eq[:, [NO_IDX, YES_IDX]], dim=-1)[:, 1].tolist()
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
                    f"Is the successor of {ai} = {(ai+1)%1000}?",
                    f"Is next of {ai} equal to {(ai+1)%1000}?",
                ]
            else:
                phr = [
                    f"Is the predecessor of {ai} equal to {ai} - 1?",
                    f"Is previous of {ai} = {ai} - 1?",
                    f"Does {ai} - 1 equal the predecessor of {ai}?",
                    f"Is the predecessor of {ai} = {(ai-1)%1000}?",
                    f"Is previous of {ai} equal to {(ai-1)%1000}?",
                ]
            qtext = phr[_r.randrange(len(phr))]
            gold = "yes" if y_eq[i] == 1 else "no"
            pred = "yes" if probs_eq[i] >= 0.5 else "no"
            print(f"  Q: {qtext}\n     gold={gold} pred={pred} p(yes)={probs_eq[i]:.3f}")

    # Counting diagnostics combined in eval
    print("\nCounting evaluation:")
    tot, hit = 0, 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            dc = gen.sample_counting(batch=min(args.batch_size, 128))
            ic, mc = _pack_count_examples_text(dc["kind"], dc["a"], dc["c"], tok, max_len=args.max_len)
            yc = dc["target"].to(device)
            _, lsc, _, _, _, _ = model(ic, mc)
            predc = lsc[:, :ecfg.max_number].argmax(dim=-1)
            hit += (predc == yc).sum().item()
            tot += yc.numel()
    print(f"Counting eval accuracy: {hit/max(1,tot):.3f}")
    # sample counting QA with variety
    import random as _r
    with torch.no_grad():
        dshow = gen.sample_counting(batch=4)
        ic, mc = _pack_count_examples_text(dshow["kind"], dshow["a"], dshow["c"], tok, max_len=args.max_len)
        yshow = dshow["target"].tolist()
        _, lss, _, _, _, _ = model(ic, mc)
        predshow = lss[:, :ecfg.max_number].argmax(dim=-1).tolist()
        kind = dshow["kind"].tolist(); a = dshow["a"].tolist(); c = dshow["c"].tolist()
        for i in range(len(kind)):
            if kind[i] == 0:
                phr = [f"What is the successor of {a[i]}?", f"What comes after {a[i]}?", f"What number comes after {a[i]}?"]
            elif kind[i] == 1:
                phr = [f"What is the predecessor of {a[i]}?", f"What comes before {a[i]}?", f"What number comes before {a[i]}?"]
            else:
                phr = [f"What number comes between {a[i]} and {c[i]}?", f"Between {a[i]} and {c[i]}, which number comes in between?"]
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
        sp.add_argument("--max_len", type=int, default=6)
        sp.add_argument("--min_base", type=int, default=5)
        sp.add_argument("--max_base", type=int, default=10)
        sp.add_argument("--batch_size", type=int, default=64)

    pt = sub.add_parser("train", help="Train and save checkpoints")
    add_shared(pt)
    pt.add_argument("--steps", type=int, default=1000)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--weight_decay", type=float, default=0.0)
    pt.add_argument("--label_smoothing", type=float, default=0.0)
    pt.add_argument("--sched", type=str, default="none", choices=["none", "plateau", "cosine"], help="LR scheduler to use")
    pt.add_argument("--min_lr", type=float, default=1e-6)
    pt.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for cosine schedule")
    pt.add_argument("--plateau_factor", type=float, default=0.5)
    pt.add_argument("--plateau_patience", type=int, default=800)
    pt.add_argument("--ema_decay", type=float, default=0.0, help="EMA decay (0 to disable)")
    pt.add_argument("--lambda_vq", type=float, default=0.1)
    pt.add_argument("--lambda_stop", type=float, default=0.1)
    pt.add_argument("--save_dir", type=str, default="runs/episodes")
    pt.add_argument("--ckpt_every", type=int, default=200)
    pt.add_argument("--log_every", type=int, default=50)
    pt.add_argument("--resume", type=str, default=None)
    pt.add_argument("--same_base", action="store_true", help="Render both items with the same base per pair (easier task)")
    pt.add_argument("--overfit_batch", type=int, default=None, help="Overfit a single cached batch of this size for sanity")
    pt.add_argument("--base10", action="store_true", help="Force base-10 rendering only (min_base=max_base=10)")

    pe = sub.add_parser("eval", help="Evaluate from a checkpoint")
    add_shared(pe)
    pe.add_argument("--checkpoint", type=str, required=True)
    pe.add_argument("--eval_batches", type=int, default=50)
    pe.add_argument("--same_base", action="store_true")
    pe.add_argument("--base10", action="store_true", help="Force base-10 rendering only (min_base=max_base=10)")

    # Counting subcommands (unified CLI)
    ct = sub.add_parser("count-train", help="Train counting tasks (successor/pred/between)")
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

    ce = sub.add_parser("count-eval", help="Evaluate a counting checkpoint")
    ce.add_argument("--device", type=str, default=None)
    ce.add_argument("--d_model", type=int, default=128)
    ce.add_argument("--max_number", type=int, default=100)
    ce.add_argument("--checkpoint", type=str, required=True)
    ce.add_argument("--max_len", type=int, default=18)
    ce.add_argument("--eval_batches", type=int, default=100)

    args = p.parse_args()
    if args.cmd == "eval":
        evaluate(args)
        return
    if args.cmd == "count-train":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        ecfg = EpisodeConfig(max_number=args.max_number, device=device, max_len=6, min_base=10, max_base=10)
        gen = EpisodeGenerator(ecfg)
        tok = HFTokenizerWrapper("bert-base-cased")
        model = CLModel(vocab_size=tok.vocab_size, d_model=args.d_model, num_classes=ecfg.max_number, pad_id=0, cls_id=1, max_len=args.max_len).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        start = 0
        if args.resume and os.path.isfile(args.resume):
            start = load_checkpoint(args.resume, model, opt, map_location=device)
            print(f"Resumed from {args.resume} at step {start}")
        best = -1.0
        for step in range(start, args.steps):
            data = gen.sample_counting(batch=args.batch_size)
            ids, mask = _pack_count_examples_text(data["kind"], data["a"], data["c"], tok, max_len=args.max_len)
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
                        ids2, mask2 = _pack_count_examples_text(data2["kind"], data2["a"], data2["c"], tok, max_len=args.max_len)
                        y2 = data2["target"].to(device)
                        _, logits_seq2, _, _, _, _ = model(ids2, mask2)
                        pred2 = logits_seq2.argmax(dim=-1)
                        hit += (pred2 == y2).sum().item()
                        tot += y2.numel()
                model.train()
                val_acc = hit / max(1, tot)
                print(f"step {step+1}/{args.steps} loss={loss.item():.4f} val_acc={val_acc:.3f}")
                if val_acc > best and args.save_dir:
                    best = val_acc
                    os.makedirs(args.save_dir, exist_ok=True)
                    save_checkpoint(os.path.join(args.save_dir, "best.pt"), model, opt, step + 1)
                    print(f"  Saved best checkpoint: {os.path.join(args.save_dir, 'best.pt')} (val_acc={val_acc:.3f})")
            if args.save_dir and (step + 1) % args.ckpt_every == 0:
                save_checkpoint(os.path.join(args.save_dir, f"ckpt_step_{step+1}.pt"), model, opt, step + 1)
                save_checkpoint(os.path.join(args.save_dir, "latest.pt"), model, opt, step + 1)
        return
    if args.cmd == "count-eval":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        ecfg = EpisodeConfig(max_number=args.max_number, device=device, max_len=6, min_base=10, max_base=10)
        gen = EpisodeGenerator(ecfg)
        tok = HFTokenizerWrapper("bert-base-cased")
        model = CLModel(vocab_size=tok.vocab_size, d_model=args.d_model, num_classes=ecfg.max_number, pad_id=0, cls_id=1, max_len=args.max_len).to(device)
        load_checkpoint(args.checkpoint, model, opt=None, map_location=device)
        model.eval()
        # eval
        tot, hit = 0, 0
        with torch.no_grad():
            for _ in range(args.eval_batches):
                data2 = gen.sample_counting(batch=min(256, 128))
                ids2, mask2 = _pack_count_examples_text(data2["kind"], data2["a"], data2["c"], tok, max_len=args.max_len)
                y2 = data2["target"].to(device)
                _, logits_seq2, _, _, _, _ = model(ids2, mask2)
                pred2 = logits_seq2.argmax(dim=-1)
                hit += (pred2 == y2).sum().item()
                tot += y2.numel()
        print(f"Counting eval accuracy: {hit/max(1,tot):.3f}")
        # sample questions
        import random as _r
        with torch.no_grad():
            data = gen.sample_counting(batch=6)
            ids, mask = _pack_count_examples_text(data["kind"], data["a"], data["c"], tok, max_len=args.max_len)
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
