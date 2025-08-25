import argparse
import os
import torch
from concept_learner.episodes import EpisodeConfig, EpisodeGenerator
from concept_learner.model import CLModel


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


def save_checkpoint(path, model, opt, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path)


def load_checkpoint(path, model, opt, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    return int(ckpt.get("step", 0))


@torch.no_grad()
def _quick_eval(model: CLModel, gen: EpisodeGenerator, device: str, batch_size: int, eval_batches: int, same_base: bool) -> float:
    model.eval()
    total, correct = 0, 0
    for _ in range(eval_batches):
        batch = gen.sample_posneg_pairs(batch=batch_size)
        a_desc, a_mask, b_desc, b_mask = _maybe_same_base(gen, batch, device, same_base)
        ids, mask = _pack_pairs(a_desc, a_mask, b_desc, b_mask, device)
        y = batch["label"].to(device)
        _, logits_seq, _, _, _, _ = model(ids, mask)
        pred = logits_seq.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(1, total)


def _maybe_same_base(gen, batch, device, same_base: bool):
    if not same_base:
        return batch["a_desc"], batch["a_mask"], batch["b_desc"], batch["b_mask"]
    # re-render B with A's base so A and B share the same base per pair
    a_desc, a_mask, a_base = batch["a_desc"], batch["a_mask"], batch["a_base"]
    b_desc, b_mask, b_base = batch["b_desc"], batch["b_mask"], batch["b_base"]
    # also re-render A to ensure consistency
    a_desc2, a_mask2, _ = gen._render_batch_with_fixed_base(batch["a_idx"], a_base)
    b_desc2, b_mask2, _ = gen._render_batch_with_fixed_base(batch["b_idx"], a_base)
    return a_desc2, a_mask2, b_desc2, b_mask2


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

    vocab_size = ecfg.max_base + 3  # 0 PAD, 1 CLS, 2 SEP, digits 3..(base+2)
    model = CLModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_classes=2,  # pos/neg pairs
        pad_id=0,
        cls_id=1,
        max_len=(1 + ecfg.max_len + 1 + ecfg.max_len),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        start_step = load_checkpoint(args.resume, model, opt, map_location=device)
        print(f"Resumed from {args.resume} at step {start_step}")

    best_acc = -1.0
    model.train()
    for step in range(start_step, args.steps):
        if args.overfit_batch is not None and step == start_step:
            cached = gen.sample_posneg_pairs(batch=args.overfit_batch)
        batch = cached if args.overfit_batch is not None else gen.sample_posneg_pairs(batch=args.batch_size)
        a_desc, a_mask, b_desc, b_mask = _maybe_same_base(gen, batch, device, args.same_base)
        ids, mask = _pack_pairs(a_desc, a_mask, b_desc, b_mask, device)
        y = batch["label"].to(device)
        logits_tok, logits_seq, vq_loss, indices, stop_logits, _ = model(ids, mask)

        loss_seq = torch.nn.functional.cross_entropy(logits_seq, y)
        stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            stop_logits, torch.zeros_like(stop_logits)
        )
        loss = loss_seq + args.lambda_vq * vq_loss + args.lambda_stop * stop_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % args.log_every == 0:
            with torch.no_grad():
                probs = torch.softmax(logits_seq, dim=-1)[:, 1].mean().item()
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

            val_acc = _quick_eval(
                model, gen, device, batch_size=min(args.batch_size, 128), eval_batches=10, same_base=args.same_base
            )
            extra = f" vq_util={vq_util:.3f}" if vq_util is not None else ""
            print(
                f"step {step+1}/{args.steps} loss={loss.item():.4f} seq={loss_seq.item():.4f} "
                f"vq={vq_loss.item():.4f} stop={stop_loss.item():.4f} p(pos)~{probs:.3f} val_acc={val_acc:.3f}{extra}"
            )
            if args.save_dir and val_acc > best_acc:
                best_acc = val_acc
                best_path = os.path.join(args.save_dir, "best.pt")
                save_checkpoint(best_path, model, opt, step + 1)
                print(f"  Saved best checkpoint: {best_path} (val_acc={val_acc:.3f})")

        if args.save_dir and (step + 1) % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"ckpt_step_{step+1}.pt")
            save_checkpoint(ckpt_path, model, opt, step + 1)
            latest = os.path.join(args.save_dir, "latest.pt")
            save_checkpoint(latest, model, opt, step + 1)


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
    vocab_size = ecfg.max_base + 3
    model = CLModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_classes=2,
        pad_id=0,
        cls_id=1,
        max_len=(1 + ecfg.max_len + 1 + ecfg.max_len),
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
        a_desc, a_mask, b_desc, b_mask = _maybe_same_base(gen, batch, device, args.same_base)
        ids, mask = _pack_pairs(a_desc, a_mask, b_desc, b_mask, device)
        y = batch["label"].to(device)
        rel = batch.get("rel").to(device)
        a_idx = batch["a_idx"].to(device)
        b_idx = batch["b_idx"].to(device)
        _, logits_seq, _, _, _, _ = model(ids, mask)
        pred = logits_seq.argmax(dim=-1)
        probs = torch.softmax(logits_seq, dim=-1)[:, 1]
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
    a_desc, a_mask, b_desc, b_mask = _maybe_same_base(gen, batch, device, args.same_base)
    ids, mask = _pack_pairs(a_desc, a_mask, b_desc, b_mask, device)
    y = batch["label"].tolist()
    a_idx = batch["a_idx"].tolist()
    b_idx = batch["b_idx"].tolist()
    rel = batch.get("rel")
    rel = rel.tolist() if rel is not None else [None] * len(y)
    _, logits_seq, _, _, _, _ = model(ids, mask)
    probs = torch.softmax(logits_seq, dim=-1)[:, 1].tolist()

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

    args = p.parse_args()
    if args.cmd == "eval":
        evaluate(args)
    else:
        # default to train when no subcommand is provided for convenience
        train(args if args.cmd == "train" else p.parse_args(["train"]))


if __name__ == "__main__":
    main()
