import argparse
import os
import torch
from typing import List

from concept_learner.episodes import EpisodeConfig, EpisodeGenerator
from concept_learner.model import CLModel


# Token ids reserved in this simple vocab
# 0=PAD, 1=CLS, 2=SEP, 3..12 = digits 0..9
TOK_SUCCESSOR = 13
TOK_PREDECESSOR = 14
TOK_NEXT = 15
TOK_PREVIOUS = 16
TOK_OF = 17
TOK_BETWEEN = 18
TOK_AND = 19

VOCAB_SIZE = 20  # update if you add tokens above


def digits_to_tokens(n: int) -> List[int]:
    s = str(n)
    return [int(ch) + 3 for ch in s]


def pack_count_examples(kind, a, c, variant_words=True, device="cpu", max_len=18):
    """
    Build sequences like:
      [CLS] <SUCCESSOR|NEXT> <OF> A [SEP]
      [CLS] <PREDECESSOR|PREVIOUS> <OF> A [SEP]
      [CLS] <BETWEEN> A <AND> C [SEP]
    """
    B = a.size(0)
    ids = torch.full((B, max_len), 0, dtype=torch.long, device=device)
    mask = torch.zeros(B, max_len, dtype=torch.long, device=device)

    for i in range(B):
        seq = [1]  # CLS
        k = int(kind[i].item())
        ai = int(a[i].item())
        if k == 0:  # successor/next
            tok_rel = TOK_SUCCESSOR if (not variant_words or torch.rand(1).item() < 0.5) else TOK_NEXT
            seq += [tok_rel, TOK_OF]
            seq += digits_to_tokens(ai)
        elif k == 1:  # predecessor/previous
            tok_rel = TOK_PREDECESSOR if (not variant_words or torch.rand(1).item() < 0.5) else TOK_PREVIOUS
            seq += [tok_rel, TOK_OF]
            seq += digits_to_tokens(ai)
        else:  # between
            seq += [TOK_BETWEEN]
            seq += digits_to_tokens(ai)
            seq += [TOK_AND]
            ci = int(c[i].item())
            seq += digits_to_tokens(ci)
        seq += [2]  # SEP

        L = min(len(seq), max_len)
        ids[i, :L] = torch.tensor(seq[:L], dtype=torch.long, device=device)
        mask[i, :L] = 1

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
def quick_eval(model, gen, device, eval_batches=50, batch_size=128, max_len=18):
    model.eval()
    total, correct = 0, 0
    for _ in range(eval_batches):
        data = gen.sample_counting(batch=batch_size)
        ids, mask = pack_count_examples(data["kind"], data["a"], data["c"], device=device, max_len=max_len)
        y = data["target"].to(device)
        _, logits_seq, _, _, _, _ = model(ids, mask)
        pred = logits_seq.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(1, total)


def main():
    p = argparse.ArgumentParser(description="Train/Eval counting tasks (successor/pred/between)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_number", type=int, default=100)
    p.add_argument("--save_dir", type=str, default="runs/counting")
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--max_len", type=int, default=18)
    p.add_argument("--eval_batches", type=int, default=50)

    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ecfg = EpisodeConfig(max_number=args.max_number, device=device, max_len=6, min_base=10, max_base=10)
    gen = EpisodeGenerator(ecfg)

    # vocab: PAD, CLS, SEP, digits(0..9), relation words
    model = CLModel(vocab_size=VOCAB_SIZE, d_model=args.d_model, num_classes=ecfg.max_number, pad_id=0, cls_id=1, max_len=args.max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start = 0
    if args.resume and os.path.isfile(args.resume):
        start = load_checkpoint(args.resume, model, opt, map_location=device)
        print(f"Resumed from {args.resume} at step {start}")

    best = -1.0
    for step in range(start, args.steps):
        data = gen.sample_counting(batch=args.batch_size)
        ids, mask = pack_count_examples(data["kind"], data["a"], data["c"], device=device, max_len=args.max_len)
        y = data["target"].to(device)
        _, logits_seq, vq_loss, _, stop_logits, _ = model(ids, mask)
        loss = torch.nn.functional.cross_entropy(logits_seq, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % args.log_every == 0:
            val_acc = quick_eval(model, gen, device, eval_batches=args.eval_batches, batch_size=min(128, args.batch_size), max_len=args.max_len)
            print(f"step {step+1}/{args.steps} loss={loss.item():.4f} val_acc={val_acc:.3f}")
            if val_acc > best and args.save_dir:
                best = val_acc
                os.makedirs(args.save_dir, exist_ok=True)
                save_checkpoint(os.path.join(args.save_dir, "best.pt"), model, opt, step + 1)
                print(f"  Saved best checkpoint: {os.path.join(args.save_dir, 'best.pt')} (val_acc={val_acc:.3f})")

        if args.save_dir and (step + 1) % args.ckpt_every == 0:
            save_checkpoint(os.path.join(args.save_dir, f"ckpt_step_{step+1}.pt"), model, opt, step + 1)
            save_checkpoint(os.path.join(args.save_dir, "latest.pt"), model, opt, step + 1)

    # final eval and a few samples
    acc = quick_eval(model, gen, device, eval_batches=100, batch_size=min(256, args.batch_size), max_len=args.max_len)
    print(f"Final eval acc: {acc:.3f}")

    @torch.no_grad()
    def show_samples(n=6):
        data = gen.sample_counting(batch=n)
        ids, mask = pack_count_examples(data["kind"], data["a"], data["c"], device=device, max_len=args.max_len)
        y = data["target"].tolist()
        _, logits_seq, _, _, _, _ = model(ids, mask)
        pred = logits_seq.argmax(dim=-1).tolist()
        kind = data["kind"].tolist()
        a = data["a"].tolist()
        c = data["c"].tolist()
        for i in range(n):
            if kind[i] == 0:
                q = f"What is the successor of {a[i]}?"
            elif kind[i] == 1:
                q = f"What is the predecessor of {a[i]}?"
            else:
                q = f"What number comes between {a[i]} and {c[i]}?"
            print(f"Q: {q}\n   gold={y[i]} pred={pred[i]}")

    show_samples(6)


if __name__ == "__main__":
    main()

