from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .data.episode_gen import EpisodeConfig, EpisodeGenerator
from .train import ConceptLearner, TrainConfig
from utils.checkpoints import CheckpointManager


@dataclass
class BaselineConfig:
    device: str = "cpu"
    steps: int = 5000
    batch: int = 128
    lr: float = 1e-3
    d_model: int = 128
    code_dim: int = 64


def train_main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    # Checkpointing
    parser.add_argument("--ckpt_dir", default="checkpoints_baseline")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--resume_latest", action="store_true")
    parser.add_argument("--resume_path", default="")
    args = parser.parse_args()

    dev = (args.device or "auto").lower()
    if dev in ("auto", "gpu"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = dev
    print(f"[baseline] device={device}")

    ecfg = EpisodeConfig(device=device)
    tcfg = TrainConfig(device=device)
    model = ConceptLearner(tcfg, ecfg).to(device)
    gen = EpisodeGenerator(ecfg)

    # Simple MLP pairwise scorer (reuse model's scorer and relation emb)
    opt = torch.optim.AdamW(
        list(model.backbone.parameters())
        + list(model.enc.parameters())
        + list(model.rel_embed.parameters())
        + list(model.pair_scorer.parameters()),
        lr=args.lr,
    )

    ckpt = CheckpointManager(args.ckpt_dir)
    start_step = 0
    if args.resume_path:
        state = torch.load(args.resume_path, map_location="cpu")
        # non-strict to tolerate incidental head changes
        try:
            model.load_state_dict(state["model"], strict=False)  # type: ignore[index]
        except Exception:
            model.load_state_dict(state, strict=False)  # type: ignore[arg-type]
        if state.get("optim") is not None:
            opt.load_state_dict(state["optim"])  # type: ignore[index]
        start_step = int(state.get("step", 0))
        print(f"[baseline] Resumed from {args.resume_path} at step {start_step}")
    elif args.resume_latest:
        try:
            # Reuse manager's latest semantics
            latest = torch.load(ckpt.update_latest, map_location="cpu")  # type: ignore[misc]
        except Exception:
            try:
                step, _ = ckpt.load_latest(model, opt, None)
                start_step = int(step)
                print(f"[baseline] Resumed latest at step {start_step}")
            except AssertionError:
                pass

    model.train()
    for i in range(1, args.steps + 1):
        step = start_step + i
        batch = gen.sample_triples(args.batch)
        s_desc, s_mask = batch["s_desc"].to(device), batch["s_mask"].to(device)
        o_desc, o_mask = batch["o_desc"].to(device), batch["o_mask"].to(device)
        on_desc, on_mask = batch["o_neg_desc"].to(device), batch["o_neg_mask"].to(device)
        r = batch["r"].to(device)

        # Encode continuous features (no VQ)
        h_s = model.backbone(s_desc, s_mask)
        h_o = model.backbone(o_desc, o_mask)
        h_on = model.backbone(on_desc, on_mask)
        z_s = model.enc(model.pre_vq_norm(h_s))
        z_o = model.enc(model.pre_vq_norm(h_o))
        z_on = model.enc(model.pre_vq_norm(h_on))

        # Pairwise MLP scorer
        r_emb = model.rel_embed(r)
        feats_pos = torch.cat([z_s, z_o, z_s * z_o, torch.abs(z_s - z_o), r_emb], dim=-1)
        feats_neg = torch.cat([z_s, z_on, z_s * z_on, torch.abs(z_s - z_on), r_emb], dim=-1)
        log_pos = model.pair_scorer(feats_pos)
        log_neg = model.pair_scorer(feats_neg)
        labels_pos = torch.ones(log_pos.size(0), dtype=torch.long, device=device)
        labels_neg = torch.zeros(log_neg.size(0), dtype=torch.long, device=device)
        ce = 0.5 * (nn.CrossEntropyLoss()(log_pos, labels_pos) + nn.CrossEntropyLoss()(log_neg, labels_neg))

        opt.zero_grad()
        ce.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            with torch.no_grad():
                acc = 0.5 * (
                    (log_pos.argmax(dim=-1) == labels_pos).float().mean().item()
                    + (log_neg.argmax(dim=-1) == labels_neg).float().mean().item()
                )
                print(f"step {step:05d} | ce {float(ce):.4f} (lnB~{math.log(2):.2f}) | acc {acc:.3f} B {args.batch}")

        if args.save_every and (step % args.save_every == 0):
            with torch.no_grad():
                acc = 0.5 * (
                    (log_pos.argmax(dim=-1) == labels_pos).float().mean().item()
                    + (log_neg.argmax(dim=-1) == labels_neg).float().mean().item()
                )
            extra = {"acc": float(acc), "ce": float(ce.item())}
            path = ckpt.save(step, model, opt, None, extra)
            latest = ckpt.update_latest(path)
            print(f"[baseline] Saved checkpoint to {path} (latest -> {latest})")

    # Final checkpoint
    with torch.no_grad():
        acc = 0.5 * (
            (log_pos.argmax(dim=-1) == labels_pos).float().mean().item()
            + (log_neg.argmax(dim=-1) == labels_neg).float().mean().item()
        )
    extra = {"acc": float(acc), "ce": float(ce.item())}
    final_path = ckpt.save(step, model, opt, None, extra)
    latest_path = ckpt.update_latest(final_path)
    print(f"[baseline] Saved final checkpoint to {final_path} (latest -> {latest_path})")

    print("[baseline] done.")


if __name__ == "__main__":  # pragma: no cover
    train_main()
