from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import json
import random
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn

from .data.episode_gen import EpisodeConfig, EpisodeGenerator
from .model.backbone import TinyBackbone
from .model.vq_layer import EmaVectorQuantizer
from .model.domain import DomainAdapter, DomainToken
from .model.relation_head import DistMultHead, AnalogyProjector
from .teacher.llm_teacher import LLMTeacher
from utils.checkpoints import CheckpointManager
from utils.ema import EMA
import shutil


@dataclass
class TrainConfig:
    device: str = "cpu"
    d_model: int = 128
    code_dim: int = 64
    num_codes: int = 32
    num_relations: int = 8  # includes numeric relations (parity, successor, mod3, predecessor, add_2, tens, ones, make10)
    max_len: int = 8
    num_domains: int = 1
    use_domain_token: bool = True
    use_domain_adapter: bool = True

    # Loss knobs (kept for compatibility with README/plan; not all are used here)
    loss_contrastive: float = 0.2
    loss_rel: float = 1.0
    loss_analogy: float = 1.0
    loss_task: float = 0.2
    loss_instance: float = 0.2
    loss_stability: float = 0.0
    loss_vq: float = 1.0
    loss_vq_entropy: float = 0.0
    loss_vq_usage: float = 0.0
    loss_mdl: float = 0.0


class ConceptLearner(nn.Module):
    """
    Minimal learner that maps sequences to a shared concept space with a VQ
    bottleneck, plus relation and analogy heads. This class intentionally
    keeps training-agnostic utilities only, so it can be imported by eval and
    the playground without pulling in a full training loop.
    """

    def __init__(self, cfg: TrainConfig, ecfg: Optional[EpisodeConfig] = None):
        super().__init__()
        self.cfg = cfg
        self.device_str = cfg.device
        self.num_codes = cfg.num_codes
        # Token space sizing: digits 1..max_base, PAD=max_base+1 -> vocab=max_base+2
        self._ecfg = ecfg or EpisodeConfig(device=cfg.device)
        vocab_size = (self._ecfg.max_base + 2)

        self.backbone = TinyBackbone(vocab_size=vocab_size, d_model=cfg.d_model, max_len=cfg.max_len)
        self.domain_token = DomainToken(cfg.num_domains, cfg.d_model) if cfg.use_domain_token else None
        self.domain_adapter = DomainAdapter(cfg.num_domains, cfg.d_model) if cfg.use_domain_adapter else None

        # Project into code space and quantize
        self.pre_vq_norm = nn.LayerNorm(cfg.d_model)
        self.enc = nn.Linear(cfg.d_model, cfg.code_dim)
        self.vq_global = EmaVectorQuantizer(num_codes=cfg.num_codes, code_dim=cfg.code_dim, commitment_cost=0.25)

        # Small projection heads used in the training loop (kept for BC)
        self.proj1 = nn.Linear(cfg.code_dim, cfg.code_dim)
        self.proj2 = nn.Linear(cfg.code_dim, cfg.code_dim)

        # Relation and analogy heads
        self.rel = DistMultHead(concept_dim=cfg.code_dim, num_relations=cfg.num_relations)
        self.analogy = AnalogyProjector(dim=cfg.code_dim, proj_dim=min(32, cfg.code_dim))
        # Auxiliary relation type classifier over (s,o)
        self.rel_cls = nn.Sequential(
            nn.Linear(4 * cfg.code_dim, 2 * cfg.code_dim),
            nn.ReLU(),
            nn.Linear(2 * cfg.code_dim, cfg.num_relations),
        )
        # Pairwise scorer MLP for (s, r, o) -> {neg,pos} logits
        self.rel_embed = nn.Embedding(cfg.num_relations, cfg.code_dim)
        self.pair_scorer = nn.Sequential(
            nn.Linear(5 * cfg.code_dim, 2 * cfg.code_dim),
            nn.ReLU(),
            nn.Linear(2 * cfg.code_dim, 2),
        )

        # Lightweight same/different classifier (used by run_step in older loop)
        self.same_head = nn.Sequential(
            nn.Linear(2 * cfg.code_dim, cfg.code_dim),
            nn.ReLU(),
            nn.Linear(cfg.code_dim, 2),
        )

        # Optional: instance permanence head placeholder
        self.use_instance_head = True
        self.instance_head = nn.Sequential(
            nn.Linear(2 * cfg.code_dim, cfg.code_dim),
            nn.ReLU(),
            nn.Linear(cfg.code_dim, 2),
        )

    def encode(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None, domain: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of token sequences into quantized concept codes.
        Returns a dict with: h (backbone out), z (pre-quant), z_q (quantized),
        indices (code indices), vq_loss (commitment loss).
        """
        h = self.backbone(tokens, mask)
        if self.domain_token is not None:
            h = self.domain_token(h, domain)
        if self.domain_adapter is not None:
            h = self.domain_adapter(h, domain)
        z = self.enc(self.pre_vq_norm(h))
        z_q, indices, vq_loss = self.vq_global(z)
        return {"h": h, "z": z, "z_q": z_q, "indices": indices, "vq_loss": vq_loss}


@torch.no_grad()
def _probe_analogy(model: ConceptLearner, gen: EpisodeGenerator, device: str, batches: int = 2, batch: int = 128) -> float:
    """
    Quick in-batch analogy probe. Returns accuracy in [0,1].
    """
    model.eval()
    correct, total = 0, 0
    for _ in range(batches):
        analog = gen.sample_analogies(batch)
        A = model.encode(analog["A_desc"].to(device), analog["A_mask"].to(device))["z_q"]
        B = model.encode(analog["B_desc"].to(device), analog["B_mask"].to(device))["z_q"]
        C = model.encode(analog["C_desc"].to(device), analog["C_mask"].to(device))["z_q"]
        D = model.encode(analog["D_desc"].to(device), analog["D_mask"].to(device))["z_q"]
        r_ab = model.analogy.rel_vec(A, B)
        r_cd_all = model.analogy.rel_vec(C.unsqueeze(1), D.unsqueeze(0))
        sim = torch.einsum(
            "bp,bnp->bn",
            torch.nn.functional.normalize(r_ab, dim=-1),
            torch.nn.functional.normalize(r_cd_all, dim=-1),
        )
        pred = sim.argmax(dim=-1)
        labels = torch.arange(batch, device=device)
        correct += (pred == labels).sum().item()
        total += batch
    acc = correct / max(1, total)
    model.train()
    return acc


def _resolve_device(arg_device: str | None) -> str:
    dev = (arg_device or "auto").lower()
    if dev in ("auto", "", "gpu"):
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return dev


def _rel_name_to_id() -> Dict[str, int]:
    """
    Mapping aligned with EpisodeGenerator's extended relation set:
      0: same_parity (unused for LLM triples), 1: successor_of, 2: same_mod3 (unused),
      3: predecessor_of, 4: add_2, 5: has_tens, 6: has_ones, 7: makes_ten_with
    """
    return {
        "same_parity": 0,
        "successor_of": 1,
        "same_mod3": 2,
        "predecessor_of": 3,
        "add_2": 4,
        "has_tens": 5,
        "has_ones": 6,
        "makes_ten_with": 7,
    }


def _build_triple_batch_from_llm(payload: Dict[str, any], gen: EpisodeGenerator, device: str, batch: int) -> Optional[Dict[str, torch.Tensor]]:
    triples: List[List[str]] = payload.get("triples", [])
    rel_map = _rel_name_to_id()
    # Filter usable triples: numeric string subjects/objects + known relations
    usable: List[Tuple[int, int, int]] = []
    for t in triples:
        if not (isinstance(t, list) and len(t) == 3):
            continue
        s, r, o = t
        if r not in rel_map:
            continue
        try:
            si = int(str(s))
            oi = int(str(o))
        except Exception:
            continue
        # clip to generator range
        si = max(0, min(gen.n_items - 1, si))
        oi = max(0, min(gen.n_items - 1, oi))
        usable.append((si, rel_map[r], oi))
    if not usable:
        return None
    import random

    random.shuffle(usable)
    usable = usable[:batch]
    s_idx = torch.tensor([u[0] for u in usable], device=device)
    r = torch.tensor([u[1] for u in usable], device=device)
    o_idx = torch.tensor([u[2] for u in usable], device=device)
    # Build hard negatives using the generator's corruption routine
    o_neg_list: List[int] = []
    for si, ri, oi in zip(s_idx.tolist(), r.tolist(), o_idx.tolist()):
        o_neg_list.append(gen._corrupt_o_hard(int(si), int(ri), int(oi)))
    o_neg_idx = torch.tensor(o_neg_list, device=device, dtype=torch.long)

    s_desc, s_mask, _ = gen._render_batch(s_idx)
    o_desc, o_mask, _ = gen._render_batch(o_idx)
    o_neg_desc, o_neg_mask, _ = gen._render_batch(o_neg_idx)
    return {
        "s_idx": s_idx,
        "r": r,
        "o_idx": o_idx,
        "o_neg_idx": o_neg_idx,
        "s_desc": s_desc,
        "s_mask": s_mask,
        "o_desc": o_desc,
        "o_mask": o_mask,
        "o_neg_desc": o_neg_desc,
        "o_neg_mask": o_neg_mask,
    }


def train_main():  # pragma: no cover
    """
    Minimal smoke-test training loop. This is intentionally tiny and CPU-friendly,
    and serves mainly as a skeleton you can extend with the full curriculum.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3, help="base learning rate")
    parser.add_argument(
        "--sched",
        choices=["none", "cosine"],
        default="cosine",
        help="LR scheduler (with linear warmup)",
    )
    parser.add_argument("--warmup", type=int, default=200, help="linear warmup steps before scheduler")
    parser.add_argument("--lr_min", type=float, default=3e-5, help="minimum lr for cosine schedule")
    parser.add_argument("--vq_weight", type=float, default=0.05, help="initial weight for each VQ commitment loss term")
    parser.add_argument(
        "--vq_final_weight",
        type=float,
        default=0.2,
        help="final weight for each VQ term by the end of training (linear anneal). Defaults to --vq_weight",
    )
    parser.add_argument("--log_every", type=int, default=20, help="logging period in steps")
    parser.add_argument("--val_every", type=int, default=0, help="if >0, run a small validation probe every N steps")
    parser.add_argument("--usage_weight", type=float, default=0.0, help="weight for normalized code usage entropy (encourage diversity if >0)")
    parser.add_argument("--ce_mode", choices=["pair", "inbatch"], default="pair", help="CE mode: pairwise (pos vs hard-neg) or full in-batch")
    parser.add_argument("--use_mlp_scorer", action="store_true", help="use MLP pairwise scorer instead of DistMult when ce_mode=pair")
    # Checkpointing/resume
    parser.add_argument("--ckpt_dir", default="checkpoints", help="directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=0, help="save checkpoint every N steps (0 disables periodic save)")
    parser.add_argument("--resume_latest", action="store_true", help="resume from latest checkpoint in --ckpt_dir if available")
    parser.add_argument("--resume_path", default="", help="resume from a specific checkpoint path")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay for tracking eval weights")
    parser.add_argument("--ce_temp", type=float, default=0.2, help="temperature for in-batch CE over triples")
    parser.add_argument("--eval_analogy_batches", type=int, default=0, help="if >0, probe analogy acc (with EMA) on save using this many batches")
    parser.add_argument("--analogy_weight", type=float, default=0.1, help="final weight for analogy consistency loss after ramp")
    parser.add_argument("--analogy_temp", type=float, default=0.1, help="temperature for analogy InfoNCE")
    parser.add_argument("--analogy_warmup", type=int, default=1000, help="steps before enabling analogy loss")
    parser.add_argument("--analogy_ramp", type=int, default=3000, help="ramp steps to reach --analogy_weight from 0 after warmup")
    parser.add_argument("--vq_warmup", type=int, default=2000, help="steps to bypass VQ (use continuous z) for stability")
    parser.add_argument("--vq_blend", type=int, default=2000, help="after warmup, linearly blend z->z_q over this many steps")
    # Optional distillation from continuous (teacher) to quantized (student) pairwise logits
    parser.add_argument("--distill_weight", type=float, default=0.5, help="weight for KD loss from z (teacher) to z_q (student) during/after VQ blend (pairwise only)")
    parser.add_argument("--distill_start", type=int, default=0, help="relative step after which to enable KD (default: start immediately when blending)")
    # Ranking and relation classifier auxiliaries
    parser.add_argument("--rank_weight", type=float, default=0.5, help="weight for margin ranking loss against hard negatives")
    parser.add_argument("--rank_margin", type=float, default=0.2, help="margin for ranking loss")
    parser.add_argument("--relcls_weight", type=float, default=0.2, help="weight for auxiliary relation-type classifier")
    # Optional LLM teacher scaffolding
    parser.add_argument("--use_llm_teacher", action="store_true", help="use OpenAI-powered LLMTeacher for numbers triples")
    parser.add_argument("--llm_model", default="gpt-4o-mini", help="OpenAI model name for LLMTeacher")
    parser.add_argument("--llm_offline", action="store_true", help="force offline mode for LLM teacher (no API calls)")
    parser.add_argument("--llm_data_json", default="", help="optional path to pre-generated LLM episodes JSON (from concept_learner.generate_data)")
    args = parser.parse_args()
    device = _resolve_device(args.device)
    print(f"[train] device={device}")

    ecfg = EpisodeConfig(device=device)
    tcfg = TrainConfig(device=device)
    gen = EpisodeGenerator(ecfg)
    model = ConceptLearner(tcfg, ecfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Scheduler: linear warmup -> cosine decay (optional)
    if args.sched == "cosine":
        total_steps = max(args.steps, 1)
        warmup = max(args.warmup, 0)
        lr_max = args.lr
        lr_min = args.lr_min

        def lr_lambda(step: int):
            if step < warmup:
                return max(1e-8, (step + 1) / max(1, warmup))
            # Cosine from warmup..total_steps
            t = min(step - warmup, total_steps - warmup)
            T = max(1, total_steps - warmup)
            import math

            cos = 0.5 * (1 + math.cos(math.pi * t / T))
            scale = (lr_min / lr_max) + (1 - lr_min / lr_max) * cos
            return max(scale, lr_min / lr_max)

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        scheduler = None
    ema = EMA(model, decay=args.ema_decay)
    ckpt = CheckpointManager(args.ckpt_dir)

    # Optionally resume
    start_step = 0
    if args.resume_path:
        state = torch.load(args.resume_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state["model"], strict=False)  # type: ignore[index]
        if missing or unexpected:
            print(f"[train] Warning: non-strict load. missing={len(missing)} unexpected={len(unexpected)}")
        if state.get("optim") is not None:
            opt.load_state_dict(state["optim"])  # type: ignore[index]
        if state.get("ema") is not None:
            ema.load_state_dict(state["ema"])  # type: ignore[index]
        start_step = int(state.get("step", 0))
        print(f"[train] Resumed from {args.resume_path} at step {start_step}")
    elif args.resume_latest:
        try:
            start_step, _ = ckpt.load_latest(model, opt, ema)
            print(f"[train] Resumed from latest checkpoint in {args.ckpt_dir} at step {start_step}")
        except AssertionError:
            print(f"[train] No checkpoints found in {args.ckpt_dir}; starting fresh")
    teacher = LLMTeacher(None, model=args.llm_model) if args.use_llm_teacher else None
    if teacher is not None and args.llm_offline:
        # Force offline fallback regardless of environment
        teacher.client = None
    pregen_payloads: list[dict] = []
    if args.use_llm_teacher and args.llm_data_json:
        try:
            with open(args.llm_data_json, "r") as f:
                data = json.load(f)
            eps = data.get("episodes", data)
            for e in eps:
                if isinstance(e, dict) and e.get("domain") == "numbers" and e.get("type") in ("kg", "triples"):
                    payload = e.get("payload")
                    if isinstance(payload, dict):
                        pregen_payloads.append(payload)
            if pregen_payloads:
                print(f"[train] Loaded {len(pregen_payloads)} pre-generated LLM payloads from {args.llm_data_json}")
        except Exception as ex:
            print(f"[train] Failed to load pre-generated LLM data from {args.llm_data_json}: {ex}")
    # If using LLM teacher and an API client is available, start a background producer
    if teacher is not None and teacher.client is not None:
        # Produce episodes with at least 2x batch triples, keep a small buffer
        teacher.start_numbers_producer(min_triples=max(2 * args.batch, 256), buffer_size=2, max_calls=4)
    model.train()
    best_rel_acc = float("-inf")
    best_an_acc = float("-inf")
    ce_ema = None
    acc_ema = None
    loss_ema = None
    usage_ent_ema = None
    for i in range(1, args.steps + 1):
        step = start_step + i
        if teacher is not None:
            triple_batch = None
            # 1) Prefer pre-generated payload if available
            if pregen_payloads:
                payload = random.choice(pregen_payloads)
                triple_batch = _build_triple_batch_from_llm(payload, gen, device, args.batch)
            # 2) Otherwise consume from background queue if an online client exists
            if triple_batch is None and teacher.client is not None:
                ep = teacher.pop_numbers_episode(timeout=0.01)
                if ep is not None:
                    triple_batch = _build_triple_batch_from_llm(ep.get("payload", {}), gen, device, args.batch)
            # If no LLM episode available yet (or offline), fall back to synthetic triples immediately
            batch = triple_batch if triple_batch is not None else gen.sample_triples(args.batch)
        else:
            batch = gen.sample_triples(args.batch)
        enc_s = model.encode(batch["s_desc"].to(device), batch["s_mask"].to(device))
        enc_o = model.encode(batch["o_desc"].to(device), batch["o_mask"].to(device))
        enc_o_neg = model.encode(batch["o_neg_desc"].to(device), batch["o_neg_mask"].to(device))
        # VQ warmup and smooth blend: use z early, then linearly blend to z_q
        rel_step = max(0, step - start_step)
        if rel_step < args.vq_warmup:
            m = 1.0
        elif rel_step < args.vq_warmup + max(1, args.vq_blend):
            m = float(args.vq_warmup + args.vq_blend - rel_step) / float(max(1, args.vq_blend))
        else:
            m = 0.0
        # m in [0,1]: 1 means use z; 0 means use z_q
        s = (1 - m) * enc_s["z_q"] + m * enc_s["z"]
        o = (1 - m) * enc_o["z_q"] + m * enc_o["z"]
        o_neg = (1 - m) * enc_o_neg["z_q"] + m * enc_o_neg["z"]
        # VQ commitment loss is faded in with the same factor (1-m)
        vq_s = (1 - m) * enc_s["vq_loss"]
        vq_o = (1 - m) * enc_o["vq_loss"]
        r = batch["r"].to(device)

        # Triple scoring
        w = model.rel.rel[r]
        v = s * w
        if args.ce_mode == "inbatch":
            # In-batch + hard negatives (2B classes)
            o_all = torch.cat([o, o_neg], dim=0)
            logits = (model.rel.scale * torch.matmul(v, o_all.t())) / max(1e-6, args.ce_temp)
            labels = torch.arange(logits.size(0), device=device)
            ce_loss = nn.CrossEntropyLoss()(logits, labels)
        else:
            # Pairwise pos vs its hard-negative (2 classes per row)
            if args.use_mlp_scorer:
                r_emb = model.rel_embed(r)
                feats_pos = torch.cat([s, o, s * o, torch.abs(s - o), r_emb], dim=-1)
                feats_neg = torch.cat([s, o_neg, s * o_neg, torch.abs(s - o_neg), r_emb], dim=-1)
                log_pos = model.pair_scorer(feats_pos)  # (B,2)
                log_neg = model.pair_scorer(feats_neg)  # (B,2)
                labels_pos = torch.ones(log_pos.size(0), dtype=torch.long, device=device)
                labels_neg = torch.zeros(log_neg.size(0), dtype=torch.long, device=device)
                ce_loss = 0.5 * (nn.CrossEntropyLoss()(log_pos, labels_pos) + nn.CrossEntropyLoss()(log_neg, labels_neg))
                # For logging
                acc_pair = 0.5 * (
                    (log_pos.argmax(dim=-1) == labels_pos).float().mean().item()
                    + (log_neg.argmax(dim=-1) == labels_neg).float().mean().item()
                )
                classes_count = 2
                rows_count = log_pos.size(0)
                # Optional KD: teacher from continuous z vs student from blended z/z_q
                kd_loss = torch.tensor(0.0, device=device)
                if rel_step >= max(args.vq_warmup, args.distill_start) and args.distill_weight > 0:
                    # Recompute teacher with pure continuous z
                    s_t, o_t, o_n_t = enc_s["z"].detach(), enc_o["z"].detach(), enc_o_neg["z"].detach()
                    r_emb_t = model.rel_embed(r)
                    fpos_t = torch.cat([s_t, o_t, s_t * o_t, torch.abs(s_t - o_t), r_emb_t], dim=-1)
                    fneg_t = torch.cat([s_t, o_n_t, s_t * o_n_t, torch.abs(s_t - o_n_t), r_emb_t], dim=-1)
                    log_pos_t = model.pair_scorer(fpos_t)
                    log_neg_t = model.pair_scorer(fneg_t)
                    # Build pair logits [neg, pos]
                    logits_t = torch.stack([log_neg_t[:, 0], log_pos_t[:, 1]], dim=-1)
                    logits_s = torch.stack([log_neg[:, 0], log_pos[:, 1]], dim=-1)
                    # KL divergence: teacher probs vs student logits
                    with torch.no_grad():
                        p_t = torch.softmax(logits_t, dim=-1)
                    log_p_s = torch.log_softmax(logits_s, dim=-1)
                    kd_loss = torch.nn.functional.kl_div(log_p_s, p_t, reduction="batchmean")
            else:
                pos = torch.sum(v * o, dim=-1) / max(1e-6, args.ce_temp)
                neg = torch.sum(v * o_neg, dim=-1) / max(1e-6, args.ce_temp)
                logits = torch.stack([neg, pos], dim=-1)  # label=1 means positive
                labels = torch.ones(logits.size(0), dtype=torch.long, device=device)
                ce_loss = nn.CrossEntropyLoss()(logits, labels)
                # For logging
                acc_pair = (logits.argmax(dim=-1) == labels).float().mean().item()
                classes_count = logits.size(1)
                rows_count = logits.size(0)
                kd_loss = torch.tensor(0.0, device=device)
        # Margin ranking auxiliary: push pos above its paired hard-neg by margin
        pos = torch.sum(v * o, dim=-1)
        neg = torch.sum(v * o_neg, dim=-1)
        rank_loss = torch.relu(args.rank_margin - (pos - neg)).mean()

        # Analogy consistency loss (separate batch for stability), after warmup
        an_loss = torch.tensor(0.0, device=device)
        # Analogy ramp: 0 until warmup, then linear ramp to --analogy_weight over --analogy_ramp steps
        aw = 0.0
        if rel_step >= args.analogy_warmup and args.analogy_weight > 0:
            if args.analogy_ramp <= 0:
                aw = float(args.analogy_weight)
            else:
                aw = float(args.analogy_weight) * min(1.0, (rel_step - args.analogy_warmup) / float(args.analogy_ramp))
            analog = gen.sample_analogies(args.batch)
            enc_A = model.encode(analog["A_desc"].to(device), analog["A_mask"].to(device))
            enc_B = model.encode(analog["B_desc"].to(device), analog["B_mask"].to(device))
            enc_C = model.encode(analog["C_desc"].to(device), analog["C_mask"].to(device))
            enc_D = model.encode(analog["D_desc"].to(device), analog["D_mask"].to(device))
            an_loss = model.analogy.analogy_loss(
                enc_A["z_q"], enc_B["z_q"], enc_C["z_q"], enc_D["z_q"], temp=args.analogy_temp
            )
        # Auxiliary relation-type classifier over (s,o)
        feats = torch.cat([s, o, s * o, torch.abs(s - o)], dim=-1)
        rel_logits = model.rel_cls(feats)
        rel_ce = nn.CrossEntropyLoss()(rel_logits, r)
        # Balance VQ commitment vs CE; default smaller so CE is not dominated
        if args.vq_final_weight is None:
            vq_w = args.vq_weight
        else:
            # Linear anneal from initial -> final over training horizon
            prog = min(max(i / max(1, args.steps), 0.0), 1.0)
            vq_w = (1 - prog) * args.vq_weight + prog * args.vq_final_weight
        # Optional code usage diversity regularizer (maximize entropy)
        from .losses import code_usage_entropy
        usage_ent = code_usage_entropy(torch.cat([enc_s["indices"], enc_o["indices"]], dim=0), model.num_codes)
        usage_norm = usage_ent / max(1.0, torch.log(torch.tensor(float(model.num_codes), device=device)))
        loss = (
            ce_loss
            + args.distill_weight * kd_loss
            + args.rank_weight * rank_loss
            + args.relcls_weight * rel_ce
            + vq_w * vq_s
            + vq_w * vq_o
            + aw * an_loss
            - args.usage_weight * usage_norm
        )

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        if scheduler is not None:
            scheduler.step()

        if args.val_every and (step % args.val_every == 0):
            # quick validation probe on a fresh batch (same distribution)
            with torch.no_grad():
                model.eval()
                vb = gen.sample_triples(args.batch)
                e_vs = model.encode(vb["s_desc"].to(device), vb["s_mask"].to(device))
                e_vo = model.encode(vb["o_desc"].to(device), vb["o_mask"].to(device))
                e_vo_neg = model.encode(vb["o_neg_desc"].to(device), vb["o_neg_mask"].to(device))
                if step - start_step < args.vq_warmup:
                    vs, vo, vo_neg = e_vs["z"], e_vo["z"], e_vo_neg["z"]
                else:
                    vs, vo, vo_neg = e_vs["z_q"], e_vo["z_q"], e_vo_neg["z_q"]
                vr = vb["r"].to(device)
                vw = model.rel.rel[vr]
                vv = vs * vw
                if args.ce_mode == "inbatch":
                    vo_all = torch.cat([vo, vo_neg], dim=0)
                    v_logits = (model.rel.scale * torch.matmul(vv, vo_all.t())) / max(1e-6, args.ce_temp)
                    v_labels = torch.arange(v_logits.size(0), device=device)
                    v_ce = nn.CrossEntropyLoss()(v_logits, v_labels)
                    v_acc = (v_logits.argmax(dim=-1) == v_labels).float().mean().item()
                else:
                    if args.use_mlp_scorer:
                        r_emb = model.rel_embed(vr)
                        feats_pos = torch.cat([vs, vo, vs * vo, torch.abs(vs - vo), r_emb], dim=-1)
                        feats_neg = torch.cat([vs, vo_neg, vs * vo_neg, torch.abs(vs - vo_neg), r_emb], dim=-1)
                        log_pos = model.pair_scorer(feats_pos)
                        log_neg = model.pair_scorer(feats_neg)
                        labels_pos = torch.ones(log_pos.size(0), dtype=torch.long, device=device)
                        labels_neg = torch.zeros(log_neg.size(0), dtype=torch.long, device=device)
                        v_ce = 0.5 * (nn.CrossEntropyLoss()(log_pos, labels_pos) + nn.CrossEntropyLoss()(log_neg, labels_neg))
                        v_acc = 0.5 * (
                            (log_pos.argmax(dim=-1) == labels_pos).float().mean().item()
                            + (log_neg.argmax(dim=-1) == labels_neg).float().mean().item()
                        )
                    else:
                        v_pos = torch.sum(vv * vo, dim=-1) / max(1e-6, args.ce_temp)
                        v_neg = torch.sum(vv * vo_neg, dim=-1) / max(1e-6, args.ce_temp)
                        v_logits = torch.stack([v_neg, v_pos], dim=-1)
                        v_labels = torch.ones(v_logits.size(0), dtype=torch.long, device=device)
                        v_ce = nn.CrossEntropyLoss()(v_logits, v_labels)
                        v_acc = (v_logits.argmax(dim=-1) == v_labels).float().mean().item()
                model.train()
            print(f"val@{step} ce {float(v_ce):.4f} | acc {v_acc:.3f}")

        if step % max(1, args.log_every) == 0:
            with torch.no_grad():
                if args.ce_mode == "inbatch":
                    B = logits.size(0)
                    pred = logits.argmax(dim=-1)
                    acc = (pred == labels).float().mean().item()
                    import math
                    lnB = math.log(max(1, logits.size(1)))
                else:
                    B = rows_count if 'rows_count' in locals() else s.size(0)
                    acc = acc_pair if 'acc_pair' in locals() else 0.0
                    import math
                    lnB = math.log(2)
                # Code usage diagnostics
                from .losses import code_usage_entropy

                indices = torch.cat([enc_s["indices"], enc_o["indices"]], dim=0)
                usage_ent = float(code_usage_entropy(indices, model.num_codes))
                dead_codes = int((model.vq_global.ema_cluster_size < 1.0).sum().item())
                # EMAs for smoother trends
                alpha = 0.95
                ce_ema = float(ce_loss.item()) if ce_ema is None else alpha * ce_ema + (1 - alpha) * float(ce_loss.item())
                acc_ema = float(acc) if acc_ema is None else alpha * acc_ema + (1 - alpha) * float(acc)
                loss_ema = float(loss.item()) if loss_ema is None else alpha * loss_ema + (1 - alpha) * float(loss.item())
                usage_ent_ema = usage_ent if usage_ent_ema is None else alpha * usage_ent_ema + (1 - alpha) * usage_ent
                print(
                    f"step {step:05d} "
                    f"total {loss.item():.4f} (ema {loss_ema:.4f}) | "
                    f"ce {ce_loss.item():.4f} (ema {ce_ema:.4f}) (lnB~{lnB:.2f}) | "
                    f"vq_s {float(vq_s):.4f} vq_o {float(vq_o):.4f} | "
                    f"acc {acc:.3f} (ema {acc_ema:.3f}) B {B} | "
                    f"usage_ent {usage_ent:.3f} (ema {usage_ent_ema:.3f}) dead {dead_codes}"
                )

        # Periodic checkpointing
        if args.save_every and (step % args.save_every == 0):
            with torch.no_grad():
                if args.ce_mode == "inbatch":
                    B = logits.size(0)
                    pred = logits.argmax(dim=-1)
                    acc_cur = float((pred == labels).float().mean().item())
                else:
                    acc_cur = float(acc if 'acc' in locals() else acc_pair)
            # Optional quick analogy probe (EMA weights)
            analogy_acc = None
            if args.eval_analogy_batches and args.eval_analogy_batches > 0:
                ema.apply_to(model)
                analogy_acc = _probe_analogy(model, gen, device, batches=args.eval_analogy_batches, batch=args.batch)
                ema.restore(model)
            extra = {"acc": acc_cur, "loss": float(loss.item()), "ce": float(ce_loss.item())}
            if analogy_acc is not None:
                extra["analogy_acc"] = float(analogy_acc)
            path = ckpt.save(step, model, opt, ema, extra)
            ckpt.update_latest(path)
            if acc_cur > best_rel_acc:
                best_rel_acc = acc_cur
                shutil.copy2(path, os.path.join(args.ckpt_dir, "best_rel.pt"))
            if analogy_acc is not None and analogy_acc > best_an_acc:
                best_an_acc = float(analogy_acc)
                shutil.copy2(path, os.path.join(args.ckpt_dir, "best_an.pt"))

    # Final checkpoint (always save at the end)
    with torch.no_grad():
        if args.ce_mode == "inbatch":
            B = logits.size(0)
            pred = logits.argmax(dim=-1)
            acc_final = float((pred == labels).float().mean().item())
        else:
            acc_final = float(acc if 'acc' in locals() else acc_pair)
    analogy_acc = None
    if args.eval_analogy_batches and args.eval_analogy_batches > 0:
        ema.apply_to(model)
        analogy_acc = _probe_analogy(model, gen, device, batches=args.eval_analogy_batches, batch=args.batch)
        ema.restore(model)
    extra = {"acc": acc_final, "loss": float(loss.item()), "ce": float(ce_loss.item())}
    if analogy_acc is not None:
        extra["analogy_acc"] = float(analogy_acc)
    final_path = ckpt.save(step, model, opt, ema, extra)
    latest_path = ckpt.update_latest(final_path)
    try:
        if os.path.exists(final_path):
            if acc_final > best_rel_acc:
                best_rel_acc = acc_final
                shutil.copy2(final_path, os.path.join(args.ckpt_dir, "best_rel.pt"))
            if analogy_acc is not None and analogy_acc > best_an_acc:
                best_an_acc = float(analogy_acc)
                shutil.copy2(final_path, os.path.join(args.ckpt_dir, "best_an.pt"))
    except Exception as ex:
        print(f"[train] Warning: could not update best checkpoints: {ex}")
    print(f"[train] Saved final checkpoint to {final_path} (latest -> {latest_path})")

    print("[train] done.")


if __name__ == "__main__":  # pragma: no cover
    train_main()
