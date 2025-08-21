import argparse
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from concept_learner.data.episode_gen import EpisodeConfig, EpisodeGenerator
from concept_learner.model.backbone import TinyBackbone
from concept_learner.model.vq_layer import EmaVectorQuantizer
from concept_learner.model.relation_head import DistMultHead, AnalogyProjector
from concept_learner.losses import EWC, info_nce, code_usage_entropy
from utils.checkpoints import CheckpointManager
from utils.ema import EMA


@dataclass
class TrainConfig:
    device: str = "cpu"
    vocab_size: int = 16  # max base <= 10; leave room
    d_model: int = 128
    num_layers: int = 2
    nhead: int = 4
    max_len: int = 6
    num_codes: int = 32
    code_dim: int = 32
    relations: int = 3
    batch: int = 128
    lr: float = 1e-3
    steps: int = 10000
    sleep_every: int = 50
    sleep_steps: int = 20
    temperature: float = 1.0
    temp_min: float = 0.5
    temp_max: float = 1.5
    temp_anneal_steps: int = 5000
    loss_contrastive: float = 0.3
    loss_rel: float = 1.0
    loss_analogy: float = 1.0
    loss_mdl: float = 0.1
    loss_stability: float = 0.1
    loss_task: float = 0.2
    loss_vq_entropy: float = 0.05
    loss_vq: float = 0.5
    loss_vq_usage: float = 0.2
    loss_codes: float = 0.2
    usage_temp: float = 2.0


class ConceptLearner(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.backbone = TinyBackbone(cfg.vocab_size, cfg.d_model, cfg.nhead, cfg.num_layers, cfg.max_len)
        self.to_code = nn.Linear(cfg.d_model, cfg.code_dim)
        self.vq = EmaVectorQuantizer(cfg.num_codes, cfg.code_dim, decay=0.995)
        self.rel = DistMultHead(cfg.code_dim, cfg.relations)
        self.analogy = AnalogyProjector(cfg.code_dim, proj_dim=min(32, cfg.code_dim))
        self.same_head = nn.Sequential(
            nn.Linear(cfg.code_dim * 2, cfg.code_dim), nn.ReLU(), nn.Linear(cfg.code_dim, 2)
        )
        self.num_codes = cfg.num_codes
        # Projection heads for contrastive stability
        self.proj1 = nn.Sequential(
            nn.Linear(cfg.code_dim, cfg.code_dim), nn.ReLU(), nn.LayerNorm(cfg.code_dim)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(cfg.code_dim, cfg.code_dim), nn.ReLU(), nn.LayerNorm(cfg.code_dim)
        )

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        h = self.backbone(tokens, mask)
        z = self.to_code(h)
        z_q, indices, vq_loss = self.vq(z)
        return {"h": h, "z": z, "z_q": z_q, "indices": indices, "vq_loss": vq_loss}


def run_step(
    model: ConceptLearner,
    gen: EpisodeGenerator,
    cfg: TrainConfig,
    optimizer,
    ewc: EWC,
    step_idx: int,
    ema: EMA | None = None,
    adapt: Dict[str, bool] | None = None,
    sleep: bool = False,
):
    model.train()
    device = cfg.device
    losses = {}

    # Invariance & identification
    # Views curriculum: keep base fixed; schedule easy positives (identical remap)
    if step_idx < 3000:
        easy_prob = 1.0
    elif step_idx < 6000:
        easy_prob = 0.5
    elif step_idx < 8000:
        easy_prob = 0.2
    else:
        easy_prob = 0.0
    if adapt and adapt.get("recovery", False):
        easy_prob = 1.0
    views = gen.sample_views(cfg.batch, change_base_prob=0.0, easy_same_remap_prob=easy_prob)
    v1 = views["view1_desc"].to(device)
    v1m = views["view1_mask"].to(device)
    v2 = views["view2_desc"].to(device)
    v2m = views["view2_mask"].to(device)
    e1 = model.encode(v1, v1m)
    e2 = model.encode(v2, v2m)
    # Anneal temperature over steps
    # Early stabilize contrastive
    if step_idx < 2000:
        temperature = 1.0
    else:
        if cfg.temp_anneal_steps > 0:
            t = min(1.0, (step_idx - 2000) / float(max(1, cfg.temp_anneal_steps)))
            temperature = cfg.temp_max * (1 - t) + cfg.temp_min * t
        else:
            temperature = cfg.temperature
    if adapt and adapt.get("recovery", False):
        # Couple alignment tightly to codes during recovery
        temperature = 1.0
        z1 = model.proj1(e1["z_q"])  # quantized
        z2 = model.proj2(e2["z_q"])  # quantized
    else:
        # Use pre-quantized features for InfoNCE (smoother alignment)
        z1 = model.proj1(e1["z"])  # pre-quantized
        z2 = model.proj2(e2["z"])  # pre-quantized
    # symmetric stop-grad InfoNCE (alignment)
    loss_c12 = info_nce(z1.detach(), z2, temperature)
    loss_c21 = info_nce(z2.detach(), z1, temperature)
    losses["contrast_align"] = 0.5 * (loss_c12 + loss_c21)
    # Code-tying CE (match codes across views)
    logits_codes = torch.matmul(e1["z_q"], model.vq.codebook.weight.t())
    losses["code_ce"] = nn.CrossEntropyLoss()(torch.log_softmax(logits_codes, dim=-1), e2["indices"])  # type: ignore

    # Relational & analogical
    triples = gen.sample_triples(cfg.batch)
    s = model.encode(triples["s_desc"].to(device), triples["s_mask"].to(device))["z_q"]
    o = model.encode(triples["o_desc"].to(device), triples["o_mask"].to(device))["z_q"]
    r = triples["r"].to(device)
    # In-batch multiclass relation loss (many negatives)
    # v_i = s_i * W_{r_i}; logits = v @ o^T
    W_sel = model.rel.rel[r]  # (B, D)
    v = s * W_sel
    logits_rel = torch.matmul(v, o.t())
    labels_rel = torch.arange(logits_rel.size(0), device=device)
    loss_rel = nn.CrossEntropyLoss()(logits_rel, labels_rel)
    losses["rel"] = loss_rel

    # Restrict analogies to parity only for first 4000 steps
    allowed = [0] if step_idx < 4000 else None
    analog = gen.sample_analogies(cfg.batch, allowed_relations=allowed)
    A = model.encode(analog["A_desc"].to(device), analog["A_mask"].to(device))["z_q"]
    B = model.encode(analog["B_desc"].to(device), analog["B_mask"].to(device))["z_q"]
    C = model.encode(analog["C_desc"].to(device), analog["C_mask"].to(device))["z_q"]
    D = model.encode(analog["D_desc"].to(device), analog["D_mask"].to(device))["z_q"]
    # Analogy classification with scheduled temperature
    an_temp = 0.5 if step_idx < 3000 else 0.2
    loss_analogy = model.analogy.analogy_loss(A, B, C, D, temp=an_temp)
    # Add offset loss in concept space (strong early)
    r_off1 = A - B
    r_off2 = C - D
    loss_offset = F.mse_loss(r_off1, r_off2)
    off_w = 1.0 if step_idx < 3000 else 0.2
    losses["analogy"] = loss_analogy + off_w * loss_offset

    # Minimality (MDL)
    indices = torch.cat([e1["indices"], e2["indices"]])
    ent = code_usage_entropy(indices, model.num_codes)
    losses["mdl"] = ent
    # Encourage usage of codebook (higher entropy)
    losses["vq_ent"] = -ent
    # Differentiable usage loss via soft assignments
    with torch.no_grad():
        codebook = model.vq.codebook.weight.detach()  # (K, D)
    z_all = torch.cat([e1["z"], e2["z"]], dim=0)
    d = (
        torch.sum(z_all ** 2, dim=1, keepdim=True)
        + torch.sum(codebook ** 2, dim=1)
        - 2 * torch.matmul(z_all, codebook.t())
    )  # (B*2, K)
    q = torch.softmax(-d / cfg.usage_temp, dim=1)
    mean_q = q.mean(dim=0)
    usage_loss = torch.sum(mean_q * torch.log(mean_q + 1e-8)) * -1.0  # maximize entropy
    losses["vq_usage"] = usage_loss

    # Light task head: same/different concept pairs
    pairs = gen.sample_posneg_pairs(cfg.batch)
    a = F.normalize(model.encode(pairs["a_desc"].to(device), pairs["a_mask"].to(device))["z_q"], dim=-1)
    b = F.normalize(model.encode(pairs["b_desc"].to(device), pairs["b_mask"].to(device))["z_q"], dim=-1)
    y = pairs["label"].to(device)
    logits = model.same_head(torch.cat([a, b], dim=-1))
    loss_task = nn.CrossEntropyLoss()(logits, y)
    losses["task"] = loss_task

    # VQ commitment/codebook
    losses["vq"] = e1["vq_loss"] + e2["vq_loss"]

    # Stability (EWC-style) during sleep, smaller during day
    pen = ewc.penalty()
    losses["stability"] = pen

    # Mix
    # Scheduled weights
    w_c = 0.1 if step_idx < 4000 else cfg.loss_contrastive
    w_rel = 2.0 if step_idx < 4000 else cfg.loss_rel
    w_an = 1.5 if step_idx < 4000 else cfg.loss_analogy
    w_mdl = cfg.loss_mdl
    w_vq_usage = cfg.loss_vq_usage
    w_codes = cfg.loss_codes
    if adapt and adapt.get("recovery", False):
        w_c = 0.0
        w_rel = max(w_rel, 2.0)
        w_an = max(w_an, 1.5)
        w_mdl = 0.0
        w_vq_usage = max(w_vq_usage, 0.3)
        w_codes = max(w_codes, 0.2)
    total = (
        w_c * losses["contrast_align"]
        + w_codes * losses["code_ce"]
        + w_rel * losses["rel"]
        + w_an * losses["analogy"]
        + w_mdl * losses["mdl"]
        + cfg.loss_task * losses["task"]
        + cfg.loss_stability * losses["stability"]
        + cfg.loss_vq_entropy * losses["vq_ent"]
        + w_vq_usage * losses["vq_usage"]
        + cfg.loss_vq * losses["vq"]
    )

    optimizer.zero_grad()
    total.backward()
    clip_val = 0.2 if (adapt and adapt.get("recovery", False)) else 0.5
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
    optimizer.step()
    if ema is not None:
        ema.update(model)

    return {k: v.detach().item() for k, v in losses.items()}, total.detach().item()


def _resolve_device(arg_device: str | None) -> str:
    dev = (arg_device or "auto").lower()
    if dev in ("auto", "", "gpu"):
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return dev


def train_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="cuda|cpu|auto (default auto)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--resume_latest", action="store_true", help="resume from latest checkpoint in ckpt_dir")
    parser.add_argument("--resume_path", default="", help="resume from a specific checkpoint path")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    print(f"[train] Using device: {device} (cuda_available={torch.cuda.is_available()})")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    tcfg = TrainConfig(device=device, steps=args.steps, batch=args.batch)
    ecfg = EpisodeConfig(device=tcfg.device)
    gen = EpisodeGenerator(ecfg)

    model = ConceptLearner(tcfg).to(tcfg.device)
    # Optimizer with param groups
    enc_params = list(model.backbone.parameters()) + list(model.to_code.parameters())
    rel_params = list(model.rel.parameters()) + list(model.analogy.parameters()) + list(model.same_head.parameters())
    vq_params = list(model.vq.parameters()) + list(model.proj1.parameters()) + list(model.proj2.parameters())
    opt = optim.AdamW(
        [
            {"params": enc_params, "lr": 1e-3},
            {"params": rel_params, "lr": 2e-3},
            {"params": vq_params, "lr": 2e-3},
        ],
        weight_decay=1e-2,
    )
    ewc = EWC(model.parameters(), weight=0.0)

    ckpts = CheckpointManager(dir=args.ckpt_dir, keep=5)
    best_rel = float("inf")
    best_an = -float("inf")
    ema = EMA(model, decay=0.999)

    # Resume logic
    loaded_step = 0
    if args.resume_latest:
        try:
            loaded_step, extra = ckpts.load_latest(model, opt, ema)
            print(f"[train] Resumed from latest checkpoint at step {loaded_step}")
        except AssertionError:
            print("[train] No latest checkpoint found; starting fresh")
    elif args.resume_path:
        ckpt = torch.load(args.resume_path, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt))
        if ckpt.get("optim") is not None:
            opt.load_state_dict(ckpt["optim"])  # type: ignore
        if ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])  # type: ignore
        loaded_step = int(ckpt.get("step", 0))
        print(f"[train] Resumed from path {args.resume_path} at step {loaded_step}")

    def probe_analogy_acc(batch_size: int = 128, allowed_eval=None) -> float:
        # Use EMA weights for a stable probe
        ema.apply_to(model)
        try:
            analog_small = gen.sample_analogies(batch_size, allowed_relations=allowed_eval)
            Aev = model.encode(analog_small["A_desc"].to(tcfg.device), analog_small["A_mask"].to(tcfg.device))["z_q"]
            Bev = model.encode(analog_small["B_desc"].to(tcfg.device), analog_small["B_mask"].to(tcfg.device))["z_q"]
            Cev = model.encode(analog_small["C_desc"].to(tcfg.device), analog_small["C_mask"].to(tcfg.device))["z_q"]
            Dev = model.encode(analog_small["D_desc"].to(tcfg.device), analog_small["D_mask"].to(tcfg.device))["z_q"]
            r_ab = model.analogy.rel_vec(Aev, Bev)
            r_cd_all = model.analogy.rel_vec(Cev.unsqueeze(1), Dev.unsqueeze(0))
            sim = torch.einsum("bp,bnp->bn", F.normalize(r_ab, dim=-1), F.normalize(r_cd_all, dim=-1))
            pred = sim.argmax(dim=-1)
            labels = torch.arange(sim.size(0), device=tcfg.device)
            return (pred == labels).float().mean().item()
        finally:
            ema.restore(model)

    curriculum_boundaries = {3000, 6000, 8000}

    # If resuming, interpret --steps as additional steps to run from loaded_step
    start_step = loaded_step + 1
    end_step = loaded_step + tcfg.steps

    collapse_counter = 0
    recovery_until = 0
    for step in range(start_step, end_step + 1):
        # Increase stability penalty a bit during sleep steps
        if step % tcfg.sleep_every == 0:
            ewc.weight = 0.2
        else:
            ewc.weight = 0.0
        adapt = {"recovery": step <= recovery_until}
        losses, total = run_step(model, gen, tcfg, opt, ewc, step, ema, adapt)
        if step % 10 == 0 or step == 1:
            # Codebook stats (approximate, from a small batch)
            # For lightweight logging, recompute a small batch indices
            small = gen.sample_views(64)
            e_small = model.encode(small["view1_desc"].to(tcfg.device), small["view1_mask"].to(tcfg.device))
            with torch.no_grad():
                cb = model.vq.codebook.weight.detach()
                z_s = e_small["z"]
                d_s = (
                    torch.sum(z_s ** 2, dim=1, keepdim=True)
                    + torch.sum(cb ** 2, dim=1)
                    - 2 * torch.matmul(z_s, cb.t())
                )
                q_s = torch.softmax(-d_s / tcfg.usage_temp, dim=1)
                mean_q_s = q_s.mean(dim=0)
                perplexity = torch.exp(-(mean_q_s * torch.log(mean_q_s + 1e-8)).sum()).item()
                unique = int((mean_q_s > (1.0 / model.num_codes) * 0.01).sum().item())
            # Quick in-batch analogy accuracy probe (parity-only early)
            if step % 500 == 0:
                with torch.no_grad():
                    allowed_eval = [0] if step < 4000 else None
                    analog_small = gen.sample_analogies(64, allowed_relations=allowed_eval)
                    Aev = model.encode(analog_small["A_desc"].to(tcfg.device), analog_small["A_mask"].to(tcfg.device))["z_q"]
                    Bev = model.encode(analog_small["B_desc"].to(tcfg.device), analog_small["B_mask"].to(tcfg.device))["z_q"]
                    Cev = model.encode(analog_small["C_desc"].to(tcfg.device), analog_small["C_mask"].to(tcfg.device))["z_q"]
                    Dev = model.encode(analog_small["D_desc"].to(tcfg.device), analog_small["D_mask"].to(tcfg.device))["z_q"]
                    r_ab = model.analogy.rel_vec(Aev, Bev)
                    r_cd_all = model.analogy.rel_vec(Cev.unsqueeze(1), Dev.unsqueeze(0))
                    sim = torch.einsum("bp,bnp->bn", F.normalize(r_ab, dim=-1), F.normalize(r_cd_all, dim=-1))
                    pred = sim.argmax(dim=-1)
                    labels = torch.arange(sim.size(0), device=tcfg.device)
                    an_acc = (pred == labels).float().mean().item()
            else:
                an_acc = float('nan')
            c_disp = (losses.get('contrast_align', 0.0) + losses.get('code_ce', 0.0))
            print(
                f"step {step:04d} total={total:.3f} "
                f"c={c_disp:.3f} rel={losses['rel']:.3f} an={losses['analogy']:.3f} "
                f"mdl={losses['mdl']:.3f} task={losses['task']:.3f} vq={losses['vq']:.3f} st={losses['stability']:.3f} "
                f"ppx={perplexity:.2f} uniq={unique}/{model.num_codes} an_acc={an_acc:.3f} rec={(step<=recovery_until)}"
            )
            # Watchdog: auto-recover on codebook collapse
            if perplexity < 5.0:
                collapse_counter += 1
            else:
                collapse_counter = 0
            if collapse_counter >= 3:
                cand = [
                    os.path.join(args.ckpt_dir, "best_an.pt"),
                    os.path.join(args.ckpt_dir, "best_rel.pt"),
                    os.path.join(args.ckpt_dir, "latest.pt"),
                ]
                chosen = None
                for p in cand:
                    if os.path.exists(p):
                        chosen = p
                        break
                if chosen is not None:
                    try:
                        ckpt = torch.load(chosen, map_location="cpu")
                        model.load_state_dict(ckpt.get("model", ckpt))
                        if ckpt.get("optim") is not None:
                            opt.load_state_dict(ckpt["optim"])  # type: ignore
                        if ckpt.get("ema") is not None:
                            ema.load_state_dict(ckpt["ema"])  # type: ignore
                        recovery_until = step + 1000
                        collapse_counter = 0
                        print(f"[watchdog] Rolled back from {chosen}; entering recovery until step {recovery_until}")
                    except Exception as e:
                        print(f"[watchdog] Failed to rollback from {chosen}: {e}")
        # Checkpointing cadence: every N steps and at curriculum boundaries
        if (step % args.save_every == 0) or (step in curriculum_boundaries):
            # Periodic checkpoint via CheckpointManager + latest copy
            os.makedirs(args.ckpt_dir, exist_ok=True)
            allowed_eval = [0] if step < 4000 else None
            probe_acc = probe_analogy_acc(batch_size=128, allowed_eval=allowed_eval)
            extra_state = {"metrics": losses, "total": total, "an_acc": probe_acc}
            ckpts.save(step, model, opt, ema=ema, extra=extra_state)
            latest = os.path.join(args.ckpt_dir, "latest.pt")
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optim": opt.state_dict(),
                "ema": ema.state_dict(),
                "extra": extra_state,
            }, latest)
            # Track best by relation CE and in-batch analogy acc (if available)
            if losses["rel"] < best_rel:
                best_rel = losses["rel"]
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optim": opt.state_dict(),
                        "ema": ema.state_dict(),
                        "extra": extra_state,
                    },
                    os.path.join(args.ckpt_dir, "best_rel.pt"),
                )
            if not (probe_acc != probe_acc):  # not NaN
                if probe_acc > best_an:
                    best_an = probe_acc
                    torch.save(
                        {
                            "step": step,
                            "model": model.state_dict(),
                            "optim": opt.state_dict(),
                            "ema": ema.state_dict(),
                            "extra": extra_state,
                        },
                        os.path.join(args.ckpt_dir, "best_an.pt"),
                    )
        if step % tcfg.sleep_every == 0:
            ewc.update(model.parameters())


if __name__ == "__main__":
    train_main()
