import argparse
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from concept_learner.data.episode_gen import EpisodeConfig, EpisodeGenerator
from concept_learner.model.backbone import TinyBackbone
from concept_learner.model.vq_layer import EmaVectorQuantizer
from concept_learner.model.relation_head import DistMultHead, AnalogyProjector
from concept_learner.losses import EWC, info_nce, code_usage_entropy


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
        self.vq = EmaVectorQuantizer(cfg.num_codes, cfg.code_dim)
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


def run_step(model: ConceptLearner, gen: EpisodeGenerator, cfg: TrainConfig, optimizer, ewc: EWC, step_idx: int, sleep: bool = False):
    model.train()
    device = cfg.device
    losses = {}

    # Invariance & identification
    # Views curriculum: keep base fixed; schedule easy positives (identical remap)
    if step_idx < 2000:
        easy_prob = 1.0
    elif step_idx < 4000:
        easy_prob = 0.5
    elif step_idx < 6000:
        easy_prob = 0.2
    else:
        easy_prob = 0.0
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
    # Use pre-quantized features for InfoNCE (smoother alignment)
    z1 = model.proj1(e1["z"])  # pre-quantized
    z2 = model.proj2(e2["z"])  # pre-quantized
    # symmetric stop-grad InfoNCE
    loss_c12 = info_nce(z1.detach(), z2, temperature)
    loss_c21 = info_nce(z2.detach(), z1, temperature)
    loss_contrast = 0.5 * (loss_c12 + loss_c21)
    # Encourage same code across views via cross-entropy on indices (small weight)
    logits_codes = torch.matmul(e1["z_q"], model.vq.codebook.weight.t())
    loss_codes = nn.CrossEntropyLoss()(torch.log_softmax(logits_codes, dim=-1), e2["indices"])  # type: ignore
    losses["contrast"] = loss_contrast + cfg.loss_codes * loss_codes

    # Relational & analogical
    triples = gen.sample_triples(cfg.batch)
    s = model.encode(triples["s_desc"].to(device), triples["s_mask"].to(device))["z_q"]
    o = model.encode(triples["o_desc"].to(device), triples["o_mask"].to(device))["z_q"]
    o_neg = model.encode(triples["o_neg_desc"].to(device), triples["o_neg_mask"].to(device))["z_q"]
    r = triples["r"].to(device)
    pos = model.rel(s, r, o)
    neg = model.rel(s, r, o_neg)
    loss_rel = torch.clamp(1.0 - pos + neg, min=0.0).mean()
    losses["rel"] = loss_rel

    # Restrict analogies to parity only for first 4000 steps
    allowed = [0] if step_idx < 4000 else None
    analog = gen.sample_analogies(cfg.batch, allowed_relations=allowed)
    A = model.encode(analog["A_desc"].to(device), analog["A_mask"].to(device))["z_q"]
    B = model.encode(analog["B_desc"].to(device), analog["B_mask"].to(device))["z_q"]
    C = model.encode(analog["C_desc"].to(device), analog["C_mask"].to(device))["z_q"]
    D = model.encode(analog["D_desc"].to(device), analog["D_mask"].to(device))["z_q"]
    loss_analogy = model.analogy.analogy_loss(A, B, C, D)
    # Add offset loss in concept space
    r_off1 = A - B
    r_off2 = C - D
    loss_offset = F.mse_loss(r_off1, r_off2)
    losses["analogy"] = loss_analogy + 0.5 * loss_offset if step_idx < 2000 else loss_analogy + 0.2 * loss_offset

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
    total = (
        w_c * losses["contrast"]
        + w_rel * losses["rel"]
        + w_an * losses["analogy"]
        + cfg.loss_mdl * losses["mdl"]
        + cfg.loss_task * losses["task"]
        + cfg.loss_stability * losses["stability"]
        + cfg.loss_vq_entropy * losses["vq_ent"]
        + cfg.loss_vq_usage * losses["vq_usage"]
        + cfg.loss_vq * losses["vq"]
    )

    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return {k: v.detach().item() for k, v in losses.items()}, total.detach().item()


def train_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    tcfg = TrainConfig(device=args.device, steps=args.steps, batch=args.batch)
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

    for step in range(1, tcfg.steps + 1):
        # Increase stability penalty a bit during sleep steps
        if step % tcfg.sleep_every == 0:
            ewc.weight = 0.2
        else:
            ewc.weight = 0.0
        losses, total = run_step(model, gen, tcfg, opt, ewc, step)
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
            print(
                f"step {step:04d} total={total:.3f} "
                f"c={losses['contrast']:.3f} rel={losses['rel']:.3f} an={losses['analogy']:.3f} "
                f"mdl={losses['mdl']:.3f} task={losses['task']:.3f} vq={losses['vq']:.3f} st={losses['stability']:.3f} "
                f"ppx={perplexity:.2f} uniq={unique}/{model.num_codes}"
            )
        if step % tcfg.sleep_every == 0:
            ewc.update(model.parameters())


if __name__ == "__main__":
    train_main()
