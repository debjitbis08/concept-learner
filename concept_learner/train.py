import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from concept_learner.data.episode_gen import EpisodeConfig, EpisodeGenerator
from concept_learner.data.multi_domain import MultiDomainEpisodeGenerator
from concept_learner.model.backbone import TinyBackbone
from concept_learner.model.vq_layer import EmaVectorQuantizer
from concept_learner.model.domain import DomainAdapter
from concept_learner.model.relation_head import DistMultHead, AnalogyProjector
from concept_learner.losses import EWC, info_nce, code_usage_entropy
from utils.checkpoints import CheckpointManager
from utils.ema import EMA


class ReplayBuffer:
    """
    Tiny replay buffer storing indices for different episode types.
    We re-render descriptors on demand using the attached generator to avoid
    storing full token tensors in memory.
    """

    def __init__(self, gen: EpisodeGenerator, cap_per_type: int = 4096):
        self.gen = gen
        self.cap = cap_per_type
        # Store domain along with indices so we can re-render from the right generator
        self.views: List[Tuple[int, int]] = []  # (idx, dom)
        self.triples: List[Tuple[int, int, int, int]] = []  # (s,r,o,dom)
        self.analogies: List[Tuple[int, int, int, int, int]] = []  # (A,B,C,D,dom)
        self.pairs: List[Tuple[int, int, int, int]] = []  # (a,b,label,dom)

    def __len__(self) -> int:
        return max(len(self.views), len(self.triples), len(self.analogies), len(self.pairs))

    def add_from_batches(self, views: Dict[str, torch.Tensor], triples: Dict[str, torch.Tensor], analog: Dict[str, torch.Tensor], pairs: Dict[str, torch.Tensor]) -> None:
        # Views: just keep indices
        if "idx" in views:
            dom = views.get("domain", torch.zeros_like(views["idx"]))
            for v, d in zip(views["idx"].tolist(), dom.tolist()):
                self.views.append((int(v), int(d)))
        # Triples: (s,r,o)
        if all(k in triples for k in ("s_idx", "r", "o_idx")):
            dom = triples.get("domain", torch.zeros_like(triples["s_idx"]))
            for s, r, o, d in zip(
                triples["s_idx"].tolist(), triples["r"].tolist(), triples["o_idx"].tolist(), dom.tolist()
            ):
                self.triples.append((int(s), int(r), int(o), int(d)))
        # Analogies: (A,B,C,D)
        if all(k in analog for k in ("A_idx", "B_idx", "C_idx", "D_idx")):
            dom = analog.get("domain", torch.zeros_like(analog["A_idx"]))
            for A, B, C, D, d in zip(
                analog["A_idx"].tolist(),
                analog["B_idx"].tolist(),
                analog["C_idx"].tolist(),
                analog["D_idx"].tolist(),
                dom.tolist(),
            ):
                self.analogies.append((int(A), int(B), int(C), int(D), int(d)))
        # Pairs: (a,b,label)
        if all(k in pairs for k in ("a_idx", "b_idx", "label")):
            dom = pairs.get("domain", torch.zeros_like(pairs["a_idx"]))
            for a, b, y, d in zip(
                pairs["a_idx"].tolist(), pairs["b_idx"].tolist(), pairs["label"].tolist(), dom.tolist()
            ):
                self.pairs.append((int(a), int(b), int(y), int(d)))

        # Truncate to capacity (keep most recent)
        self.views = self.views[-self.cap :]
        self.triples = self.triples[-self.cap :]
        self.analogies = self.analogies[-self.cap :]
        self.pairs = self.pairs[-self.cap :]

    def _choice(self, n: int, size: int) -> List[int]:
        idx = torch.randint(0, max(1, n), (size,), device=self.gen.cfg.device)
        return idx.tolist()

    def _render_one(self, idx: int, dom: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Support EpisodeGenerator and MultiDomainEpisodeGenerator
        if hasattr(self.gen, "_render_batch"):
            import torch as _torch

            t = _torch.tensor([idx], device=self.gen.cfg.device)
            return self.gen._render_batch(t)  # type: ignore[attr-defined]
        elif hasattr(self.gen, "gens"):
            g = self.gen.gens[dom]  # type: ignore[attr-defined]
            import torch as _torch

            t = _torch.tensor([idx], device=g.cfg.device)
            return g._render_batch(t)
        else:
            raise RuntimeError("Unsupported generator type for ReplayBuffer re-rendering")

    def sample_views(self, batch: int) -> Optional[Dict[str, torch.Tensor]]:
        if len(self.views) < batch:
            return None
        choice = self._choice(len(self.views), batch)
        samples = [self.views[i] for i in choice]
        # Render per-sample to match domain
        outs1 = [self._render_one(ix, dm) for ix, dm in samples]
        outs2 = [self._render_one(ix, dm) for ix, dm in samples]
        v1_desc = torch.cat([o[0] for o in outs1], dim=0)
        v1_mask = torch.cat([o[1] for o in outs1], dim=0)
        v1_base = torch.cat([o[2] for o in outs1], dim=0)
        v2_desc = torch.cat([o[0] for o in outs2], dim=0)
        v2_mask = torch.cat([o[1] for o in outs2], dim=0)
        v2_base = torch.cat([o[2] for o in outs2], dim=0)
        idx = torch.tensor([ix for ix, _ in samples], device=self.gen.cfg.device)
        dom = torch.tensor([dm for _, dm in samples], device=self.gen.cfg.device)
        return {
            "idx": idx,
            "view1_desc": v1_desc,
            "view1_mask": v1_mask,
            "view1_base": v1_base,
            "view2_desc": v2_desc,
            "view2_mask": v2_mask,
            "view2_base": v2_base,
            "domain": dom,
        }

    def sample_triples(self, batch: int) -> Optional[Dict[str, torch.Tensor]]:
        if len(self.triples) < batch:
            return None
        choice = self._choice(len(self.triples), batch)
        s = torch.tensor([self.triples[i][0] for i in choice], device=self.gen.cfg.device)
        r = torch.tensor([self.triples[i][1] for i in choice], device=self.gen.cfg.device)
        o = torch.tensor([self.triples[i][2] for i in choice], device=self.gen.cfg.device)
        dom = torch.tensor([self.triples[i][3] for i in choice], device=self.gen.cfg.device)
        outs_s = [self._render_one(int(si), int(di)) for si, di in zip(s.tolist(), dom.tolist())]
        outs_o = [self._render_one(int(oi), int(di)) for oi, di in zip(o.tolist(), dom.tolist())]
        s_desc = torch.cat([o_[0] for o_ in outs_s], dim=0)
        s_mask = torch.cat([o_[1] for o_ in outs_s], dim=0)
        s_base = torch.cat([o_[2] for o_ in outs_s], dim=0)
        o_desc = torch.cat([o_[0] for o_ in outs_o], dim=0)
        o_mask = torch.cat([o_[1] for o_ in outs_o], dim=0)
        o_base = torch.cat([o_[2] for o_ in outs_o], dim=0)
        return {
            "s_idx": s,
            "r": r,
            "o_idx": o,
            "s_desc": s_desc,
            "s_mask": s_mask,
            "s_base": s_base,
            "o_desc": o_desc,
            "o_mask": o_mask,
            "o_base": o_base,
            "domain": dom,
        }

    def sample_analogies(self, batch: int) -> Optional[Dict[str, torch.Tensor]]:
        if len(self.analogies) < batch:
            return None
        choice = self._choice(len(self.analogies), batch)
        A = torch.tensor([self.analogies[i][0] for i in choice], device=self.gen.cfg.device)
        B = torch.tensor([self.analogies[i][1] for i in choice], device=self.gen.cfg.device)
        C = torch.tensor([self.analogies[i][2] for i in choice], device=self.gen.cfg.device)
        D = torch.tensor([self.analogies[i][3] for i in choice], device=self.gen.cfg.device)
        dom = torch.tensor([self.analogies[i][4] for i in choice], device=self.gen.cfg.device)
        outA = [self._render_one(int(ai), int(di)) for ai, di in zip(A.tolist(), dom.tolist())]
        outB = [self._render_one(int(bi), int(di)) for bi, di in zip(B.tolist(), dom.tolist())]
        outC = [self._render_one(int(ci), int(di)) for ci, di in zip(C.tolist(), dom.tolist())]
        outD = [self._render_one(int(di_), int(di2)) for di_, di2 in zip(D.tolist(), dom.tolist())]
        A_desc = torch.cat([o_[0] for o_ in outA], dim=0)
        A_mask = torch.cat([o_[1] for o_ in outA], dim=0)
        A_base = torch.cat([o_[2] for o_ in outA], dim=0)
        B_desc = torch.cat([o_[0] for o_ in outB], dim=0)
        B_mask = torch.cat([o_[1] for o_ in outB], dim=0)
        B_base = torch.cat([o_[2] for o_ in outB], dim=0)
        C_desc = torch.cat([o_[0] for o_ in outC], dim=0)
        C_mask = torch.cat([o_[1] for o_ in outC], dim=0)
        C_base = torch.cat([o_[2] for o_ in outC], dim=0)
        D_desc = torch.cat([o_[0] for o_ in outD], dim=0)
        D_mask = torch.cat([o_[1] for o_ in outD], dim=0)
        D_base = torch.cat([o_[2] for o_ in outD], dim=0)
        return {
            "A_idx": A,
            "B_idx": B,
            "C_idx": C,
            "D_idx": D,
            "A_desc": A_desc,
            "A_mask": A_mask,
            "A_base": A_base,
            "B_desc": B_desc,
            "B_mask": B_mask,
            "B_base": B_base,
            "C_desc": C_desc,
            "C_mask": C_mask,
            "C_base": C_base,
            "D_desc": D_desc,
            "D_mask": D_mask,
            "D_base": D_base,
            "domain": dom,
        }

    def sample_pairs(self, batch: int) -> Optional[Dict[str, torch.Tensor]]:
        if len(self.pairs) < batch:
            return None
        choice = self._choice(len(self.pairs), batch)
        a = torch.tensor([self.pairs[i][0] for i in choice], device=self.gen.cfg.device)
        b = torch.tensor([self.pairs[i][1] for i in choice], device=self.gen.cfg.device)
        y = torch.tensor([self.pairs[i][2] for i in choice], device=self.gen.cfg.device)
        dom = torch.tensor([self.pairs[i][3] for i in choice], device=self.gen.cfg.device)
        outA = [self._render_one(int(ai), int(di)) for ai, di in zip(a.tolist(), dom.tolist())]
        outB = [self._render_one(int(bi), int(di)) for bi, di in zip(b.tolist(), dom.tolist())]
        a_desc = torch.cat([o_[0] for o_ in outA], dim=0)
        a_mask = torch.cat([o_[1] for o_ in outA], dim=0)
        a_base = torch.cat([o_[2] for o_ in outA], dim=0)
        b_desc = torch.cat([o_[0] for o_ in outB], dim=0)
        b_mask = torch.cat([o_[1] for o_ in outB], dim=0)
        b_base = torch.cat([o_[2] for o_ in outB], dim=0)
        return {
            "a_idx": a,
            "b_idx": b,
            "a_desc": a_desc,
            "a_mask": a_mask,
            "a_base": a_base,
            "b_desc": b_desc,
            "b_mask": b_mask,
            "b_base": b_base,
            "label": y.long(),
            "domain": dom,
        }


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
    # Multi-domain settings
    num_domains: int = 3
    private_codes: int = 16
    use_private_vq: bool = True
    use_domain_token: bool = True
    adapter_rank: int = 16
    use_domain_rel_offsets: bool = True
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
    # Multi-domain overlap regularizer
    loss_cross_domain_align: float = 0.2
    # Context token controls
    use_context_tokens: bool = True
    # Instance permanence head
    use_instance_head: bool = True
    loss_instance: float = 0.3


class ConceptLearner(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.backbone = TinyBackbone(cfg.vocab_size, cfg.d_model, cfg.nhead, cfg.num_layers, cfg.max_len)
        self.domain_adapter = DomainAdapter(cfg.num_domains, cfg.d_model, adapter_rank=cfg.adapter_rank)
        self.use_domain_token = cfg.use_domain_token and cfg.num_domains > 0
        if self.use_domain_token:
            from concept_learner.model.domain import DomainToken

            self.domain_token = DomainToken(cfg.num_domains, cfg.d_model)
        else:
            self.domain_token = None
        self.to_code = nn.Linear(cfg.d_model, cfg.code_dim)
        # Global (shared) VQ
        self.vq_global = EmaVectorQuantizer(cfg.num_codes, cfg.code_dim, decay=0.995)
        # Optional per-domain private residual VQs
        self.use_private_vq = cfg.use_private_vq and cfg.num_domains > 0 and cfg.private_codes > 0
        if self.use_private_vq:
            self.vq_private = nn.ModuleList(
                [EmaVectorQuantizer(cfg.private_codes, cfg.code_dim, decay=0.995) for _ in range(cfg.num_domains)]
            )
        else:
            self.vq_private = None
        self.rel = DistMultHead(cfg.code_dim, cfg.relations)
        self.use_domain_rel_offsets = cfg.use_domain_rel_offsets and cfg.num_domains > 1
        if self.use_domain_rel_offsets:
            self.rel_offsets = nn.Parameter(torch.zeros(cfg.num_domains, cfg.relations, cfg.code_dim))
        else:
            self.rel_offsets = None
        self.analogy = AnalogyProjector(cfg.code_dim, proj_dim=min(32, cfg.code_dim))
        self.same_head = nn.Sequential(
            nn.Linear(cfg.code_dim * 2, cfg.code_dim), nn.ReLU(), nn.Linear(cfg.code_dim, 2)
        )
        # Optional instance permanence head (same-instance across views)
        self.use_instance_head = cfg.use_instance_head
        if self.use_instance_head:
            self.instance_head = nn.Sequential(
                nn.Linear(cfg.code_dim * 2, cfg.code_dim), nn.ReLU(), nn.Linear(cfg.code_dim, 2)
            )
        else:
            self.instance_head = None
        # Expose global codebook meta for compatibility
        self.num_codes = cfg.num_codes
        # Projection heads for contrastive stability
        self.proj1 = nn.Sequential(
            nn.Linear(cfg.code_dim, cfg.code_dim), nn.ReLU(), nn.LayerNorm(cfg.code_dim)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(cfg.code_dim, cfg.code_dim), nn.ReLU(), nn.LayerNorm(cfg.code_dim)
        )

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor | None = None, domain: torch.Tensor | int | None = None) -> Dict[str, torch.Tensor]:
        h = self.backbone(tokens, mask)
        if self.use_domain_token and self.domain_token is not None:
            h = self.domain_token(h, domain)
        h = self.domain_adapter(h, domain)
        z = self.to_code(h)
        # Global quantization
        z_g, idx_g, loss_g = self.vq_global(z)
        if self.use_private_vq:
            # Residual quantization per domain
            if domain is None:
                d = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            elif isinstance(domain, int):
                d = torch.tensor([domain], dtype=torch.long, device=z.device).expand(z.size(0))
            else:
                d = domain
            z_res = z - z_g.detach()
            # Gather per-sample private VQ by looping in small batches; domains are tiny here
            z_priv_list = []
            loss_p_list = []
            for dom_id in d.unique().tolist():
                dom_id_int = int(dom_id)
                sel = (d == dom_id_int)
                if sel.any():
                    z_chunk = z_res[sel]
                    z_p, _idx_p, loss_p = self.vq_private[dom_id_int](z_chunk)
                    z_priv_list.append((sel, z_p))
                    loss_p_list.append(loss_p)
            # Merge private chunks back
            z_p_full = torch.zeros_like(z)
            for sel, z_p in z_priv_list:
                z_p_full[sel] = z_p
            z_q = z_g + z_p_full
            vq_loss = loss_g + (sum(loss_p_list) if len(loss_p_list) > 0 else 0.0)
        else:
            z_q, idx_g, loss_g = z_g, idx_g, loss_g
            vq_loss = loss_g
        return {"h": h, "z": z, "z_q": z_q, "indices": idx_g, "vq_loss": vq_loss}


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
    replay: ReplayBuffer | None = None,
):
    model.train()
    device = cfg.device
    losses = {}

    # Invariance & identification
    # Views curriculum: keep base fixed; schedule easy positives (identical remap)
    if step_idx < 6000:
        easy_prob = 1.0
    elif step_idx < 10000:
        easy_prob = 0.5
    elif step_idx < 14000:
        easy_prob = 0.2
    else:
        easy_prob = 0.0
    # Base-change schedule (stay off for a long time)
    if step_idx < 10000:
        change_base_prob = 0.0
    elif step_idx < 14000:
        change_base_prob = 0.5
    else:
        change_base_prob = 1.0
    if adapt and adapt.get("recovery", False):
        easy_prob = 1.0
        change_base_prob = 0.0
    if sleep and replay is not None:
        views = replay.sample_views(cfg.batch) or gen.sample_views(
            cfg.batch, change_base_prob=change_base_prob, easy_same_remap_prob=easy_prob
        )
    else:
        views = gen.sample_views(cfg.batch, change_base_prob=change_base_prob, easy_same_remap_prob=easy_prob)
    v1 = views["view1_desc"].to(device)
    v1m = views["view1_mask"].to(device)
    v2 = views["view2_desc"].to(device)
    v2m = views["view2_mask"].to(device)
    dom_views = views.get("domain", torch.zeros(v1.size(0), dtype=torch.long, device=device)).to(device)
    e1 = model.encode(v1, v1m, dom_views)
    e2 = model.encode(v2, v2m, dom_views)
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
    logits_codes = torch.matmul(e1["z_q"], model.vq_global.codebook.weight.t())
    losses["code_ce"] = nn.CrossEntropyLoss()(torch.log_softmax(logits_codes, dim=-1), e2["indices"])  # type: ignore

    # Relational & analogical
    triples = replay.sample_triples(cfg.batch) if (sleep and replay is not None) else None
    if triples is None:
        triples = gen.sample_triples(cfg.batch)
    dom_tri = triples.get("domain", torch.zeros(cfg.batch, dtype=torch.long, device=device)).to(device)
    s = model.encode(triples["s_desc"].to(device), triples["s_mask"].to(device), dom_tri)["z_q"]
    o = model.encode(triples["o_desc"].to(device), triples["o_mask"].to(device), dom_tri)["z_q"]
    r = triples["r"].to(device)
    # In-batch multiclass relation loss (many negatives)
    # v_i = s_i * W_{r_i}; logits = v @ o^T
    if getattr(model, "use_domain_rel_offsets", False) and getattr(model, "rel_offsets", None) is not None:
        dom_rel = dom_tri if 'dom_tri' in locals() else torch.zeros_like(r)
        W_sel = model.rel.rel[r] + model.rel_offsets[dom_rel, r]
    else:
        W_sel = model.rel.rel[r]  # (B, D)
    v = s * W_sel
    logits_rel = torch.matmul(v, o.t())
    labels_rel = torch.arange(logits_rel.size(0), device=device)
    loss_rel = nn.CrossEntropyLoss()(logits_rel, labels_rel)
    losses["rel"] = loss_rel

    # Restrict analogies to parity only for a long warmup
    allowed = [0] if step_idx < 8000 else None
    analog = replay.sample_analogies(cfg.batch) if (sleep and replay is not None) else None
    if analog is None:
        analog = gen.sample_analogies(cfg.batch, allowed_relations=allowed)
    dom_an = analog.get("domain", torch.zeros(cfg.batch, dtype=torch.long, device=device)).to(device)
    A = model.encode(analog["A_desc"].to(device), analog["A_mask"].to(device), dom_an)["z_q"]
    B = model.encode(analog["B_desc"].to(device), analog["B_mask"].to(device), dom_an)["z_q"]
    C = model.encode(analog["C_desc"].to(device), analog["C_mask"].to(device), dom_an)["z_q"]
    D = model.encode(analog["D_desc"].to(device), analog["D_mask"].to(device), dom_an)["z_q"]
    # Analogy classification with scheduled temperature
    an_temp = 0.5 if step_idx < 8000 else 0.2
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
        codebook = model.vq_global.codebook.weight.detach()  # (K, D)
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
    pairs = replay.sample_pairs(cfg.batch) if (sleep and replay is not None) else None
    if pairs is None:
        pairs = gen.sample_posneg_pairs(cfg.batch)
    dom_pairs = pairs.get("domain", torch.zeros(cfg.batch, dtype=torch.long, device=device)).to(device)
    a = F.normalize(model.encode(pairs["a_desc"].to(device), pairs["a_mask"].to(device), dom_pairs)["z_q"], dim=-1)
    b = F.normalize(model.encode(pairs["b_desc"].to(device), pairs["b_mask"].to(device), dom_pairs)["z_q"], dim=-1)
    y = pairs["label"].to(device)
    logits = model.same_head(torch.cat([a, b], dim=-1))
    loss_task = nn.CrossEntropyLoss()(logits, y)
    losses["task"] = loss_task

    # Instance permanence head: predict whether two views are the same instance
    if getattr(model, "use_instance_head", False) and model.instance_head is not None:
        # Positive pairs: views from the same idx; Negatives: mismatched pairs within batch
        pos_a = F.normalize(e1["z_q"], dim=-1)
        pos_b = F.normalize(e2["z_q"], dim=-1)
        neg_b = torch.roll(pos_b, shifts=1, dims=0)
        logits_pos = model.instance_head(torch.cat([pos_a, pos_b], dim=-1))
        logits_neg = model.instance_head(torch.cat([pos_a, neg_b], dim=-1))
        logits_all = torch.cat([logits_pos, logits_neg], dim=0)
        labels_all = torch.cat([
            torch.ones(pos_a.size(0), dtype=torch.long, device=device),
            torch.zeros(pos_a.size(0), dtype=torch.long, device=device),
        ], dim=0)
        loss_inst = nn.CrossEntropyLoss()(logits_all, labels_all)
        losses["instance"] = loss_inst
    else:
        losses["instance"] = torch.tensor(0.0, device=device)

    # VQ commitment/codebook
    losses["vq"] = e1["vq_loss"] + e2["vq_loss"]

    # Cross-domain alignment regularizer on known equivalent pairs (if available)
    if cfg.num_domains > 1 and getattr(cfg, "loss_cross_domain_align", 0.0) > 0.0:
        if hasattr(gen, "sample_equivalent_pairs"):
            eq = gen.sample_equivalent_pairs(max(8, cfg.batch // 4))  # small extra batch
            X1 = model.encode(eq["x1_desc"].to(device), eq["x1_mask"].to(device), eq["x1_domain"].to(device))
            X2 = model.encode(eq["x2_desc"].to(device), eq["x2_mask"].to(device), eq["x2_domain"].to(device))
            zx1 = model.proj1(X1["z_q"])  # use quantized for tight alignment
            zx2 = model.proj2(X2["z_q"])
            # Symmetric InfoNCE across the two domains
            loss_x12 = info_nce(zx1.detach(), zx2, temperature=0.5)
            loss_x21 = info_nce(zx2.detach(), zx1, temperature=0.5)
            losses["xdom_align"] = 0.5 * (loss_x12 + loss_x21)

    # Stability (EWC-style) during sleep, smaller during day
    pen = ewc.penalty()
    losses["stability"] = pen

    # Mix
    # Scheduled weights
    w_c = 0.1 if step_idx < 8000 else cfg.loss_contrastive
    w_rel = 2.0 if step_idx < 8000 else cfg.loss_rel
    w_an = 1.5 if step_idx < 8000 else cfg.loss_analogy
    w_mdl = cfg.loss_mdl
    w_vq_usage = cfg.loss_vq_usage
    w_codes = cfg.loss_codes
    if adapt and adapt.get("recovery", False):
        w_c = 0.0
        w_rel = max(w_rel, 3.0)
        w_an = max(w_an, 2.0)
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
        + cfg.loss_instance * losses.get("instance", torch.tensor(0.0, device=device))
        + cfg.loss_stability * losses["stability"]
        + cfg.loss_vq_entropy * losses["vq_ent"]
        + w_vq_usage * losses["vq_usage"]
        + cfg.loss_vq * losses["vq"]
        + (getattr(cfg, "loss_cross_domain_align", 0.0) * losses.get("xdom_align", torch.tensor(0.0, device=device)))
    )

    optimizer.zero_grad()
    total.backward()
    clip_val = 0.2 if (adapt and adapt.get("recovery", False)) else 0.5
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
    optimizer.step()
    if ema is not None:
        ema.update(model)

    # Update replay with most recent on-policy batches (not during sleep)
    if (replay is not None) and (not sleep):
        try:
            replay.add_from_batches(views, triples, analog, pairs)
        except Exception:
            pass

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

    # Allow a slightly larger vocab by default to host context tokens comfortably
    tcfg = TrainConfig(device=device, steps=args.steps, batch=args.batch, vocab_size=32)
    ecfg = EpisodeConfig(device=tcfg.device)
    if tcfg.num_domains > 1:
        gens: List[EpisodeGenerator] = []
        for d in range(tcfg.num_domains):
            # Create slight domain variations to simulate distinct domains
            cfg_d = EpisodeConfig(
                device=tcfg.device,
                max_number=ecfg.max_number,
                max_len=ecfg.max_len,
                min_base=5 + (d % 3),
                max_base=10 - (d % 2),
                canonicalize=(d % 2 == 0),
            )
            g_d = EpisodeGenerator(cfg_d)
            # Optionally enable context tokens as <domain> tags
            if tcfg.use_context_tokens:
                start_tok = cfg_d.max_base + 2  # 1..max_base digits; PAD=max_base+1; context starts here
                g_d.set_context_token_id(start_tok + d)
            gens.append(g_d)
        gen: EpisodeGenerator | MultiDomainEpisodeGenerator = MultiDomainEpisodeGenerator(gens, domain_batch_mix="uniform")
    else:
        gen = EpisodeGenerator(ecfg)
        if tcfg.use_context_tokens:
            start_tok = ecfg.max_base + 2
            gen.set_context_token_id(start_tok)

    model = ConceptLearner(tcfg).to(tcfg.device)
    # Optimizer with param groups
    enc_params = list(model.backbone.parameters()) + list(model.to_code.parameters()) + list(model.domain_adapter.parameters())
    rel_params = list(model.rel.parameters()) + list(model.analogy.parameters()) + list(model.same_head.parameters())
    if getattr(model, "rel_offsets", None) is not None:
        rel_params += [model.rel_offsets]
    vq_params = list(model.vq_global.parameters()) + (list(model.vq_private.parameters()) if model.vq_private is not None else []) + list(model.proj1.parameters()) + list(model.proj2.parameters())
    opt = optim.AdamW(
        [
            {"params": enc_params, "lr": 1e-3},
            {"params": rel_params, "lr": 2e-3},
            {"params": vq_params, "lr": 2e-3},
        ],
        weight_decay=1e-2,
    )
    ewc = EWC(model.parameters(), weight=0.0)
    replay = ReplayBuffer(gen)

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
    last_an_acc = float('nan')
    for step in range(start_step, end_step + 1):
        # Increase stability penalty a bit during sleep steps
        if step % tcfg.sleep_every == 0:
            ewc.weight = 0.2
        else:
            ewc.weight = 0.0
        adapt = {"recovery": step <= recovery_until}
        losses, total = run_step(model, gen, tcfg, opt, ewc, step, ema, adapt, sleep=False, replay=replay)
        # Mini-nap: a few replay-only consolidation steps
        if step % tcfg.sleep_every == 0 and len(replay) > 0:
            nap_iters = max(1, min(tcfg.sleep_steps, len(replay) // max(1, tcfg.batch)))
            for _ in range(nap_iters):
                ewc.weight = 0.2
                _losses, _total = run_step(model, gen, tcfg, opt, ewc, step, ema, {"recovery": True}, sleep=True, replay=replay)
        if step % 10 == 0 or step == 1:
            # Codebook stats (approximate, from a small batch)
            # For lightweight logging, recompute a small batch indices
            small = gen.sample_views(64)
            e_small = model.encode(small["view1_desc"].to(tcfg.device), small["view1_mask"].to(tcfg.device), small.get("domain", None))
            with torch.no_grad():
                cb = model.vq_global.codebook.weight.detach()
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
            # Quick in-batch analogy accuracy probe (parity-only early), more frequent and persistent value
            if step % 100 == 0:
                with torch.no_grad():
                    allowed_eval = [0] if step < 8000 else None
                    probe_bs = 64
                    analog_small = gen.sample_analogies(probe_bs, allowed_relations=allowed_eval)
                    Aev = model.encode(analog_small["A_desc"].to(tcfg.device), analog_small["A_mask"].to(tcfg.device))["z_q"]
                    Bev = model.encode(analog_small["B_desc"].to(tcfg.device), analog_small["B_mask"].to(tcfg.device))["z_q"]
                    Cev = model.encode(analog_small["C_desc"].to(tcfg.device), analog_small["C_mask"].to(tcfg.device))["z_q"]
                    Dev = model.encode(analog_small["D_desc"].to(tcfg.device), analog_small["D_mask"].to(tcfg.device))["z_q"]
                    r_ab = model.analogy.rel_vec(Aev, Bev)
                    r_cd_all = model.analogy.rel_vec(Cev.unsqueeze(1), Dev.unsqueeze(0))
                    sim = torch.einsum("bp,bnp->bn", F.normalize(r_ab, dim=-1), F.normalize(r_cd_all, dim=-1))
                    pred = sim.argmax(dim=-1)
                    labels = torch.arange(sim.size(0), device=tcfg.device)
                    last_an_acc = (pred == labels).float().mean().item()
            an_acc = last_an_acc
            # Additional child-like metrics every 200 steps
            if step % 200 == 0:
                try:
                    # Code sharing ratio across two domains for same item ids
                    if hasattr(gen, "sample_equivalent_pairs") and tcfg.num_domains > 1:
                        eq = gen.sample_equivalent_pairs(128)
                        X1 = model.encode(eq["x1_desc"].to(tcfg.device), eq["x1_mask"].to(tcfg.device), eq["x1_domain"].to(tcfg.device))
                        X2 = model.encode(eq["x2_desc"].to(tcfg.device), eq["x2_mask"].to(tcfg.device), eq["x2_domain"].to(tcfg.device))
                        share = (X1["indices"] == X2["indices"]).float().mean().item()
                    else:
                        share = float('nan')
                    # Nearest-neighbor diversity across domains
                    mv = gen.sample_views(128)
                    Z = model.encode(mv["view1_desc"].to(tcfg.device), mv["view1_mask"].to(tcfg.device), mv.get("domain", None))["z_q"]
                    doms = mv.get("domain", torch.zeros(Z.size(0), dtype=torch.long, device=tcfg.device)).to(tcfg.device)
                    sims = torch.matmul(F.normalize(Z, -1), F.normalize(Z, -1).t())
                    sims.fill_diagonal_(-1.0)
                    topk = sims.topk(k=5, dim=1).indices
                    nn_div = []
                    for i in range(Z.size(0)):
                        nn_div.append((doms[topk[i]] != doms[i]).float().mean().item())
                    nn_div = sum(nn_div) / max(1, len(nn_div))
                except Exception:
                    share, nn_div = float('nan'), float('nan')
            c_disp = (losses.get('contrast_align', 0.0) + losses.get('code_ce', 0.0))
            baseline = 1.0 / 64.0
            print(
                f"step {step:04d} total={total:.3f} "
                f"c={c_disp:.3f} rel={losses['rel']:.3f} an={losses['analogy']:.3f} "
                f"mdl={losses['mdl']:.3f} task={losses['task']:.3f} vq={losses['vq']:.3f} st={losses['stability']:.3f} "
                f"ppx={perplexity:.2f} uniq={unique}/{model.num_codes} "
                f"an_acc={an_acc:.3f} (EMA, in-batch parity; random~{baseline:.3f}) rec={(step<=recovery_until)}"
            )
            if step % 200 == 0:
                print(f"  metrics: code_share={share:.3f} nn_div={nn_div:.3f}")
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
                        state = torch.load(chosen, map_location="cpu")
                        model.load_state_dict(state.get("model", state))
                        if state.get("optim") is not None:
                            opt.load_state_dict(state["optim"])  # type: ignore
                        if state.get("ema") is not None:
                            ema.load_state_dict(state["ema"])  # type: ignore
                        # Refresh EWC snapshot to the recovered params
                        ewc.update(model.parameters())
                        recovery_until = step + 500
                        print(f"[watchdog] Recovered from {chosen}; entering recovery mode until step {recovery_until}")
                    except Exception as e:
                        print(f"[watchdog] Failed to recover from {chosen}: {e}")
                collapse_counter = 0

        # Checkpointing and periodic eval
        if (step % max(1, args.save_every)) == 0 or step == end_step:
            latest_path = os.path.join(args.ckpt_dir, "latest.pt")
            state = {
                "step": step,
                "model": model.state_dict(),
                "optim": opt.state_dict(),
                "ema": ema.state_dict(),
                "extra": {},
            }
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save(state, latest_path)
            # Update bests
            try:
                if losses["rel"] < best_rel:
                    best_rel = float(losses["rel"]) 
                    torch.save(state, os.path.join(args.ckpt_dir, "best_rel.pt"))
                # Probe EMA analogy accuracy (parity-only early)
                allowed_eval = [0] if step < 8000 else None
                an_acc_probe = probe_analogy_acc(batch_size=128, allowed_eval=allowed_eval)
                if an_acc_probe > best_an:
                    best_an = float(an_acc_probe)
                    torch.save(state, os.path.join(args.ckpt_dir, "best_an.pt"))
            except Exception:
                pass

        # Curriculum hook: example toggle for canonicalization later
        if step in curriculum_boundaries:
            if step >= 10000:
                gen.cfg.canonicalize = False

    print("[train] Done.")


if __name__ == "__main__":
    train_main()
