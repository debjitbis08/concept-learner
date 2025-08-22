from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn

from .data.episode_gen import EpisodeConfig, EpisodeGenerator
from .model.backbone import TinyBackbone
from .model.vq_layer import EmaVectorQuantizer
from .model.domain import DomainAdapter, DomainToken
from .model.relation_head import DistMultHead, AnalogyProjector
from .teacher.llm_teacher import LLMTeacher


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
        self.enc = nn.Linear(cfg.d_model, cfg.code_dim)
        self.vq_global = EmaVectorQuantizer(num_codes=cfg.num_codes, code_dim=cfg.code_dim, commitment_cost=0.25)

        # Small projection heads used in the training loop (kept for BC)
        self.proj1 = nn.Linear(cfg.code_dim, cfg.code_dim)
        self.proj2 = nn.Linear(cfg.code_dim, cfg.code_dim)

        # Relation and analogy heads
        self.rel = DistMultHead(concept_dim=cfg.code_dim, num_relations=cfg.num_relations)
        self.analogy = AnalogyProjector(dim=cfg.code_dim, proj_dim=min(32, cfg.code_dim))

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
        z = self.enc(h)
        z_q, indices, vq_loss = self.vq_global(z)
        return {"h": h, "z": z, "z_q": z_q, "indices": indices, "vq_loss": vq_loss}


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
    s_desc, s_mask, _ = gen._render_batch(s_idx)
    o_desc, o_mask, _ = gen._render_batch(o_idx)
    return {
        "s_idx": s_idx,
        "r": r,
        "o_idx": o_idx,
        "s_desc": s_desc,
        "s_mask": s_mask,
        "o_desc": o_desc,
        "o_mask": o_mask,
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
    parser.add_argument("--use_llm_teacher", action="store_true", help="use OpenAI-powered LLMTeacher for numbers triples")
    parser.add_argument("--llm_model", default="gpt-4o-mini", help="OpenAI model name for LLMTeacher")
    args = parser.parse_args()
    device = _resolve_device(args.device)
    print(f"[train] device={device}")

    ecfg = EpisodeConfig(device=device)
    tcfg = TrainConfig(device=device)
    gen = EpisodeGenerator(ecfg)
    model = ConceptLearner(tcfg, ecfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    teacher = LLMTeacher(None, model=args.llm_model) if args.use_llm_teacher else None
    model.train()
    for step in range(1, args.steps + 1):
        if teacher is not None:
            eps = teacher.gen_batch(args.batch)
            triple_batch = None
            # pick first numbers KG payload if available
            for e in eps:
                if e.get("domain") == "numbers" and e.get("type") == "kg":
                    triple_batch = _build_triple_batch_from_llm(e.get("payload", {}), gen, device, args.batch)
                    if triple_batch is not None:
                        break
            batch = triple_batch if triple_batch is not None else gen.sample_triples(args.batch)
        else:
            batch = gen.sample_triples(args.batch)
        enc_s = model.encode(batch["s_desc"].to(device), batch["s_mask"].to(device))
        enc_o = model.encode(batch["o_desc"].to(device), batch["o_mask"].to(device))
        s = enc_s["z_q"]
        o = enc_o["z_q"]
        r = batch["r"].to(device)

        # In-batch CE for triple scoring (DistMult via in-batch negatives)
        w = model.rel.rel[r]
        v = s * w
        logits = torch.matmul(v, o.t())
        labels = torch.arange(logits.size(0), device=device)
        ce_loss = nn.CrossEntropyLoss()(logits, labels)

        # VQ commitment
        vq_s = enc_s["vq_loss"]
        vq_o = enc_o["vq_loss"]
        loss = ce_loss + 0.5 * vq_s + 0.5 * vq_o

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0:
            with torch.no_grad():
                B = logits.size(0)
                pred = logits.argmax(dim=-1)
                acc = (pred == labels).float().mean().item()
                import math
                lnB = math.log(max(1, B))
                print(
                    f"step {step:05d} total {loss.item():.4f} | ce {ce_loss.item():.4f} (lnB~{lnB:.2f}) | "
                    f"vq_s {float(vq_s):.4f} vq_o {float(vq_o):.4f} | acc {acc:.3f} B {B}"
                )

    print("[train] done.")


if __name__ == "__main__":  # pragma: no cover
    train_main()
