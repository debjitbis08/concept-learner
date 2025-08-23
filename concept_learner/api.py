from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List

import torch
import torch.nn.functional as F

from .train import ConceptLearner, TrainConfig
from .data.episode_gen import EpisodeConfig, EpisodeGenerator
from utils.ema import EMA


@dataclass
class AnalogyResult:
    prediction: Any
    scores: Dict[str, float]
    concepts: Dict[str, Any]


class ConceptAPI:
    """
    Lightweight API around the ConceptLearner for inference and tooling.
    """

    def __init__(self, model: ConceptLearner, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.device = device
        self._ecfg = EpisodeConfig(device=device)
        self._gen = EpisodeGenerator(self._ecfg)

    @classmethod
    def load(cls, ckpt_path: str, device: str = "cpu") -> "ConceptAPI":
        cfg = TrainConfig(device=device)
        model = ConceptLearner(cfg)
        state = torch.load(ckpt_path, map_location="cpu")
        payload = state.get("model", state)
        try:
            missing, unexpected = model.load_state_dict(payload, strict=False)  # type: ignore[arg-type]
            if missing or unexpected:
                print(f"[api] Warning: non-strict load. missing={len(missing)} unexpected={len(unexpected)}")
        except TypeError:
            # Torch <2.2 returns None, fall back to strict=False without unpack
            model.load_state_dict(payload, strict=False)  # type: ignore[arg-type]
        # Apply EMA weights if available
        ema_state = state.get("ema", None)
        if ema_state is not None:
            ema = EMA(model)
            ema.load_state_dict(ema_state)
            ema.apply_to(model)
        return cls(model, device=device)

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            return self.model.encode(tokens.to(self.device), mask.to(self.device) if mask is not None else None)

    def complete_analogy(self, A: Any, B: Any, C: Any, top_k: int = 5) -> AnalogyResult:
        """
        Complete analogy A:B :: C:?. Numbers domain: A,B,C are integers in [0, max_number).
        Returns the best D and top-k scores over candidate D in the current generator domain.
        """
        # Parse inputs as integers for the toy numbers domain
        try:
            a = int(str(A).strip())
            b = int(str(B).strip())
            c = int(str(C).strip())
        except Exception:
            return AnalogyResult(prediction=None, scores={"error": "Inputs must be integers for the numbers domain."}, concepts={})

        n = self._gen.n_items
        a = max(0, min(n - 1, a))
        b = max(0, min(n - 1, b))
        c = max(0, min(n - 1, c))

        with torch.no_grad():
            A_desc, A_mask, _ = self._gen._render_batch(torch.tensor([a], device=self.device))
            B_desc, B_mask, _ = self._gen._render_batch(torch.tensor([b], device=self.device))
            C_desc, C_mask, _ = self._gen._render_batch(torch.tensor([c], device=self.device))
            D_idx = torch.arange(0, n, device=self.device)
            D_desc, D_mask, _ = self._gen._render_batch(D_idx)

            A_enc = self.model.encode(A_desc, A_mask)
            B_enc = self.model.encode(B_desc, B_mask)
            C_enc = self.model.encode(C_desc, C_mask)
            D_enc = self.model.encode(D_desc, D_mask)

            A_z, B_z, C_z, D_z = A_enc["z_q"], B_enc["z_q"], C_enc["z_q"], D_enc["z_q"]

            # 1) Relation subspace similarity (projected difference)
            r_ab = self.model.analogy.rel_vec(A_z, B_z)               # (1,P)
            r_cd = self.model.analogy.rel_vec(C_z.repeat(n, 1), D_z)   # (N,P)
            sim_rel = torch.einsum("ip,np->in", F.normalize(r_ab, -1), F.normalize(r_cd, -1)).squeeze(0)

            # 2) Translational consistency in code space: D ≈ C + (B - A)
            delta = B_z - A_z
            target = F.normalize(C_z + delta, dim=-1)
            sim_trans = torch.einsum("dp,np->n", target, F.normalize(D_z, dim=-1))

            # Combine with weights; sharpen via temperature
            w_rel, w_trans = 0.6, 0.4
            sim = w_rel * (self.model.analogy.scale.detach().clamp_min(1.0) * sim_rel) + w_trans * sim_trans
            # Convert to probabilities to reflect confidence (low temperature)
            temp = 0.05
            prob = torch.softmax(sim / temp, dim=-1)
            topk_score, topk_idx = torch.topk(prob, k=min(top_k, n))
            pred = int(topk_idx[0].item())

        # Build a small score dict for the top-k predictions
        scores = {str(int(D_idx[int(i)].item())): float(s) for i, s in zip(topk_idx.tolist(), topk_score.tolist())}

        # Populate a compact concept view using VQ code indices
        concepts: Dict[str, Any] = {
            "A": {"value": a, "code": int(A_enc.get("indices")[0].item()) if "indices" in A_enc else None},
            "B": {"value": b, "code": int(B_enc.get("indices")[0].item()) if "indices" in B_enc else None},
            "C": {"value": c, "code": int(C_enc.get("indices")[0].item()) if "indices" in C_enc else None},
        }
        # Include codes for the predicted and top-k candidates
        if "indices" in D_enc:
            d_codes = D_enc["indices"]  # (N,)
            pred_code = int(d_codes[pred].item())
            concepts["pred"] = {"value": pred, "code": pred_code}
            topk_concepts: Dict[str, Any] = {}
            for i, s in zip(topk_idx.tolist(), topk_score.tolist()):
                idx_i = int(D_idx[int(i)].item())
                topk_concepts[str(idx_i)] = {"code": int(d_codes[int(i)].item()), "score": float(s)}
            concepts["topk_codes"] = topk_concepts

        return AnalogyResult(prediction=pred, scores=scores, concepts=concepts)

    def explain(self, x: Any) -> Dict[str, Any]:
        # Placeholder: return nearest codes, prototypes, etc.
        return {}

    def record_feedback(self, A: Any, B: Any, C: Any, D: Any, ok: bool) -> None:
        # Placeholder: append to a small buffer for replay learning.
        return None
