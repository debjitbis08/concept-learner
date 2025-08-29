from __future__ import annotations

import torch
import torch.nn.functional as F


class ProtoRouter(torch.nn.Module):
    """
    Per-relation prototypical routing over normalized features.

    Maintains EMA prototypes mu_r for r in {0..R-1} over h~ features and
    adds a symmetric bias w * cos(h~, mu_r) to YES/NO logits.

    Features can be taken from the typed-state space (Reasoner pooled state)
    or from the encoder pooled state (h). Optionally gate application until
    at least `warmup` examples have been seen per relation to avoid collapse.
    """

    def __init__(
        self,
        num_relations: int,
        feat_dim: int,
        *,
        beta: float = 0.98,
        weight: float = 0.2,
        weight_end: float | None = 0.1,
        decay_steps: int = 0,
        space: str = "typed",
        warmup: int = 50,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.num_rel = int(num_relations)
        self.feat_dim = int(feat_dim)
        self.beta = float(beta)
        self.space = str(space)
        self.warmup = int(warmup)
        self.register_buffer("mu", torch.zeros(self.num_rel, self.feat_dim))
        self.register_buffer("cnt", torch.zeros(self.num_rel, dtype=torch.long))
        self.weight = float(weight)
        self.weight_end = float(weight_end if weight_end is not None else weight)
        self.decay_steps = int(decay_steps)
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))
        if device is not None:
            self.to(device)

    def extra_repr(self) -> str:
        return (
            f"R={self.num_rel} d={self.feat_dim} beta={self.beta} "
            f"w={self.weight}->{self.weight_end} decay={self.decay_steps} "
            f"space={self.space} warmup={self.warmup}"
        )

    def _current_weight(self) -> float:
        if self.decay_steps and self.weight_end != self.weight and self.decay_steps > 0:
            # linear decay to weight_end over decay_steps, then clamp
            s = int(self._step.item())
            t = min(max(s, 0), self.decay_steps)
            w = self.weight + (self.weight_end - self.weight) * (t / float(self.decay_steps))
            return float(w)
        return float(self.weight)

    @staticmethod
    def build_feature(
        h_pooled: torch.Tensor | None,
        s_typed: torch.Tensor | None,
        space: str,
    ) -> torch.Tensor:
        """Return normalized features according to `space`.

        - space == 'typed': use s_typed
        - space == 'pooled': use h_pooled

        Returns L2-normalized vector.
        """
        if space == "typed":
            assert s_typed is not None, "s_typed required for space='typed'"
            x = s_typed
        elif space == "pooled":
            assert h_pooled is not None, "h_pooled required for space='pooled'"
            x = h_pooled
        else:
            raise ValueError(f"Unsupported space='{space}'")
        return F.normalize(x, dim=-1, eps=1e-6)

    @torch.no_grad()
    def update(self, feats: torch.Tensor, rel_ids: torch.Tensor) -> None:
        """EMA-update prototypes for the provided batch.

        feats: (B, D) normalized
        rel_ids: (B,) int, expected in [0, num_rel)
        """
        if feats.numel() == 0:
            return
        assert feats.dim() == 2 and feats.size(-1) == self.feat_dim
        assert rel_ids.dim() == 1 and rel_ids.size(0) == feats.size(0)
        # Only update valid relation ids
        valid = (rel_ids >= 0) & (rel_ids < self.num_rel)
        if not valid.any():
            return
        r = rel_ids[valid].to(torch.long)
        f = feats[valid]
        # per relation gather and EMA
        for rid in r.unique().tolist():
            mask = (r == int(rid))
            if not mask.any():
                continue
            f_mean = F.normalize(f[mask].mean(dim=0, keepdim=True), dim=-1, eps=1e-6)  # (1,D)
            mu_old = self.mu[rid : rid + 1]
            self.mu[rid : rid + 1] = self.beta * mu_old + (1.0 - self.beta) * f_mean
            self.cnt[rid] = self.cnt[rid] + mask.sum().to(self.cnt.dtype)

    def apply_bias(
        self,
        logits_seq: torch.Tensor,
        feats: torch.Tensor,
        rel_ids: torch.Tensor,
        yes_idx: int,
        no_idx: int,
        *,
        train_mode: bool = False,
    ) -> torch.Tensor:
        """Return logits with proto bias applied on rows with valid relation ids.

        - Symmetric bias: YES += w*cos; NO += -w*cos
        - Gate until count[r] >= warmup
        - Optionally update prototypes if train_mode=True
        """
        if feats.numel() == 0 or logits_seq.numel() == 0:
            return logits_seq
        B = feats.size(0)
        assert logits_seq.size(0) == B
        device = logits_seq.device
        rel_ids = rel_ids.to(device)
        # Optionally update prototypes
        if train_mode:
            self.update(feats.detach(), rel_ids.detach())
            self._step = self._step + 1
        # mask applicable rows: valid relation ids AND warmed up
        valid = (rel_ids >= 0) & (rel_ids < self.num_rel)
        if not valid.any():
            return logits_seq
        # compute cosine per row
        mu_sel = self.mu[rel_ids.clamp(0, self.num_rel - 1)]  # (B,D)
        cos = F.cosine_similarity(feats, mu_sel, dim=-1)  # (B,)
        # gate by warmup
        cnt_sel = self.cnt[rel_ids.clamp(0, self.num_rel - 1)]  # (B,)
        gate = (cnt_sel >= self.warmup).to(logits_seq.dtype)
        bias = (self._current_weight()) * cos * gate  # (B,)
        # Add symmetric bias to YES/NO
        out = logits_seq
        out[:, yes_idx] = out[:, yes_idx] + bias
        out[:, no_idx] = out[:, no_idx] - bias
        return out

