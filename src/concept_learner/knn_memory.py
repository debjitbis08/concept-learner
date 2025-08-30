import torch
import torch.nn as nn
import torch.nn.functional as F


class KNNSoftClassifier(nn.Module):
    """
    Lightweight kNN posterior over a rolling memory of (feature, label).

    - Similarity: cosine; temperature tau scales similarity before softmax.
    - Posterior: class-prob as weighted vote of k nearest by softmax(sim/tau).
    - Mixing alpha is managed externally by caller.
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        capacity: int = 10_000,
        k: int = 32,
        tau: float = 0.2,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.C = int(num_classes)
        self.D = int(feat_dim)
        self.N = int(capacity)
        self.k = int(k)
        self.tau = float(tau)
        self.register_buffer("feats", torch.zeros(self.N, self.D), persistent=False)
        self.register_buffer("labels", torch.full((self.N,), -1, dtype=torch.long), persistent=False)
        self.ptr = 0
        self.full = False
        self.device_used = device

    @torch.no_grad()
    def add(self, x: torch.Tensor, y: torch.Tensor):
        if x is None or y is None or x.numel() == 0:
            return
        x = x.detach().to(self.feats.device)
        y = y.detach().to(self.labels.device).long()
        B = x.size(0)
        if x.size(1) != self.D:
            # project or truncate if mismatch (simple truncate/pad)
            if x.size(1) > self.D:
                x = x[:, : self.D]
            else:
                pad = torch.zeros(B, self.D - x.size(1), device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=-1)
        for i in range(B):
            self.feats[self.ptr] = x[i]
            self.labels[self.ptr] = y[i]
            self.ptr = (self.ptr + 1) % self.N
            if self.ptr == 0:
                self.full = True

    @torch.no_grad()
    def posterior(self, q: torch.Tensor) -> torch.Tensor:
        """Return class posterior for each query in q: (B,C)."""
        if q is None or q.numel() == 0:
            return torch.zeros(0, self.C, device=self.feats.device)
        # select valid memory entries (labels within [0, C))
        valid = (self.labels >= 0) & (self.labels < self.C)
        if valid.sum().item() == 0:
            return torch.zeros(q.size(0), self.C, device=self.feats.device)
        M = self.feats[valid]  # (M,D)
        Y = self.labels[valid]  # (M,)
        q = q.to(M.device)
        # cosine similarities
        qn = F.normalize(q, dim=-1)
        Mn = F.normalize(M, dim=-1)
        S = torch.matmul(qn, Mn.t())  # (B,M)
        # top-k
        k = min(self.k, S.size(1))
        vals, idx = torch.topk(S, k=k, dim=-1)
        # weights via temperature softmax
        W = torch.softmax(vals / max(1e-6, self.tau), dim=-1)  # (B,k)
        # accumulate class votes
        B = q.size(0)
        probs = torch.zeros(B, self.C, device=M.device)
        for bi in range(B):
            nbr_y = Y[idx[bi]]  # (k,)
            probs[bi].index_add_(0, nbr_y, W[bi])
        # normalize
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs

    @torch.no_grad()
    def snapshot(self, model_hash: str, metric: str = "cosine", normalized: bool = True) -> dict:
        """Build a compact state for persistence. Stores only valid entries as float16."""
        valid = self.labels >= 0
        M = int(valid.sum().item())
        feats = self.feats[valid][:M].detach().to("cpu").to(torch.float16)
        labels = self.labels[valid][:M].detach().to("cpu")
        return {
            "model_hash": str(model_hash),
            "feat_dim": int(self.D),
            "metric": str(metric),
            "normalized": bool(normalized),
            "keys": feats,
            "labels": labels,
            "capacity": int(self.N),
            "k": int(self.k),
            "tau": float(self.tau),
        }

    @torch.no_grad()
    def load_snapshot(self, state: dict, expected_hash: str | None = None) -> bool:
        """Load snapshot if model hash matches; returns True on success."""
        if not isinstance(state, dict):
            return False
        h = str(state.get("model_hash", ""))
        if expected_hash is not None and h != str(expected_hash):
            return False
        keys = state.get("keys")
        labels = state.get("labels")
        if not isinstance(keys, torch.Tensor) or not isinstance(labels, torch.Tensor):
            return False
        # reset
        self.feats.zero_()
        self.labels.fill_(-1)
        self.ptr = 0
        self.full = False
        # copy (truncate if needed)
        K = min(keys.size(0), self.N)
        D = min(keys.size(1), self.D)
        self.feats[:K, :D].copy_(keys[:K, :D].to(self.feats.dtype))
        self.labels[:K].copy_(labels[:K].to(self.labels.dtype))
        self.ptr = int(K % self.N)
        self.full = (K == self.N)
        # adopt params if present
        try:
            self.k = int(state.get("k", self.k))
            self.tau = float(state.get("tau", self.tau))
        except Exception:
            pass
        return True
