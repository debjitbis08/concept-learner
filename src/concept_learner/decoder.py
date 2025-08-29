import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedDecoder(nn.Module):
    """Maps final hidden states (and optional scalar stream) to task outputs."""

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.token_head = nn.Linear(d_model, num_classes)
        self.seq_head = nn.Linear(d_model, num_classes)
        # optional head to read number from scalar value stream 'val'
        self.num_head = nn.Linear(1, num_classes)
        # initialize as no-op so legacy behavior is preserved initially
        nn.init.zeros_(self.num_head.weight)
        nn.init.zeros_(self.num_head.bias)
        # --- Fast-weights (Hebbian) for seq_head ---
        self.fast_enable = True
        self.fast_eta = 0.05
        self.fast_decay = 0.90
        self.fast_norm_max = 1.0
        self.register_buffer("fast_seq_W", torch.zeros(num_classes, d_model), persistent=False)
        # caches from last forward
        self._last_h_pool: torch.Tensor | None = None
        self._last_val: torch.Tensor | None = None
        self._last_logits_seq: torch.Tensor | None = None

    @torch.no_grad()
    def reset_fast(self):
        if hasattr(self, "fast_seq_W") and isinstance(self.fast_seq_W, torch.Tensor):
            self.fast_seq_W.zero_()

    @torch.no_grad()
    def hebbian_update(self, y: torch.Tensor):
        """One-shot Hebbian update: W_fast += eta * (y_onehot - p) @ h_pool.

        - y: (B,) long labels
        Uses caches from the last forward: logits_seq (for p) and h_pool.
        Clamps Frobenius norm of fast weights to fast_norm_max.
        """
        if not self.fast_enable:
            return
        if self._last_h_pool is None or self._last_logits_seq is None:
            return
        B = y.size(0)
        if B <= 0:
            return
        device = self._last_logits_seq.device
        C = self._last_logits_seq.size(-1)
        # probabilities from last logits
        p = F.softmax(self._last_logits_seq.detach(), dim=-1)  # (B,C)
        # one-hot targets
        y = y.detach().to(device)
        y = y.clamp_min(0).clamp_max(C - 1)
        y_onehot = torch.zeros_like(p)
        y_onehot.scatter_(1, y.view(-1, 1), 1.0)
        delta = (y_onehot - p)  # (B,C)
        h = self._last_h_pool.detach()  # (B,D)
        dW = torch.einsum("bc,bd->cd", delta, h) / float(max(1, B))
        self.fast_seq_W.add_(self.fast_eta * dW)
        # clamp Frobenius norm
        norm = torch.norm(self.fast_seq_W, p=2)
        if norm > self.fast_norm_max:
            self.fast_seq_W.mul_(self.fast_norm_max / (norm + 1e-8))

    @torch.no_grad()
    def recompute_seq_logits(self) -> torch.Tensor | None:
        """Recompute sequence logits with current fast weights using last cached h_pool/val."""
        if self._last_h_pool is None:
            return None
        W_eff = self.seq_head.weight + (self.fast_seq_W if self.fast_enable else 0.0)
        logits_seq = F.linear(self._last_h_pool, W_eff, self.seq_head.bias)
        if self._last_val is not None:
            logits_seq = logits_seq + self.num_head(self._last_val)
        # Numerical safety: replace NaN/Inf and clamp range to keep CE stable
        logits_seq = torch.nan_to_num(logits_seq, nan=0.0, posinf=20.0, neginf=-20.0)
        logits_seq = logits_seq.clamp(min=-20.0, max=20.0)
        self._last_logits_seq = logits_seq
        return logits_seq

    def forward(self, H: torch.Tensor, mask: torch.Tensor, val: torch.Tensor | None = None):
        logits_tok = self.token_head(H)  # (B,T,C)
        mask_f = mask.float().unsqueeze(-1)
        H_sum = (H * mask_f).sum(dim=1)
        T_eff = mask_f.sum(dim=1).clamp_min(1.0)
        h_pool = H_sum / T_eff
        # decay fast weights per call (episode step)
        if self.fast_enable and isinstance(self.fast_seq_W, torch.Tensor):
            self.fast_seq_W.mul_(self.fast_decay)
        # effective weights
        W_eff = self.seq_head.weight + (self.fast_seq_W if self.fast_enable else 0.0)
        logits_seq = F.linear(h_pool, W_eff, self.seq_head.bias)  # (B,C)
        if val is not None:
            logits_seq = logits_seq + self.num_head(val)
        # Numerical safety: replace NaN/Inf and clamp range to keep CE stable
        logits_seq = torch.nan_to_num(logits_seq, nan=0.0, posinf=20.0, neginf=-20.0)
        logits_seq = logits_seq.clamp(min=-20.0, max=20.0)
        # cache for optional fast-weight update
        self._last_h_pool = h_pool
        self._last_val = val
        self._last_logits_seq = logits_seq
        return logits_tok, logits_seq
