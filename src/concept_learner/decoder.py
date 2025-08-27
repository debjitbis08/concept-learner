import torch
import torch.nn as nn


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

    def forward(self, H: torch.Tensor, mask: torch.Tensor, val: torch.Tensor | None = None):
        logits_tok = self.token_head(H)  # (B,T,C)
        mask_f = mask.float().unsqueeze(-1)
        H_sum = (H * mask_f).sum(dim=1)
        T_eff = mask_f.sum(dim=1).clamp_min(1.0)
        h_pool = H_sum / T_eff
        logits_seq = self.seq_head(h_pool)  # (B,C)
        if val is not None:
            logits_seq = logits_seq + self.num_head(val)
        return logits_tok, logits_seq
