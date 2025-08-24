import torch
import torch.nn as nn


class UnifiedDecoder(nn.Module):
    """Stub: maps final hidden states to task outputs."""

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.token_head = nn.Linear(d_model, num_classes)
        self.seq_head = nn.Linear(d_model, num_classes)

    def forward(self, H: torch.Tensor, mask: torch.Tensor):
        logits_tok = self.token_head(H)  # (B,T,C)
        mask_f = mask.float().unsqueeze(-1)
        H_sum = (H * mask_f).sum(dim=1)
        T_eff = mask_f.sum(dim=1).clamp_min(1.0)
        h_pool = H_sum / T_eff
        logits_seq = self.seq_head(h_pool)  # (B,C)
        return logits_tok, logits_seq
