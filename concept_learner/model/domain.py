from __future__ import annotations

import torch
import torch.nn as nn


class DomainAdapter(nn.Module):
    """
    Lightweight FiLM-style adapter that conditions a representation on a domain id.
    h' = (1 + gamma_d) * h + beta_d, where [gamma_d, beta_d] = MLP(Emb(domain)).
    """

    def __init__(self, num_domains: int, hidden_dim: int, adapter_rank: int = 16):
        super().__init__()
        self.emb = nn.Embedding(num_domains, hidden_dim)
        r = max(1, int(adapter_rank))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, r),
            nn.ReLU(),
            nn.Linear(r, 2 * hidden_dim),
        )

    def forward(self, h: torch.Tensor, domain: torch.Tensor | int | None) -> torch.Tensor:
        if domain is None:
            return h
        if isinstance(domain, int):
            d = torch.tensor([domain], dtype=torch.long, device=h.device).expand(h.size(0))
        else:
            d = domain.to(h.device)
        gb = self.mlp(self.emb(d))  # (B, 2H)
        H = h.size(-1)
        gamma, beta = gb[:, :H], gb[:, H:]
        return h * (1.0 + gamma) + beta


class DomainToken(nn.Module):
    """
    Simple domain token/embedding added to the pooled representation.
    This is an additive bias per domain; complements the FiLM adapter.
    """

    def __init__(self, num_domains: int, hidden_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_domains, hidden_dim)

    def forward(self, h: torch.Tensor, domain: torch.Tensor | int | None) -> torch.Tensor:
        if domain is None:
            return h
        if isinstance(domain, int):
            d = torch.tensor([domain], dtype=torch.long, device=h.device).expand(h.size(0))
        else:
            d = domain.to(h.device)
        return h + self.emb(d)
