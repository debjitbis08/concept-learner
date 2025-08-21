from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.matmul(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) * 0.5


class EWC:
    """
    Lightweight EWC-like stability penalty toward a snapshot of parameters.
    Uses L2 toward snapshot weights as a proxy (no Fisher).
    """

    def __init__(self, params: Iterable[nn.Parameter], weight: float = 0.0):
        self.weight = weight
        self.snapshot = [p.detach().clone() for p in params if p.requires_grad]
        self.param_refs = [p for p in params if p.requires_grad]

    def penalty(self) -> torch.Tensor:
        # Handle empty param list robustly
        device = self.param_refs[0].device if self.param_refs else "cpu"
        if self.weight == 0.0 or not self.param_refs:
            return torch.tensor(0.0, device=device)
        losses = []
        for p, s in zip(self.param_refs, self.snapshot):
            losses.append(torch.sum((p - s.to(p.device)) ** 2))
        return self.weight * torch.stack(losses).sum()

    def update(self, params: Iterable[nn.Parameter]):
        self.snapshot = [p.detach().clone() for p in params if p.requires_grad]
        self.param_refs = [p for p in params if p.requires_grad]


def code_usage_entropy(indices: torch.Tensor, num_codes: int) -> torch.Tensor:
    if indices.numel() == 0:
        return torch.tensor(0.0, device=indices.device)
    hist = torch.bincount(indices, minlength=num_codes).float()
    p = hist / (hist.sum() + 1e-8)
    ent = -(p * torch.log(p + 1e-8)).sum()
    # Encourage low entropy (few codes used)
    return ent
