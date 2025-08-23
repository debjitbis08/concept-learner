from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistMultHead(nn.Module):
    """
    DistMult-style bilinear scoring over concept embeddings.
    """

    def __init__(self, concept_dim: int, num_relations: int):
        super().__init__()
        self.rel = nn.Parameter(torch.randn(num_relations, concept_dim) * 0.02)
        self.scale = nn.Parameter(torch.tensor(5.0))

    def score(self, s: torch.Tensor, r: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        # s, o: (B, D); r: (B,)
        w = self.rel[r]  # (B, D)
        return self.scale * torch.sum(s * w * o, dim=-1)

    def forward(self, s: torch.Tensor, r: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        return self.score(s, r, o)


class AnalogyProjector(nn.Module):
    """
    Learns a relation subspace; encourages analogical structure via projection.
    """

    def __init__(self, dim: int, proj_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(dim, proj_dim, bias=False)
        # Learnable logit scale (temperature inverse)
        self.scale = nn.Parameter(torch.tensor(5.0))

    def rel_vec(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Normalize inputs to stabilize differences
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        return self.proj(a_n - b_n)

    def analogy_loss(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        temp: float = 0.1,
        w_rel: float = 0.7,
        w_trans: float = 0.3,
    ) -> torch.Tensor:
        # In-batch negatives: compute similarity of r_AB to r_CD for all D in batch
        r1 = self.rel_vec(A, B)  # (B, P)
        r2 = self.rel_vec(C, D)  # (B, P)
        sim = torch.matmul(F.normalize(r1, dim=-1), F.normalize(r2, dim=-1).t())
        logits = (self.scale * sim) / temp
        labels = torch.arange(A.size(0), device=A.device)
        ce = F.cross_entropy(logits, labels)

        # Translational consistency in concept space: D ≈ C + (B - A)
        delta = B - A
        target = F.normalize(C + delta, dim=-1)
        pred = F.normalize(D, dim=-1)
        # Use 1 - cosine similarity as loss
        trans = 1.0 - torch.sum(target * pred, dim=-1).mean()
        return w_rel * ce + w_trans * trans
