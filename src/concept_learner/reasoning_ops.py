from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TypedState:
    """Typed state used by operators: H (items), mask in [0,1], val (scalar), bool (scalar)."""

    H: torch.Tensor  # (B, T, d)
    mask: torch.Tensor  # (B, T) float 0..1
    val: torch.Tensor  # (B, 1) float
    boolean: torch.Tensor  # (B, 1) float 0..1

    @staticmethod
    def from_H_mask(H: torch.Tensor, mask_int: torch.Tensor) -> "TypedState":
        B, T, d = H.shape
        mask = mask_int.float().clamp(0, 1)
        val = H.new_zeros(B, 1)
        boolean = H.new_zeros(B, 1)
        return TypedState(H=H, mask=mask, val=val, boolean=boolean)


class OpBase(nn.Module):
    """Operator returns an UPDATED (mask, val, boolean). H is never mutated here."""

    name: str

    def forward(
        self, state: TypedState, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


# ------------------------------ FILTER(p) --------------------------------


class OpFilterMLP(OpBase):
    """
    Per-token scorer in [0,1], monotone mask update: mask' = min(mask, score).
    Predicate p is implicit via the learned scorer over tokens (optionally conditioned on z).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.name = "filter"
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, state: TypedState, z: torch.Tensor):
        B, T, d = state.H.shape
        z_rep = z.unsqueeze(1).expand(B, T, z.size(-1))
        logits = self.scorer(torch.cat([state.H, z_rep], dim=-1)).squeeze(-1)  # (B,T)
        score = torch.sigmoid(logits)
        new_mask = torch.minimum(state.mask, score)  # idempotent & monotone
        return new_mask, state.val, state.boolean


# ------------------------------ COUNT() ----------------------------------


class OpCount(OpBase):
    """
    DeepSets aggregator VAL = sum_i phi(item_i) * MASK_i
    phi defaults to '1' (pure count). You can swap an MLP if you want weighted counts.
    """

    def __init__(self, d_model: int, learn_phi: bool = False):
        super().__init__()
        self.name = "count"
        if learn_phi:
            self.phi = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
            )
        else:
            self.phi = None  # constant 1

    def forward(self, state: TypedState, z: torch.Tensor):
        B, T, d = state.H.shape
        if self.phi is None:
            contrib = state.H.new_ones(B, T)  # 1 per item
        else:
            contrib = self.phi(state.H).squeeze(-1)  # (B,T)
        val = (contrib * state.mask).sum(dim=1, keepdim=True)  # (B,1)
        return state.mask, val, state.boolean


# ------------------------------ ADD(k) -----------------------------------


class OpAddConst(OpBase):
    """VAL' = VAL + α * k ; α is learned ~1.0 (clipped positive with softplus)."""

    def __init__(self, k: int):
        super().__init__()
        self.name = f"add_{k:+d}"
        self.k = float(k)
        self.alpha_raw = nn.Parameter(
            torch.tensor(0.0)
        )  # softplus -> ~1.0 after training

    def forward(self, state: TypedState, z: torch.Tensor):
        alpha = F.softplus(self.alpha_raw) + 1.0e-6
        val = state.val + alpha * self.k
        return state.mask, val, state.boolean


# ------------------------------ COMPARE(op,k) ----------------------------


class OpCompare(OpBase):
    """
    BOOL' = σ(a * f(VAL, k)), with a>=0.
      - op='gt' : f = (VAL - k)
      - op='lt' : f = (k - VAL)
      - op='eq' : f = -|VAL - k|   (soft equality)
    """

    def __init__(self, k: float, op: Literal["gt", "lt", "eq"] = "gt"):
        super().__init__()
        self.name = f"cmp_{op}_{k}"
        self.k = float(k)
        self.op = op
        self.a_raw = nn.Parameter(torch.zeros(1))

    def forward(self, state: TypedState, z: torch.Tensor):
        a = F.softplus(self.a_raw) + 1e-6
        diff = state.val - self.k
        if self.op == "gt":
            score = diff
        elif self.op == "lt":
            score = -diff
        else:  # eq
            score = -diff.abs()
        boolean = torch.sigmoid(a * score)
        return state.mask, state.val, boolean
