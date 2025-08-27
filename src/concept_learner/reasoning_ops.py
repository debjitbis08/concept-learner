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
    # Optional span pointer [start,end) as scalar indices; -1 means unset
    ptr_start: torch.Tensor | None = None  # (B,1)
    ptr_end: torch.Tensor | None = None  # (B,1)
    # Small scratch env/stack of size K (e.g., K=4)
    env: torch.Tensor | None = None  # (B,K)

    @staticmethod
    def from_H_mask(H: torch.Tensor, mask_int: torch.Tensor) -> "TypedState":
        B, T, d = H.shape
        mask = mask_int.float().clamp(0, 1)
        val = H.new_zeros(B, 1)
        boolean = H.new_zeros(B, 1)
        ptr_s = H.new_full((B, 1), -1.0)
        ptr_e = H.new_full((B, 1), -1.0)
        env = H.new_zeros(B, 4)
        return TypedState(H=H, mask=mask, val=val, boolean=boolean, ptr_start=ptr_s, ptr_end=ptr_e, env=env)


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


class OpFilterSpan(OpBase):
    """Apply filter within a pointer span [start,end); fallback to full mask if unset."""

    def __init__(self):
        super().__init__()
        self.name = "filter_span"

    def forward(self, state: TypedState, z: torch.Tensor):
        m = state.mask
        B, T = m.shape
        if state.ptr_start is None or state.ptr_end is None:
            return m, state.val, state.boolean
        s = state.ptr_start.clamp_min(0).long().view(-1)
        e = state.ptr_end.clamp_min(0).long().view(-1)
        span_mask = m.new_zeros(B, T)
        for i in range(B):
            si = int(s[i].item()) if s[i] >= 0 else 0
            ei = int(e[i].item()) if e[i] >= 0 else T
            si = max(0, min(T, si)); ei = max(si, min(T, ei))
            span_mask[i, si:ei] = 1.0
        new_mask = m * span_mask
        return new_mask, state.val, state.boolean


class OpCountSpan(OpBase):
    """Count only within the current pointer span if set."""

    def __init__(self):
        super().__init__()
        self.name = "count_span"

    def forward(self, state: TypedState, z: torch.Tensor):
        B, T = state.mask.shape
        if state.ptr_start is None or state.ptr_end is None:
            val = (state.mask).sum(dim=1, keepdim=True)
            return state.mask, val, state.boolean
        s = state.ptr_start.clamp_min(0).long().view(-1)
        e = state.ptr_end.clamp_min(0).long().view(-1)
        vals = []
        for i in range(B):
            si = int(s[i].item()) if s[i] >= 0 else 0
            ei = int(e[i].item()) if e[i] >= 0 else T
            si = max(0, min(T, si)); ei = max(si, min(T, ei))
            vals.append(state.mask[i, si:ei].sum())
        val = torch.stack(vals, dim=0).unsqueeze(-1)
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


# ------------------------------ GENERIC ADD -------------------------------


class OpAdd(OpBase):
    """
    Generic add that subsumes OpAddConst and allows optional conditioning on z.

    VAL' = beta * VAL + alpha * k, with beta>=0, alpha>=0.
    - If use_z=True, k is predicted per-sample from z via a small linear head.
    - Otherwise, k is a single learned scalar parameter.
    """

    def __init__(self, d_model: int, use_z: bool = False):
        super().__init__()
        self.name = "add"
        self.use_z = use_z
        # scale factors kept positive via softplus, initialized near 0 -> softplus ~ 0
        # add tiny epsilon to avoid exact zeros
        self.beta_raw = nn.Parameter(torch.tensor(0.0))
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        if use_z:
            self.k_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.k_head.weight)
            nn.init.zeros_(self.k_head.bias)
        else:
            self.k = nn.Parameter(torch.tensor(0.0))

    def forward(self, state: TypedState, z: torch.Tensor):
        beta = F.softplus(self.beta_raw) + 1e-6
        alpha = F.softplus(self.alpha_raw) + 1e-6
        if self.use_z:
            k = self.k_head(z)  # (B,1)
            # cache for optional regularization
            self._last_k = k
        else:
            # broadcast scalar parameter to (B,1)
            B = state.val.size(0)
            k = self.k.expand(B, 1)
        val = beta * state.val + alpha * k
        return state.mask, val, state.boolean

    def regularization(self, z: torch.Tensor | None = None) -> torch.Tensor:
        """Soft penalty to discourage fractional drift of k.

        Encourages k toward {-1, +1, +10} by minimizing min(|k-1|, |k+1|, |k-10|).
        Returns a scalar tensor.
        """
        if self.use_z:
            if z is not None:
                k = self.k_head(z)
            else:
                k = getattr(self, "_last_k", None)
                if k is None:
                    return torch.tensor(0.0, device=self.beta_raw.device)
        else:
            k = self.k
        targets = torch.stack([(k - 1.0).abs(), (k + 1.0).abs(), (k - 10.0).abs()], dim=-1)
        pen = targets.min(dim=-1).values
        return pen.mean()


class OpAddInt(OpBase):
    """Integer-locked add: project k_hat onto {-10, -1, +1, +10} via soft weights.

    k_hat is predicted from z (or learned scalar). We form k_proj = sum w_i * cand_i,
    where w = softmax(affine([k_hat, z]))
    """

    def __init__(self, d_model: int, use_z: bool = True):
        super().__init__()
        self.name = "add_int"
        self.use_z = use_z
        self.candidates = torch.tensor([-10.0, -1.0, 1.0, 10.0])
        self.k_head = nn.Linear(d_model, 1) if use_z else None
        self.mix = nn.Linear(d_model + 1, 4)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        if self.k_head is not None:
            nn.init.zeros_(self.k_head.weight)
            nn.init.zeros_(self.k_head.bias)

    def forward(self, state: TypedState, z: torch.Tensor):
        B = state.val.size(0)
        if self.use_z and self.k_head is not None:
            k_hat = self.k_head(z)  # (B,1)
        else:
            k_hat = state.val.new_zeros(B, 1)
        mix_in = torch.cat([z, k_hat], dim=-1)
        w = torch.softmax(self.mix(mix_in), dim=-1)  # (B,4)
        c = self.candidates.to(z.device).view(1, 4)
        k_proj = (w * c).sum(dim=-1, keepdim=True)
        val = state.val + k_proj
        return state.mask, val, state.boolean


# ------------------------------ GENERIC COMPARE --------------------------


class OpCompareGeneric(OpBase):
    """
    Generic comparator unifying gt/lt/eq via a small non-negative feature basis.

    BOOL' = σ( w1*(VAL-k) + w2*(-(VAL-k)) + w3*(-|VAL-k|) + b ), with w_i >= 0.
    - If use_z=True, k is predicted from z; else it's a single learned scalar.
    """

    def __init__(self, d_model: int, use_z: bool = False):
        super().__init__()
        self.name = "cmp_generic"
        self.use_z = use_z
        self.w_raw = nn.Parameter(torch.zeros(3))
        self.b = nn.Parameter(torch.zeros(1))
        if use_z:
            self.k_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.k_head.weight)
            nn.init.zeros_(self.k_head.bias)
        else:
            self.k = nn.Parameter(torch.tensor(0.0))

    def forward(self, state: TypedState, z: torch.Tensor):
        if self.use_z:
            k = self.k_head(z)  # (B,1)
        else:
            B = state.val.size(0)
            k = self.k.expand(B, 1)
        diff = state.val - k  # (B,1)
        # features: [diff, -diff, -|diff|]
        feats = torch.stack([diff, -diff, -diff.abs()], dim=-1)  # (B,1,3)
        w = F.softplus(self.w_raw) + 1e-6  # (3,)
        score = (feats * w).sum(dim=-1) + self.b  # (B,1)
        boolean = torch.sigmoid(score)
        return state.mask, state.val, boolean
