from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Literal, List, Optional
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


# ------------------------------ SUBSTITUTE MASK --------------------------


class OpSubstituteMask(OpBase):
    """
    Rewrite the active mask using a learned scorer over tokens.

    Unlike OpFilterMLP which is monotone (mask' = min(mask, score)), this operator
    directly proposes a fresh mask based on H and z: mask' = σ(MLP([H; z]))
    This is useful as a primitive for substitution/branching in learned programs.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.name = "substitute_mask"
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, state: TypedState, z: torch.Tensor):
        B, T, d = state.H.shape
        z_rep = z.unsqueeze(1).expand(B, T, z.size(-1))
        logits = self.scorer(torch.cat([state.H, z_rep], dim=-1)).squeeze(-1)
        new_mask = torch.sigmoid(logits)
        return new_mask, state.val, state.boolean


# ------------------------------ ENV: LET / LOAD --------------------------


class _EnvOpBase(OpBase):
    """Mixin-like base that exposes a side-channel env_update hook.

    This keeps OpBase.forward signature unchanged while allowing Reasoner to
    query per-operator proposed env updates and mix them using the controller's
    action distribution.
    """

    def env_update(self, state: TypedState, z: torch.Tensor) -> torch.Tensor | None:
        return None


class OpLet(_EnvOpBase):
    """
    Write the current scalar value (state.val) into a small register file 'env'.

    - A soft address w = softmax(W z) selects which register(s) to update.
    - A write strength α = σ(a(z)) gates between keep vs write.
    - Update rule: env' = env * (1 - α w) + (α w) * val
    Forward pass leaves (mask, val, boolean) unchanged and relies on env_update
    to communicate the new env to the Reasoner.
    """

    def __init__(self, d_model: int, num_regs: int = 4):
        super().__init__()
        self.name = "let"
        self.num_regs = int(num_regs)
        self.addr = nn.Linear(d_model, num_regs)
        self.strength = nn.Linear(d_model, 1)
        nn.init.zeros_(self.addr.weight)
        nn.init.zeros_(self.addr.bias)
        nn.init.zeros_(self.strength.weight)
        nn.init.zeros_(self.strength.bias)

    def forward(self, state: TypedState, z: torch.Tensor):
        # No change to visible streams; env is handled by env_update
        return state.mask, state.val, state.boolean

    def env_update(self, state: TypedState, z: torch.Tensor) -> torch.Tensor | None:
        if state.env is None:
            # lazily create if missing
            B = state.H.size(0)
            env = state.H.new_zeros(B, self.num_regs)
        else:
            env = state.env
            if env.size(-1) != self.num_regs:
                # project or pad to expected width
                if env.size(-1) > self.num_regs:
                    env = env[..., : self.num_regs]
                else:
                    pad = env.new_zeros(env.size(0), self.num_regs - env.size(-1))
                    env = torch.cat([env, pad], dim=-1)
        w = torch.softmax(self.addr(z), dim=-1)  # (B,K)
        alpha = torch.sigmoid(self.strength(z))  # (B,1)
        delta = (w * alpha) * state.val  # (B,K)
        keep = 1.0 - (w * alpha)
        new_env = env * keep + delta
        return new_env


class OpLoad(_EnvOpBase):
    """
    Read from env into the scalar stream.

    - Soft address w = softmax(W z) reads a convex combination r = w · env.
    - Blend with current val: val' = β * val + α * r, with α,β >= 0.
    """

    def __init__(self, d_model: int, num_regs: int = 4):
        super().__init__()
        self.name = "load"
        self.num_regs = int(num_regs)
        self.addr = nn.Linear(d_model, num_regs)
        self.alpha_raw = nn.Parameter(torch.zeros(1))
        self.beta_raw = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.addr.weight)
        nn.init.zeros_(self.addr.bias)

    def forward(self, state: TypedState, z: torch.Tensor):
        B = state.H.size(0)
        if state.env is None:
            env = state.H.new_zeros(B, self.num_regs)
        else:
            env = state.env
            if env.size(-1) != self.num_regs:
                if env.size(-1) > self.num_regs:
                    env = env[..., : self.num_regs]
                else:
                    pad = env.new_zeros(env.size(0), self.num_regs - env.size(-1))
                    env = torch.cat([env, pad], dim=-1)
        w = torch.softmax(self.addr(z), dim=-1)  # (B,K)
        r = (w * env).sum(dim=-1, keepdim=True)  # (B,1)
        alpha = F.softplus(self.alpha_raw) + 1e-6
        beta = F.softplus(self.beta_raw) + 1e-6
        val = beta * state.val + alpha * r
        return state.mask, val, state.boolean

    def env_update(self, state: TypedState, z: torch.Tensor) -> torch.Tensor | None:
        # Load does not modify env
        return state.env


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


# ------------------------------ NAC / NALU --------------------------------


class OpNAC(OpBase):
    """
    Neural Accumulator (NAC): implements addition/subtraction via a constrained
    linear transform with no bias and no nonlinearity.

    a = W x,   W = tanh(Wh) * sigmoid(Mh)

    Inputs x are the concatenation of the running scalar value and conditioning z:
      x = [val, z]  -> shape (B, 1 + d_model)

    Output updates only the numeric stream (val'); mask and boolean are passed through.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.name = "nac"
        d_in = 1 + int(d_model)
        # parameters that produce W via tanh-sigmoid factorization
        self.Wh = nn.Parameter(torch.zeros(1, d_in))
        self.Mh = nn.Parameter(torch.zeros(1, d_in))

    def forward(self, state: TypedState, z: torch.Tensor):
        # x = concat(val, z)
        x = torch.cat([state.val, z], dim=-1)  # (B, 1 + d)
        W = torch.tanh(self.Wh) * torch.sigmoid(self.Mh)  # (1, 1+d)
        a = torch.matmul(x, W.t())  # (B,1)
        return state.mask, a, state.boolean


class OpNALU(OpBase):
    """
    Neural Arithmetic Logic Unit (NALU): blends additive (NAC) and multiplicative
    paths using a learned gate g.

        y = g * a + (1 - g) * m
        a = W_a x
        m = exp(W_m log(|x| + eps))
        g = sigmoid(G x)

    Inputs x are [val, z]. Outputs update only the numeric stream.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.name = "nalu"
        d_in = 1 + int(d_model)
        self.eps = float(eps)
        # NAC-like parameterizations for additive and multiplicative paths
        self.Wh_a = nn.Parameter(torch.zeros(1, d_in))
        self.Mh_a = nn.Parameter(torch.zeros(1, d_in))
        self.Wh_m = nn.Parameter(torch.zeros(1, d_in))
        self.Mh_m = nn.Parameter(torch.zeros(1, d_in))
        # gate from inputs
        self.G = nn.Linear(d_in, 1)
        nn.init.zeros_(self.G.weight)
        nn.init.zeros_(self.G.bias)

    def forward(self, state: TypedState, z: torch.Tensor):
        x = torch.cat([state.val, z], dim=-1)  # (B, 1 + d)
        # additive path
        W_a = torch.tanh(self.Wh_a) * torch.sigmoid(self.Mh_a)  # (1, D)
        a = torch.matmul(x, W_a.t())  # (B,1)
        # multiplicative path in log-space
        x_abs = x.abs() + self.eps
        logx = torch.log(x_abs)
        W_m = torch.tanh(self.Wh_m) * torch.sigmoid(self.Mh_m)
        m = torch.exp(torch.matmul(logx, W_m.t()))  # (B,1)
        # gate
        g = torch.sigmoid(self.G(x))  # (B,1)
        y = g * a + (1.0 - g) * m
        return state.mask, y, state.boolean


# ------------------------------ FUNCTION SLOTS ---------------------------


@dataclass
class StepSpec:
    """One micro-step within a function body.

    kind: 'primitive' -> run base_ops[idx]
          'function'  -> inline-expand slot idx (if enabled)
          'return'    -> early stop
    arg_tmpl_id: optional identifier of an argument template (index into OpFunction.templates)
    """

    kind: Literal["primitive", "function", "return", "cond"]
    idx: Optional[int] = None
    # For kind=="cond": idx is THEN slot id, idx2 is ELSE slot id
    arg_tmpl_id: Optional[int] = None
    idx2: Optional[int] = None


@dataclass
class Slot:
    id: int
    steps: List[StepSpec]
    arg_templates: List[int]  # indices into OpFunction.templates
    ret_policy: Literal["implicit", "return_head"]
    version: int = 0


class _ArgTemplate(nn.Module):
    """Tiny MLP that maps [s; z] -> args vector for an op.

    Current primitives in this repo do not consume extra args, but we keep the
    template as a hook for future ops. The returned vector can be cached/regularized
    by the caller (Reasoner) if desired.
    """

    def __init__(self, d_model: int, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, out_dim)
        )

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, z], dim=-1))


class FnCandidate:
    """Lightweight container proposed by Reasoner during sleep-time mining."""

    def __init__(
        self,
        steps: List[StepSpec],
        ret_policy: Literal["implicit", "return_head"] = "implicit",
        arg_templates: Optional[List[_ArgTemplate]] = None,
    ):
        self.steps = steps
        self.ret_policy = ret_policy
        self.arg_templates = arg_templates or []


class OpFunction(OpBase):
    """
    Runtime executor of learned function slots.
    The library (bodies, arg templates, return policy) is owned here,
    but creation/selection/pruning is orchestrated by Reasoner.
    """

    def __init__(
        self,
        d_model: int,
        base_ops: List[OpBase],
        num_slots: int = 16,
        max_body_len: int = 4,
        temperature: float = 1.0,
        use_straight_through: bool = True,
        allow_higher_order: bool = True,
        allow_self_call: bool = True,
    ):
        super().__init__()
        self.name = "function"
        self.d_model = int(d_model)
        self.base_ops = base_ops  # references to primitives
        self.num_slots = int(num_slots)
        self.max_body_len = int(max_body_len)
        self.temperature = float(temperature)
        self.use_straight_through = bool(use_straight_through)
        self.allow_higher_order = bool(allow_higher_order)
        self.allow_self_call = bool(allow_self_call)

        # Shared pool of arg templates; steps refer to indices here.
        self.templates = nn.ModuleList()

        # Return head used in Phase B (bounded recursion) mode.
        self._return_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

        # Initialize empty slots
        self._slots: List[Slot] = []
        for sid in range(self.num_slots):
            self._slots.append(
                Slot(
                    id=sid,
                    steps=[],
                    arg_templates=[],
                    ret_policy="implicit",
                    version=0,
                )
            )
        # Flat (expanded) bodies for eval-time feed-forward execution
        self._flat_bodies: List[List[StepSpec]] = [[] for _ in range(self.num_slots)]
        self._flat_versions: List[int] = [
            -1 for _ in range(self.num_slots)
        ]  # cache invalid when != slot.version

    # --- RUNTIME ---

    def forward(self, state: TypedState, z: torch.Tensor):
        """As an OpBase, forward runs slot 0 by default (useful for quick tests).

        For explicit control and to expose K distinct actions, call run_slot(k,...).
        """
        return self.run_slot(0, state, z)

    def run_slot(
        self, k: int, state: TypedState, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Phase A (macro): execute slot k's body to completion in one outer step.
        Training: allows dynamic inline expansion (program-like loop) for richer gradients.
        Eval: uses pre-flattened static bodies for feed-forward execution with a fixed unroll.
        Returns (mask', val', bool') like a primitive op.
        """
        if k < 0 or k >= self.num_slots:
            return state.mask, state.val, state.boolean

        if self.training:
            return self._run_slot_dynamic(k, state, z)
        else:
            return self._run_slot_flat(k, state, z)

    def run_slot_env(
        self, k: int, state: TypedState, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run slot and also return the resulting environment tensor.

        Returns (mask, val, bool, env).
        """
        m, v, b = self.run_slot(k, state, z)
        env_out = getattr(self, "_last_env_out", None)
        if env_out is None:
            env_out = state.env if state.env is not None else state.H.new_zeros(state.H.size(0), 4)
        return m, v, b, env_out

    def _run_slot_dynamic(
        self, k: int, state: TypedState, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slot = self._slots[k]
        queue: List[StepSpec] = list(slot.steps)
        steps_executed = 0
        cur_mask, cur_val, cur_bool = state.mask, state.val, state.boolean
        cur_env = state.env
        while steps_executed < self.max_body_len and len(queue) > 0:
            step = queue.pop(0)
            if step.kind == "return":
                break
            if step.kind == "cond":
                # Evaluate both branches as macro-calls and convex mix by guard (cur_bool)
                then_id = int(step.idx) if step.idx is not None else -1
                else_id = int(step.idx2) if step.idx2 is not None else -1
                m_t, v_t, b_t = cur_mask, cur_val, cur_bool
                m_e, v_e, b_e = cur_mask, cur_val, cur_bool
                if 0 <= then_id < self.num_slots and self._slots[then_id].steps:
                    ts = TypedState(
                        H=state.H,
                        mask=cur_mask,
                        val=cur_val,
                        boolean=cur_bool,
                        ptr_start=state.ptr_start,
                        ptr_end=state.ptr_end,
                        env=cur_env,
                    )
                    m_t, v_t, b_t, e_t = self.run_slot_env(then_id, ts, z)
                if 0 <= else_id < self.num_slots and self._slots[else_id].steps:
                    ts = TypedState(
                        H=state.H,
                        mask=cur_mask,
                        val=cur_val,
                        boolean=cur_bool,
                        ptr_start=state.ptr_start,
                        ptr_end=state.ptr_end,
                        env=cur_env,
                    )
                    m_e, v_e, b_e, e_e = self.run_slot_env(else_id, ts, z)
                g = cur_bool.clamp(0.0, 1.0)
                cur_mask = g * m_t + (1 - g) * m_e
                cur_val = g * v_t + (1 - g) * v_e
                cur_bool = g * b_t + (1 - g) * b_e
                # blend environments as well if both branches changed it
                try:
                    cur_env = g * e_t + (1 - g) * e_e
                except Exception:
                    pass
                steps_executed += 1
                continue
            if step.kind == "primitive":
                if step.idx is None or step.idx < 0 or step.idx >= len(self.base_ops):
                    steps_executed += 1
                    continue
                op = self.base_ops[step.idx]
                ts = TypedState(
                    H=state.H,
                    mask=cur_mask,
                    val=cur_val,
                    boolean=cur_bool,
                    ptr_start=state.ptr_start,
                    ptr_end=state.ptr_end,
                    env=cur_env,
                )
                m, v, b = op(ts, z)
                cur_mask, cur_val, cur_bool = m, v, b
                # apply env update if provided
                if hasattr(op, "env_update") and callable(getattr(op, "env_update")):
                    e_new = op.env_update(ts, z)
                    if e_new is not None:
                        cur_env = e_new
                steps_executed += 1
                continue
            if step.kind == "function":
                if not self.allow_higher_order:
                    steps_executed += 1
                    continue
                callee = int(step.idx) if step.idx is not None else -1
                if callee == k and not self.allow_self_call:
                    steps_executed += 1
                    continue
                if callee < 0 or callee >= self.num_slots:
                    steps_executed += 1
                    continue
                callee_steps = list(self._slots[callee].steps)
                if len(callee_steps) == 0:
                    steps_executed += 1
                    continue
                queue = callee_steps + queue
                steps_executed += 1
                continue
            steps_executed += 1
        self._last_env_out = cur_env
        return cur_mask, cur_val, cur_bool

    def _ensure_flat(self, k: int):
        if self._flat_versions[k] != self._slots[k].version:
            self._flat_bodies[k] = self._flatten_slot(k)
            self._flat_versions[k] = self._slots[k].version

    def _flatten_slot(self, k: int) -> List[StepSpec]:
        budget = self.max_body_len
        out: List[StepSpec] = []

        def dfs(sid: int, remaining: int, visited: set[int]) -> int:
            if remaining <= 0:
                return 0
            if sid < 0 or sid >= self.num_slots:
                return 0
            slot = self._slots[sid]
            for st in slot.steps:
                if len(out) >= self.max_body_len:
                    break
                if st.kind == "return":
                    out.append(st)
                    break
                if st.kind == "primitive":
                    out.append(st)
                    continue
                if st.kind == "cond":
                    # keep as a single step; resolved at runtime using current boolean
                    out.append(st)
                    continue
                if st.kind == "function":
                    if not self.allow_higher_order:
                        continue
                    callee = int(st.idx) if st.idx is not None else -1
                    if callee == sid and not self.allow_self_call:
                        continue
                    if callee in visited:
                        continue
                    # Recurse
                    visited.add(callee)
                    dfs(callee, remaining, visited)
                    visited.discard(callee)
                    continue
            return len(out)

        dfs(k, budget, {k})
        # Truncate to budget
        return out[: self.max_body_len]

    def _run_slot_flat(
        self, k: int, state: TypedState, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_flat(k)
        steps = self._flat_bodies[k]
        cur_mask, cur_val, cur_bool = state.mask, state.val, state.boolean
        cur_env = state.env
        for st in steps:
            if st.kind == "return":
                break
            if st.kind == "cond":
                then_id = int(st.idx) if st.idx is not None else -1
                else_id = int(st.idx2) if st.idx2 is not None else -1
                m_t, v_t, b_t = cur_mask, cur_val, cur_bool
                m_e, v_e, b_e = cur_mask, cur_val, cur_bool
                if 0 <= then_id < self.num_slots and self._slots[then_id].steps:
                    ts = TypedState(
                        H=state.H,
                        mask=cur_mask,
                        val=cur_val,
                        boolean=cur_bool,
                        ptr_start=state.ptr_start,
                        ptr_end=state.ptr_end,
                        env=cur_env,
                    )
                    m_t, v_t, b_t = self._run_slot_flat(then_id, ts, z)
                if 0 <= else_id < self.num_slots and self._slots[else_id].steps:
                    ts = TypedState(
                        H=state.H,
                        mask=cur_mask,
                        val=cur_val,
                        boolean=cur_bool,
                        ptr_start=state.ptr_start,
                        ptr_end=state.ptr_end,
                        env=cur_env,
                    )
                    m_e, v_e, b_e = self._run_slot_flat(else_id, ts, z)
                g = cur_bool.clamp(0.0, 1.0)
                cur_mask = g * m_t + (1 - g) * m_e
                cur_val = g * v_t + (1 - g) * v_e
                cur_bool = g * b_t + (1 - g) * b_e
                continue
            if st.kind != "primitive":
                continue
            if st.idx is None or st.idx < 0 or st.idx >= len(self.base_ops):
                continue
            op = self.base_ops[st.idx]
            ts = TypedState(
                H=state.H,
                mask=cur_mask,
                val=cur_val,
                boolean=cur_bool,
                ptr_start=state.ptr_start,
                ptr_end=state.ptr_end,
                env=cur_env,
            )
            m, v, b = op(ts, z)
            cur_mask, cur_val, cur_bool = m, v, b
            if hasattr(op, "env_update") and callable(getattr(op, "env_update")):
                e_new = op.env_update(ts, z)
                if e_new is not None:
                    cur_env = e_new
        self._last_env_out = cur_env
        return cur_mask, cur_val, cur_bool

    def return_logit(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Phase B (bounded recursion): when Reasoner is 'inside' a function,
        this head provides a RETURN probability (NPI-style).
        Returns a logit of shape (B,1).
        """
        return self._return_head(torch.cat([s, z], dim=-1))

    # --- OPTIONAL HOOKS CALLED BY REASONER DURING SLEEP ---

    def propose_candidates(self, traces, cfg) -> List[FnCandidate]:
        """Optional: mine subsequences from traces.

        This reference implementation stays passive and returns no proposals.
        """
        return []

    def install(self, candidates: List[FnCandidate]) -> List[int]:
        """Create/overwrite slots; returns installed slot ids.

        Strategy: overwrite empty slots first, else round-robin overwrite.
        Each candidate's arg_templates are appended to the shared pool and their
        local indices are remapped into global template ids.
        """
        if not candidates:
            return []
        installed: List[int] = []

        # Find empty slots first
        empty_ids = [s.id for s in self._slots if len(s.steps) == 0]
        rr_ids = [s.id for s in self._slots if len(s.steps) > 0]
        rr_ptr = 0

        for cand in candidates:
            # append candidate's templates and record global ids
            tmpl_ids = []
            for t in cand.arg_templates:
                self.templates.append(t)
                tmpl_ids.append(len(self.templates) - 1)

            # remap step.arg_tmpl_id if provided (assume local 0..len-1)
            remapped_steps: List[StepSpec] = []
            for st in cand.steps:
                if st.arg_tmpl_id is None:
                    remapped_steps.append(StepSpec(kind=st.kind, idx=st.idx, arg_tmpl_id=None))
                else:
                    if st.arg_tmpl_id < 0 or st.arg_tmpl_id >= len(tmpl_ids):
                        remapped_steps.append(StepSpec(kind=st.kind, idx=st.idx, arg_tmpl_id=None))
                    else:
                        remapped_steps.append(
                            StepSpec(kind=st.kind, idx=st.idx, arg_tmpl_id=tmpl_ids[st.arg_tmpl_id])
                        )

            if len(empty_ids) > 0:
                sid = empty_ids.pop(0)
            else:
                if len(rr_ids) == 0:
                    sid = 0
                else:
                    sid = rr_ids[rr_ptr % len(rr_ids)]
                    rr_ptr += 1

            self._slots[sid] = Slot(
                id=sid,
                steps=remapped_steps[: self.max_body_len],
                arg_templates=tmpl_ids,
                ret_policy=cand.ret_policy,
                version=self._slots[sid].version + 1,
            )
            installed.append(sid)
        # Refresh flat cache for deterministic eval
        self._recompute_all_flat()
        return installed

    def prune(self, usage_ema: List[float], threshold: float) -> List[int]:
        """Free slots with low usage; returns pruned slot ids."""
        pruned: List[int] = []
        for i, u in enumerate(usage_ema[: self.num_slots]):
            if u < threshold and len(self._slots[i].steps) > 0:
                self._slots[i] = Slot(
                    id=i, steps=[], arg_templates=[], ret_policy="implicit", version=self._slots[i].version + 1
                )
                pruned.append(i)
        if pruned:
            self._recompute_all_flat()
        return pruned

    def _recompute_all_flat(self):
        for i in range(self.num_slots):
            self._flat_bodies[i] = self._flatten_slot(i)
            self._flat_versions[i] = self._slots[i].version

    def clear_slots(self, ids: List[int]) -> None:
        """Clear specific slots (make empty) and refresh caches."""
        changed = False
        for sid in ids:
            if 0 <= sid < self.num_slots and len(self._slots[sid].steps) > 0:
                self._slots[sid] = Slot(
                    id=sid,
                    steps=[],
                    arg_templates=[],
                    ret_policy="implicit",
                    version=self._slots[sid].version + 1,
                )
                changed = True
        if changed:
            self._recompute_all_flat()

    # --- Convenience / Introspection ---
    @property
    def slots(self) -> List[Slot]:
        return list(self._slots)

    def num_primitives(self) -> int:
        return len(self.base_ops)
