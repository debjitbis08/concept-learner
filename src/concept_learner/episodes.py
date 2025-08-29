from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class EpisodeConfig:
    max_number: int = 100
    max_len: int = 4
    min_base: int = 5
    max_base: int = 10
    device: str = "cpu"
    canonicalize: bool = True
    enable_numeric_relations: bool = True


class EpisodeGenerator:
    """
    Generates synthetic episodes over integers with hidden factors and simple relations.
    Raw descriptors are digit sequences under random base and random digit remapping.
    Token space (by default):
      PAD=0, digits=1..base, optional context token id user-chosen (>=1).
    """

    def __init__(self, cfg: EpisodeConfig):
        self.cfg = cfg
        self.n_items = cfg.max_number
        self._context_token_id: int | None = None

        device = cfg.device
        self.parity = torch.tensor([i % 2 for i in range(self.n_items)], device=device)
        self.mod3 = torch.tensor([i % 3 for i in range(self.n_items)], device=device)
        self.magnitude = torch.tensor([i // 10 for i in range(self.n_items)], device=device)

    # ----- Context tokens -----
    def set_context_token_id(self, token_id: int | None) -> None:
        self._context_token_id = int(token_id) if token_id is not None else None

    # ----- Public API -----
    def sample_views(
        self, batch: int, change_base_prob: float = 0.0, easy_same_remap_prob: float = 0.2
    ) -> Dict[str, torch.Tensor]:
        device = self.cfg.device
        idx = torch.randint(0, self.n_items, (batch,), device=device)
        v1_desc, v1_mask, v1_base = self._render_batch(idx)
        if random.random() < change_base_prob:
            v2_desc, v2_mask, v2_base = self._render_batch(idx)
        else:
            if random.random() < easy_same_remap_prob:
                v2_desc, v2_mask, v2_base = v1_desc.clone(), v1_mask.clone(), v1_base.clone()
            else:
                v2_desc, v2_mask, v2_base = self._render_batch_with_fixed_base(idx, v1_base)
        return {
            "idx": idx,
            "view1_desc": v1_desc,
            "view1_mask": v1_mask,
            "view1_base": v1_base,
            "view2_desc": v2_desc,
            "view2_mask": v2_mask,
            "view2_base": v2_base,
            "domain": torch.zeros(batch, dtype=torch.long, device=device),
        }

    def sample_posneg_pairs(
        self,
        batch: int,
        allowed_relations: List[int] | None = None,
        idx_range: tuple[int, int] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Binary classification pairs on simple kindergarten-friendly relations.

        Positive if pair (a,b) satisfies one randomly chosen relation from:
          - same_parity
          - successor (b = a+1)
          - predecessor (b = a-1)
          - add_2 (b = a+2)
          - same_tens (tens(a) == tens(b))
          - same_ones (ones(a) == ones(b))
          - makes_ten (ones(a) + ones(b) == 10)
          - greater (a > b)
          - smaller (a < b)
        Negatives are sampled to violate the selected relation.
        """
        device = self.cfg.device
        n = batch
        if idx_range is None:
            a = torch.randint(0, self.n_items, (n,), device=device)
        else:
            lo, hi = int(idx_range[0]), int(idx_range[1])
            hi = min(hi, self.n_items - 1)
            lo = max(0, lo)
            if hi < lo:
                lo, hi = 0, self.n_items - 1
            a = torch.randint(lo, hi + 1, (n,), device=device)
        b = torch.empty_like(a)
        # choose relations uniformly from allowed set (default: all 0..8)
        if allowed_relations is None or len(allowed_relations) == 0:
            rel_choices = torch.arange(0, 9, device=device)
        else:
            rel_choices = torch.tensor(allowed_relations, dtype=torch.long, device=device)
        sel = torch.randint(0, len(rel_choices), (n,), device=device)
        rel = rel_choices[sel]
        y = torch.empty(n, dtype=torch.long, device=device)

        def _tens(x: int) -> int:
            return (x // 10) % 10

        def _ones(x: int) -> int:
            return x % 10

        def _make_b_with_ones_digit(ones: int) -> int:
            # choose a tens so that 10*t + ones < max_number
            max_tens = max(0, (self.n_items - 1 - ones) // 10)
            t = int(torch.randint(0, max_tens + 1, (1,), device=device).item())
            return t * 10 + ones

        # half positives, half negatives
        pos_mask = torch.zeros(n, dtype=torch.bool, device=device)
        pos_mask[: n // 2] = True
        pos_mask = pos_mask[torch.randperm(n, device=device)]

        for i in range(n):
            ai = int(a[i].item())
            ri = int(rel[i].item())
            if pos_mask[i]:
                # POSITIVE: construct b to satisfy relation ri
                if ri == 0:  # same_parity
                    # choose b with same parity but not equal to a
                    parity = ai % 2
                    # try to flip by +/- 2 within range
                    cand = ai + 2 if (ai + 2) < self.n_items else ai - 2
                    if cand >= 0 and cand < self.n_items and cand % 2 == parity:
                        bi = cand
                    else:
                        # fallback search
                        offs = torch.arange(0, self.n_items, device=device)
                        choices = offs[(offs % 2) == parity]
                        bi = int(choices[torch.randint(0, len(choices), (1,), device=device)].item())
                        if bi == ai and bi + 2 < self.n_items:
                            bi = ai + 2
                elif ri == 1:  # successor
                    bi = (ai + 1) % self.n_items
                elif ri == 2:  # predecessor
                    bi = (ai - 1) % self.n_items
                elif ri == 3:  # add_2
                    bi = (ai + 2) % self.n_items
                elif ri == 4:  # same_tens
                    bi = _tens(ai) * 10 + _ones(torch.randint(0, self.n_items, (1,), device=device).item()) % 10
                    # clamp if out of range
                    if bi >= self.n_items:
                        bi = _tens(ai) * 10
                elif ri == 5:  # same_ones
                    bi = _make_b_with_ones_digit(_ones(ai))
                elif ri == 6:  # makes_ten
                    comp = (10 - _ones(ai)) % 10
                    bi = _make_b_with_ones_digit(comp)
                elif ri == 7:  # greater (a > b)
                    bi = int(torch.randint(0, ai, (1,), device=device).item()) if ai > 0 else 0
                else:  # smaller (a < b)
                    bi = int(
                        torch.randint(ai + 1, self.n_items, (1,), device=device).item()
                    ) if ai + 1 < self.n_items else ai
                y[i] = 1
            else:
                # NEGATIVE: construct b to violate relation ri
                if ri == 0:  # same_parity -> choose different parity
                    bi = ai + 1 if (ai % 2 == 0) else ai + 1
                    if bi >= self.n_items:
                        bi = max(0, ai - 1)
                    if bi % 2 == ai % 2:
                        bi = (bi + 1) % self.n_items
                elif ri == 1:  # successor -> choose not successor
                    bi = (ai + 2) % self.n_items
                elif ri == 2:  # predecessor -> choose not predecessor
                    bi = (ai + 2) % self.n_items
                elif ri == 3:  # add_2 -> choose not a+2
                    bi = (ai + 1) % self.n_items
                elif ri == 4:  # same_tens -> choose different tens
                    ai_t = _tens(ai)
                    cand_t = (ai_t + 1) % 10
                    bi = cand_t * 10 + _ones(ai)
                    if bi >= self.n_items:
                        bi = (ai_t * 10 + ((_ones(ai) + 5) % 10))
                elif ri == 5:  # same_ones -> choose different ones
                    bi = _make_b_with_ones_digit((_ones(ai) + 1) % 10)
                elif ri == 6:  # makes_ten -> choose non-complement
                    wrong = (_ones(ai) + 1) % 10
                    bi = _make_b_with_ones_digit(wrong)
                elif ri == 7:  # greater -> choose b >= a
                    bi = ai
                else:  # smaller -> choose b <= a
                    bi = ai
                y[i] = 0
            b[i] = bi

        # shuffle and render
        perm = torch.randperm(n, device=device)
        a, b, y = a[perm], b[perm], y[perm]
        a_desc, a_mask, a_base = self._render_batch(a)
        b_desc, b_mask, b_base = self._render_batch(b)
        return {
            "a_idx": a,
            "b_idx": b,
            "rel": rel[perm],
            "a_desc": a_desc,
            "a_mask": a_mask,
            "a_base": a_base,
            "b_desc": b_desc,
            "b_mask": b_mask,
            "b_base": b_base,
            "label": y,
            "domain": torch.zeros(n, dtype=torch.long, device=self.cfg.device),
        }

    def sample_posneg_pairs_boundary(
        self,
        batch: int,
        diffs: Tuple[int, int, int] = (0, 1, 2),
        relations: Tuple[int, int] = (7, 8),
        idx_range: tuple[int, int] | None = None,
    ) -> Dict[str, torch.Tensor]:
        device = self.cfg.device
        n = batch
        # Safe sampling range to avoid boundary artifacts
        if idx_range is None:
            lo_safe, hi_safe = 1, max(1, self.n_items - 2)
        else:
            lo, hi = int(idx_range[0]), int(idx_range[1])
            hi = min(hi, self.n_items - 1)
            lo = max(0, lo)
            if hi < lo:
                lo, hi = 0, self.n_items - 1
            lo_safe = max(1, lo + 1)
            hi_safe = max(lo_safe, min(self.n_items - 2, hi - 1))
        # Half base items; we will add their counterfactual flips to reach n
        m = max(2, (n // 2) * 2)
        half = m // 2
        a_base = torch.randint(lo_safe, hi_safe + 1, (half,), device=device)
        rel_choices = torch.tensor(list(relations), device=device)
        rel_base = rel_choices[torch.randint(0, len(rel_choices), (half,), device=device)]
        # Balanced positives/negatives
        pos_mask = torch.zeros(half, dtype=torch.bool, device=device)
        pos_mask[: half // 2] = True
        pos_mask = pos_mask[torch.randperm(half, device=device)]
        b_base = torch.empty_like(a_base)
        y_base = torch.empty_like(a_base)
        # Use only diffs in {0,1} for boundary; treat 0 as equal for negatives
        for i in range(half):
            ai = int(a_base[i].item())
            ri = int(rel_base[i].item())
            if ri == 7:  # greater
                if pos_mask[i]:
                    bi = ai - 1  # ensure within [lo_safe-1, hi_safe+1]; ai>=1 guaranteed
                    y_base[i] = 1
                else:
                    bi = ai if torch.rand(1, device=device).item() < 0.5 else ai + 1
                    y_base[i] = 0
            else:  # smaller
                if pos_mask[i]:
                    bi = ai + 1
                    y_base[i] = 1
                else:
                    bi = ai if torch.rand(1, device=device).item() < 0.5 else ai - 1
                    y_base[i] = 0
            bi = min(max(bi, lo_safe - 1), hi_safe + 1)
            b_base[i] = bi
        # Counterfactual flips: invert boundary direction for each base item
        a_cf = a_base.clone()
        rel_cf = rel_base.clone()
        y_cf = 1 - y_base
        b_cf = torch.empty_like(b_base)
        for i in range(half):
            ai = int(a_base[i].item())
            ri = int(rel_base[i].item())
            if ri == 7:  # greater
                # If base was positive (b=ai-1), flip to b>=ai; else flip to b=ai-1
                if y_base[i] == 1:
                    b_cf[i] = ai if torch.rand(1, device=device).item() < 0.5 else ai + 1
                else:
                    b_cf[i] = ai - 1
            else:  # smaller
                if y_base[i] == 1:
                    b_cf[i] = ai if torch.rand(1, device=device).item() < 0.5 else ai - 1
                else:
                    b_cf[i] = ai + 1
            b_cf[i] = min(max(int(b_cf[i].item()), lo_safe - 1), hi_safe + 1)
        # Combine and trim to n
        a_all = torch.cat([a_base, a_cf], dim=0)
        b_all = torch.cat([b_base, b_cf], dim=0)
        rel_all = torch.cat([rel_base, rel_cf], dim=0)
        y_all = torch.cat([y_base, y_cf], dim=0)
        # Shuffle and take first n
        perm = torch.randperm(a_all.size(0), device=device)
        a = a_all[perm][:n]
        b = b_all[perm][:n]
        rel = rel_all[perm][:n]
        y = y_all[perm][:n]
        a_desc, a_mask, a_base_tok = self._render_batch(a)
        b_desc, b_mask, b_base_tok = self._render_batch(b)
        return {
            "a_idx": a,
            "b_idx": b,
            "rel": rel,
            "a_desc": a_desc,
            "a_mask": a_mask,
            "a_base": a_base_tok,
            "b_desc": b_desc,
            "b_mask": b_mask,
            "b_base": b_base_tok,
            "label": y,
            "domain": torch.zeros(n, dtype=torch.long, device=self.cfg.device),
        }

    def sample_place_value(
        self, batch: int, allowed_kinds: List[int] | None = None, idx_range: tuple[int, int] | None = None
    ) -> Dict[str, torch.Tensor]:
        """
        Returns questions about digits and place values with numeric targets.
        kind encodes the task type:
          0 -> ones digit (target = ones(a))
          1 -> tens digit (target = tens(a))
          2 -> tens place value (target = 10 * tens(a))
          3 -> face value of the specified digit (ones/tens) (target = that digit)
        """
        device = self.cfg.device
        N = self.n_items
        if allowed_kinds is None or len(allowed_kinds) == 0:
            kind_choices = torch.arange(0, 4, device=device)
        else:
            kind_choices = torch.tensor(allowed_kinds, dtype=torch.long, device=device)
        sel = torch.randint(0, len(kind_choices), (batch,), device=device)
        kind = kind_choices[sel]
        if idx_range is None:
            a = torch.randint(0, N, (batch,), device=device)
        else:
            lo, hi = int(idx_range[0]), int(idx_range[1])
            hi = min(hi, N - 1)
            lo = max(0, lo)
            if hi < lo:
                lo, hi = 0, N - 1
            a = torch.randint(lo, hi + 1, (batch,), device=device)
        y = torch.zeros_like(a)
        face_place = torch.full_like(a, -1)  # -1: N/A, 0: ones, 1: tens
        for i in range(batch):
            ai = int(a[i].item())
            k = int(kind[i].item())
            tens = (ai // 10) % 10
            ones = ai % 10
            if k == 0:  # ones digit
                y[i] = ones
            elif k == 1:  # tens digit
                y[i] = tens
            elif k == 2:  # tens place value
                y[i] = tens * 10
            else:  # face value of a specific place (choose ones or tens)
                fp = int(torch.randint(0, 2, (1,), device=device).item())
                face_place[i] = fp
                y[i] = ones if fp == 0 else tens
        return {"kind": kind, "a": a, "target": y, "face_place": face_place}

    def sample_triples(self, batch: int) -> Dict[str, torch.Tensor]:
        device = self.cfg.device
        s = torch.randint(0, self.n_items, (batch,), device=device)
        max_r = 8 if self.cfg.enable_numeric_relations else 3
        r = torch.randint(0, max_r, (batch,), device=device)
        o = torch.zeros_like(s)
        for i in range(batch):
            if r[i] == 0:
                cand = torch.arange(0, self.n_items, device=device)[
                    self.parity == self.parity[s[i]]
                ]
                while True:
                    choice = cand[torch.randint(0, len(cand), (1,), device=device)]
                    if choice != s[i]:
                        o[i] = choice
                        break
            elif r[i] == 1:
                o[i] = (s[i] + 1) % self.n_items
            else:
                if (self.cfg.enable_numeric_relations) and r[i] >= 3:
                    ri = int(r[i].item())
                    si = int(s[i].item())
                    if ri == 3:
                        o[i] = (s[i] - 1) % self.n_items
                    elif ri == 4:
                        o[i] = (s[i] + 2) % self.n_items
                    elif ri == 5:
                        tens = (si // 10) % 10
                        o[i] = torch.tensor(tens, device=device)
                    elif ri == 6:
                        ones = si % 10
                        o[i] = torch.tensor(ones, device=device)
                    elif ri == 7:
                        o_val = (10 - (si % 10)) % 10
                        o[i] = torch.tensor(o_val, device=device)
                    else:
                        o[i] = s[i]
                else:
                    cand = torch.arange(0, self.n_items, device=device)[
                        self.mod3 == self.mod3[s[i]]
                    ]
                    while True:
                        choice = cand[torch.randint(0, len(cand), (1,), device=device)]
                        if choice != s[i]:
                            o[i] = choice
                            break
        # Hard negatives
        o_corrupt = torch.empty_like(o)
        for i in range(batch):
            o_corrupt[i] = self._corrupt_o_hard(
                int(s[i].item()), int(r[i].item()), int(o[i].item())
            )
        s_desc, s_mask, s_base = self._render_batch(s)
        o_desc, o_mask, o_base = self._render_batch(o)
        o_neg_desc, o_neg_mask, o_neg_base = self._render_batch(o_corrupt)
        return {
            "s_idx": s,
            "r": r,
            "o_idx": o,
            "o_neg_idx": o_corrupt,
            "s_desc": s_desc,
            "s_mask": s_mask,
            "s_base": s_base,
            "o_desc": o_desc,
            "o_mask": o_mask,
            "o_base": o_base,
            "o_neg_desc": o_neg_desc,
            "o_neg_mask": o_neg_mask,
            "o_neg_base": o_neg_base,
            "domain": torch.zeros(batch, dtype=torch.long, device=self.cfg.device),
        }

    def sample_analogies(
        self, batch: int, allowed_relations: List[int] | None = None
    ) -> Dict[str, torch.Tensor]:
        device = self.cfg.device
        A = torch.randint(0, self.n_items, (batch,), device=device)
        if allowed_relations is None or len(allowed_relations) == 0:
            max_r = 3 + (5 if self.cfg.enable_numeric_relations else 0)
            r = torch.randint(0, max_r, (batch,), device=device)
        else:
            choices = torch.tensor(allowed_relations, device=device)
            sel = torch.randint(0, len(choices), (batch,), device=device)
            r = choices[sel]
        Bv = torch.zeros_like(A)
        C = torch.randint(0, self.n_items, (batch,), device=device)
        D = torch.zeros_like(A)
        for i in range(batch):
            if r[i] == 0:
                candA = torch.arange(0, self.n_items, device=device)[
                    self.parity == self.parity[A[i]]
                ]
                candC = torch.arange(0, self.n_items, device=device)[
                    self.parity == self.parity[C[i]]
                ]
                while True:
                    Bcand = candA[torch.randint(0, len(candA), (1,), device=device)]
                    if Bcand != A[i]:
                        Bv[i] = Bcand
                        break
                while True:
                    Dcand = candC[torch.randint(0, len(candC), (1,), device=device)]
                    if Dcand != C[i]:
                        D[i] = Dcand
                        break
            elif r[i] == 1:
                Bv[i] = (A[i] + 1) % self.n_items
                D[i] = (C[i] + 1) % self.n_items
            else:
                if self.cfg.enable_numeric_relations and r[i] >= 3:
                    ri = int(r[i].item())
                    if ri == 3:
                        Bv[i] = (A[i] - 1) % self.n_items
                        D[i] = (C[i] - 1) % self.n_items
                    elif ri == 4:
                        Bv[i] = (A[i] + 2) % self.n_items
                        D[i] = (C[i] + 2) % self.n_items
                    elif ri == 7:
                        Bv[i] = (10 - (A[i] % 10)) % 10
                        D[i] = (10 - (C[i] % 10)) % 10
                    else:
                        candA = torch.arange(0, self.n_items, device=device)[
                            self.mod3 == self.mod3[A[i]]
                        ]
                        candC = torch.arange(0, self.n_items, device=device)[
                            self.mod3 == self.mod3[C[i]]
                        ]
                        while True:
                            Bcand = candA[
                                torch.randint(0, len(candA), (1,), device=device)
                            ]
                            if Bcand != A[i]:
                                Bv[i] = Bcand
                                break
                        while True:
                            Dcand = candC[
                                torch.randint(0, len(candC), (1,), device=device)
                            ]
                            if Dcand != C[i]:
                                D[i] = Dcand
                                break
                else:
                    candA = torch.arange(0, self.n_items, device=device)[
                        self.mod3 == self.mod3[A[i]]
                    ]
                    candC = torch.arange(0, self.n_items, device=device)[
                        self.mod3 == self.mod3[C[i]]
                    ]
                    while True:
                        Bcand = candA[torch.randint(0, len(candA), (1,), device=device)]
                        if Bcand != A[i]:
                            Bv[i] = Bcand
                            break
                    while True:
                        Dcand = candC[torch.randint(0, len(candC), (1,), device=device)]
                        if Dcand != C[i]:
                            D[i] = Dcand
                            break
        A_desc, A_mask, A_base = self._render_batch(A)
        B_desc, B_mask, B_base = self._render_batch(Bv)
        C_desc, C_mask, C_base = self._render_batch(C)
        D_desc, D_mask, D_base = self._render_batch(D)
        return {
            "A_idx": A,
            "B_idx": Bv,
            "C_idx": C,
            "D_idx": D,
            "A_desc": A_desc,
            "A_mask": A_mask,
            "A_base": A_base,
            "B_desc": B_desc,
            "B_mask": B_mask,
            "B_base": B_base,
            "C_desc": C_desc,
            "C_mask": C_mask,
            "C_base": C_base,
            "D_desc": D_desc,
            "D_mask": D_mask,
            "D_base": D_base,
            "domain": torch.zeros(batch, dtype=torch.long, device=self.cfg.device),
        }

    # ----- Internal helpers -----
    def _pair_same_factor(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        which = torch.randint(0, 3, (n,), device=self.cfg.device)
        a = torch.randint(0, self.n_items, (n,), device=self.cfg.device)
        b = torch.zeros_like(a)
        for i in range(n):
            if which[i] == 0:
                cand = torch.arange(0, self.n_items, device=self.cfg.device)[
                    self.parity == self.parity[a[i]]
                ]
            elif which[i] == 1:
                cand = torch.arange(0, self.n_items, device=self.cfg.device)[
                    self.mod3 == self.mod3[a[i]]
                ]
            else:
                cand = torch.arange(0, self.n_items, device=self.cfg.device)[
                    self.magnitude == self.magnitude[a[i]]
                ]
            b[i] = cand[torch.randint(0, len(cand), (1,), device=self.cfg.device)]
        return a, b

    def _pair_diff_factor(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        which = torch.randint(0, 3, (n,), device=self.cfg.device)
        a = torch.randint(0, self.n_items, (n,), device=self.cfg.device)
        b = torch.zeros_like(a)
        for i in range(n):
            if which[i] == 0:
                cand = torch.arange(0, self.n_items, device=self.cfg.device)[
                    self.parity != self.parity[a[i]]
                ]
            elif which[i] == 1:
                cand = torch.arange(0, self.n_items, device=self.cfg.device)[
                    self.mod3 != self.mod3[a[i]]
                ]
            else:
                cand = torch.arange(0, self.n_items, device=self.cfg.device)[
                    self.magnitude != self.magnitude[a[i]]
                ]
            b[i] = cand[torch.randint(0, len(cand), (1,), device=self.cfg.device)]
        return a, b

    def _render_batch(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = idx.shape[0]
        max_len = self.cfg.max_len
        device = self.cfg.device
        PAD_ID = 0
        seq = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=device)
        attn = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        base = torch.randint(self.cfg.min_base, self.cfg.max_base + 1, (B,), device=device)
        for i in range(B):
            b = int(base[i].item())
            x = int(idx[i].item())
            digits = self._to_base(x, b)  # 0..b-1
            remap = list(range(b))
            random.shuffle(remap)
            if self.cfg.canonicalize:
                tokens = [d + 1 for d in digits]
            else:
                remapped = [remap[d] for d in digits]
                tokens = [d + 1 for d in remapped]
            L = min(len(tokens), max_len)
            seq[i, max_len - L : max_len] = torch.tensor(tokens[-L:], device=device)
            attn[i, max_len - L : max_len] = True
            if self._context_token_id is not None and max_len > 0:
                seq[i, 0] = int(self._context_token_id)
                attn[i, 0] = True
        return seq, attn, base

    def _render_batch_with_fixed_base(
        self, idx: torch.Tensor, base: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = idx.shape[0]
        max_len = self.cfg.max_len
        device = self.cfg.device
        PAD_ID = 0
        seq = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=device)
        attn = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        for i in range(B):
            b = int(base[i].item())
            x = int(idx[i].item())
            digits = self._to_base(x, b)
            remap = list(range(b))
            random.shuffle(remap)
            if self.cfg.canonicalize:
                tokens = [d + 1 for d in digits]
            else:
                tokens = [d + 1 for d in (remap[d] for d in digits)]
            L = min(len(tokens), max_len)
            seq[i, max_len - L : max_len] = torch.tensor(tokens[-L:], device=device)
            attn[i, max_len - L : max_len] = True
            if self._context_token_id is not None and max_len > 0:
                seq[i, 0] = int(self._context_token_id)
                attn[i, 0] = True
        return seq, attn, base

    @staticmethod
    def _to_base(x: int, base: int) -> List[int]:
        if x == 0:
            return [0]
        out: List[int] = []
        while x > 0:
            out.append(x % base)
            x //= base
        return list(reversed(out))

    def _corrupt_o_hard(self, s: int, r: int, o_true: int) -> int:
        while True:
            cand = int(torch.randint(0, self.n_items, (1,), device=self.cfg.device).item())
            if r == 0:
                if (
                    self.mod3[cand] == self.mod3[s]
                    and self.parity[cand] != self.parity[s]
                    and cand != o_true
                ):
                    return cand
            elif r == 2:
                if (
                    self.parity[cand] == self.parity[s]
                    and self.mod3[cand] != self.mod3[s]
                    and cand != o_true
                ):
                    return cand
            elif r == 1:
                if cand != (s + 1) % self.n_items and abs(cand - s) <= 3 and cand != o_true:
                    return cand
            elif r == 3:
                if cand != (s - 1) % self.n_items and abs(cand - s) <= 3 and cand != o_true:
                    return cand
            elif r == 4:
                if cand != (s + 2) % self.n_items and abs(cand - s) <= 4 and cand != o_true:
                    return cand
            elif r in (5, 6):
                if cand < 10 and cand != o_true:
                    return cand
            elif r == 7:
                if cand < 10 and cand != ((10 - (s % 10)) % 10) and cand != o_true:
                    return cand
            else:
                if cand != (s + 1) % self.n_items and abs(cand - s) <= 3 and cand != o_true:
                    return cand

    def numeric_gold_atoms(self) -> Dict[str, List]:
        gold = {
            "triples": [
                ["1", "successor_of", "0"],
                ["2", "successor_of", "1"],
                ["3", "predecessor_of", "4"],
                ["5", "add_2", "7"],
                ["10", "has_tens", "1"],
                ["10", "has_ones", "0"],
                ["7", "makes_ten_with", "3"],
                ["3_apples", "count", "3"],
                ["triangle", "has_sides", "3"],
            ],
            "analogies": [
                [["2", "3"], ["5", "6"]],
                [["2", "4"], ["3", "5"]],
                [["7", "10"], ["6", "?"]],
                [["1", "1st"], ["3", "3rd"]],
            ],
            "equivalences": [["three", "3"], ["ten", "10"], ["bike", "bicycle"]],
        }
        return gold

    def sample_number_equality(
        self, batch: int, idx_range: tuple[int, int] | None = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate simple numeric equality questions over integers.
        Returns a, b, and binary label (1 if a == b else 0).
        """
        device = self.cfg.device
        N = self.n_items
        if idx_range is None:
            a = torch.randint(0, N, (batch,), device=device)
        else:
            lo, hi = int(idx_range[0]), int(idx_range[1])
            hi = min(hi, N - 1)
            lo = max(0, lo)
            if hi < lo:
                lo, hi = 0, N - 1
            a = torch.randint(lo, hi + 1, (batch,), device=device)
        y = torch.zeros(batch, dtype=torch.long, device=device)
        b = torch.zeros_like(a)
        # half positives where b == a
        pos_mask = torch.zeros(batch, dtype=torch.bool, device=device)
        pos_mask[: batch // 2] = True
        pos_mask = pos_mask[torch.randperm(batch, device=device)]
        for i in range(batch):
            ai = int(a[i].item())
            if pos_mask[i]:
                b[i] = ai
                y[i] = 1
            else:
                # pick a different value
                if ai + 1 < N:
                    b[i] = ai + 1
                else:
                    b[i] = max(0, ai - 1)
                y[i] = 0
        return {"a": a, "b": b, "label": y}

    def sample_symbolic_compare(
        self, batch: int, idx_range: tuple[int, int] | None = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate symbolic comparison questions a>b? or a<b? with binary labels.
        Returns a, b, op (0: '>', 1: '<'), label (1 if expression true else 0).
        """
        device = self.cfg.device
        N = self.n_items
        if idx_range is None:
            a = torch.randint(0, N, (batch,), device=device)
            b = torch.randint(0, N, (batch,), device=device)
        else:
            lo, hi = int(idx_range[0]), int(idx_range[1])
            hi = min(hi, N - 1)
            lo = max(0, lo)
            if hi < lo:
                lo, hi = 0, N - 1
            a = torch.randint(lo, hi + 1, (batch,), device=device)
            b = torch.randint(lo, hi + 1, (batch,), device=device)
        op = torch.randint(0, 2, (batch,), device=device)  # 0:'>', 1:'<'
        y = torch.zeros(batch, dtype=torch.long, device=device)
        for i in range(batch):
            if op[i] == 0:
                y[i] = 1 if a[i] > b[i] else 0
            else:
                y[i] = 1 if a[i] < b[i] else 0
        return {"a": a, "b": b, "op": op, "label": y}

    def sample_addition(
        self, batch: int, idx_range: tuple[int, int] | None = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate addition questions a+b=? with numeric targets within range [0, N-1].
        Ensures a+b < N by sampling b accordingly.
        Returns a, b, target=a+b.
        """
        device = self.cfg.device
        N = self.n_items
        if idx_range is None:
            a = torch.randint(0, N, (batch,), device=device)
        else:
            lo, hi = int(idx_range[0]), int(idx_range[1])
            hi = min(hi, N - 1)
            lo = max(0, lo)
            if hi < lo:
                lo, hi = 0, N - 1
            a = torch.randint(lo, hi + 1, (batch,), device=device)
        b = torch.zeros_like(a)
        y = torch.zeros_like(a)
        for i in range(batch):
            ai = int(a[i].item())
            max_b = max(0, N - 1 - ai)
            bi = int(torch.randint(0, max_b + 1, (1,), device=device).item())
            b[i] = bi
            y[i] = ai + bi
        return {"a": a, "b": b, "target": y}

    def sample_parity(
        self, batch: int, idx_range: tuple[int, int] | None = None
    ) -> Dict[str, torch.Tensor]:
        """Parity classification for a single number: target 0=even, 1=odd."""
        device = self.cfg.device
        N = self.n_items
        if idx_range is None:
            a = torch.randint(0, N, (batch,), device=device)
        else:
            lo, hi = int(idx_range[0]), int(idx_range[1])
            hi = min(hi, N - 1)
            lo = max(0, lo)
            if hi < lo:
                lo, hi = 0, N - 1
            a = torch.randint(lo, hi + 1, (batch,), device=device)
        y = (a % 2).long()
        return {"a": a, "target": y}

    def sample_offset(
        self,
        batch: int,
        allowed_offsets: Tuple[int, ...] = (-3, -2, -1, 1, 2, 3),
        idx_range: tuple[int, int] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Simple +/- k tasks: choose offset o in allowed_offsets, pick a so that a+o in [0,N-1].
        Returns a, offset o, target=a+o.
        """
        device = self.cfg.device
        N = self.n_items
        offs = torch.tensor(list(allowed_offsets), dtype=torch.long, device=device)
        sel = torch.randint(0, len(offs), (batch,), device=device)
        o = offs[sel]
        a = torch.zeros(batch, dtype=torch.long, device=device)
        y = torch.zeros(batch, dtype=torch.long, device=device)
        for i in range(batch):
            oi = int(o[i].item())
            lo = max(0, -oi)
            hi = min(N - 1, N - 1 - oi)
            if idx_range is not None:
                lo = max(lo, int(idx_range[0]))
                hi = min(hi, int(idx_range[1]))
            if hi < lo:
                lo, hi = 0, N - 1 - abs(oi)
            ai = int(torch.randint(lo, hi + 1, (1,), device=device).item())
            a[i] = ai
            y[i] = ai + oi
        return {"a": a, "offset": o, "target": y}

    # ----- Counting tasks (next / previous / between) -----
    def sample_counting(self, batch: int, idx_range: tuple[int, int] | None = None) -> Dict[str, torch.Tensor]:
        """
        Returns counting questions with numeric targets (classification over 0..max_number-1):
          kind=0 successor/next:      input=a,          target=(a+1)%N
          kind=1 predecessor/previous: input=a,         target=(a-1)%N
          kind=2 between:             input=(a,c=a+2),  target=a+1
        """
        device = self.cfg.device
        N = self.n_items
        kind = torch.randint(0, 3, (batch,), device=device)
        if idx_range is None:
            a = torch.randint(0, N, (batch,), device=device)
        else:
            lo, hi = int(idx_range[0]), int(idx_range[1])
            hi = min(hi, N - 1)
            lo = max(0, lo)
            if hi < lo:
                lo, hi = 0, N - 1
            a = torch.randint(lo, hi + 1, (batch,), device=device)
        c = torch.full_like(a, -1)
        y = torch.zeros_like(a)
        for i in range(batch):
            k = int(kind[i].item())
            ai = int(a[i].item())
            if k == 0:  # successor
                y[i] = (ai + 1) % N
            elif k == 1:  # predecessor
                y[i] = (ai - 1) % N
            else:  # between
                # choose contiguous triple (a, a+1, a+2) without wrap-around
                if N >= 3:
                    ai = int(torch.randint(0, N - 2, (1,), device=device).item())
                    a[i] = ai
                    c[i] = ai + 2
                    y[i] = ai + 1
                else:
                    # degenerate small-N fallback: keep modulo behavior
                    ci = (ai + 2) % N
                    c[i] = ci
                    y[i] = (ai + 1) % N
        return {
            "kind": kind,  # 0 succ, 1 pred, 2 between
            "a": a,
            "c": c,  # valid only for between
            "target": y,
        }
