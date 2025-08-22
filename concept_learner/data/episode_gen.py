import math
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


class EpisodeGenerator:
    """
    Generates synthetic episodes over integers with hidden factors and simple relations.
    Raw descriptors are digit sequences under random base and random digit remapping.
    """

    def __init__(self, cfg: EpisodeConfig):
        self.cfg = cfg
        self.n_items = cfg.max_number

        # Precompute hidden factors (not shown to the model)
        device = cfg.device
        self.parity = torch.tensor([i % 2 for i in range(self.n_items)], device=device)
        self.mod3 = torch.tensor([i % 3 for i in range(self.n_items)], device=device)
        self.magnitude = torch.tensor([i // 10 for i in range(self.n_items)], device=device)

    # ----- Public API -----
    def sample_views(self, batch: int, change_base_prob: float = 0.0, easy_same_remap_prob: float = 0.2) -> Dict[str, torch.Tensor]:
        """
        Returns two augmented views of the same items.
        change_base_prob controls how often the base also changes across views.
        """
        device = self.cfg.device
        idx = torch.randint(0, self.n_items, (batch,), device=device)
        v1_desc, v1_mask, v1_base = self._render_batch(idx)
        if random.random() < change_base_prob:
            v2_desc, v2_mask, v2_base = self._render_batch(idx)
        else:
            # Occasionally keep the same remap exactly (easy positives)
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

    def sample_posneg_pairs(self, batch: int) -> Dict[str, torch.Tensor]:
        """
        Returns pairs of items that share a hidden factor (positive) or not (negative).
        Label 1 for positive, 0 for negative.
        """
        half = batch // 2
        a_pos, b_pos = self._pair_same_factor(half)
        a_neg, b_neg = self._pair_diff_factor(batch - half)
        a = torch.cat([a_pos, a_neg], dim=0)
        b = torch.cat([b_pos, b_neg], dim=0)
        y = torch.cat([torch.ones(half), torch.zeros(batch - half)]).long().to(self.cfg.device)
        perm = torch.randperm(batch, device=self.cfg.device)
        a, b, y = a[perm], b[perm], y[perm]
        a_desc, a_mask, a_base = self._render_batch(a)
        b_desc, b_mask, b_base = self._render_batch(b)
        return {
            "a_idx": a,
            "b_idx": b,
            "a_desc": a_desc,
            "a_mask": a_mask,
            "a_base": a_base,
            "b_desc": b_desc,
            "b_mask": b_mask,
            "b_base": b_base,
            "label": y,
            "domain": torch.zeros(batch, dtype=torch.long, device=self.cfg.device),
        }

    def sample_triples(self, batch: int) -> Dict[str, torch.Tensor]:
        """
        Sample true and corrupted triples (s, r, o).
        Relations: 0 same_parity, 1 next_in_sequence, 2 same_mod3.
        """
        device = self.cfg.device
        s = torch.randint(0, self.n_items, (batch,), device=device)
        r = torch.randint(0, 3, (batch,), device=device)
        o = torch.zeros_like(s)
        for i in range(batch):
            if r[i] == 0:
                cand = torch.arange(0, self.n_items, device=device)[self.parity == self.parity[s[i]]]
                while True:
                    choice = cand[torch.randint(0, len(cand), (1,), device=device)]
                    if choice != s[i]:
                        o[i] = choice
                        break
            elif r[i] == 1:
                o[i] = (s[i] + 1) % self.n_items
            else:
                cand = torch.arange(0, self.n_items, device=device)[self.mod3 == self.mod3[s[i]]]
                while True:
                    choice = cand[torch.randint(0, len(cand), (1,), device=device)]
                    if choice != s[i]:
                        o[i] = choice
                        break
        # Hard negatives
        o_corrupt = torch.empty_like(o)
        for i in range(batch):
            o_corrupt[i] = self._corrupt_o_hard(int(s[i].item()), int(r[i].item()), int(o[i].item()))
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

    def sample_analogies(self, batch: int, allowed_relations: List[int] | None = None) -> Dict[str, torch.Tensor]:
        """
        Sample analogy tuples (A, B, C, D) where relation between A,B equals C,D.
        We use the same 3 relations as in triples. Provide in-batch negatives.
        """
        device = self.cfg.device
        A = torch.randint(0, self.n_items, (batch,), device=device)
        if allowed_relations is None or len(allowed_relations) == 0:
            r = torch.randint(0, 3, (batch,), device=device)
        else:
            choices = torch.tensor(allowed_relations, device=device)
            sel = torch.randint(0, len(choices), (batch,), device=device)
            r = choices[sel]
        B = torch.zeros_like(A)
        C = torch.randint(0, self.n_items, (batch,), device=device)
        D = torch.zeros_like(A)
        for i in range(batch):
            if r[i] == 0:
                candA = torch.arange(0, self.n_items, device=device)[self.parity == self.parity[A[i]]]
                candC = torch.arange(0, self.n_items, device=device)[self.parity == self.parity[C[i]]]
                while True:
                    Bcand = candA[torch.randint(0, len(candA), (1,), device=device)]
                    if Bcand != A[i]:
                        B[i] = Bcand
                        break
                while True:
                    Dcand = candC[torch.randint(0, len(candC), (1,), device=device)]
                    if Dcand != C[i]:
                        D[i] = Dcand
                        break
            elif r[i] == 1:
                B[i] = (A[i] + 1) % self.n_items
                D[i] = (C[i] + 1) % self.n_items
            else:
                candA = torch.arange(0, self.n_items, device=device)[self.mod3 == self.mod3[A[i]]]
                candC = torch.arange(0, self.n_items, device=device)[self.mod3 == self.mod3[C[i]]]
                while True:
                    Bcand = candA[torch.randint(0, len(candA), (1,), device=device)]
                    if Bcand != A[i]:
                        B[i] = Bcand
                        break
                while True:
                    Dcand = candC[torch.randint(0, len(candC), (1,), device=device)]
                    if Dcand != C[i]:
                        D[i] = Dcand
                        break
        A_desc, A_mask, A_base = self._render_batch(A)
        B_desc, B_mask, B_base = self._render_batch(B)
        C_desc, C_mask, C_base = self._render_batch(C)
        D_desc, D_mask, D_base = self._render_batch(D)
        return {
            "A_idx": A,
            "B_idx": B,
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
                cand = torch.arange(0, self.n_items, device=self.cfg.device)[self.parity == self.parity[a[i]]]
            elif which[i] == 1:
                cand = torch.arange(0, self.n_items, device=self.cfg.device)[self.mod3 == self.mod3[a[i]]]
            else:
                cand = torch.arange(0, self.n_items, device=self.cfg.device)[self.magnitude == self.magnitude[a[i]]]
            b[i] = cand[torch.randint(0, len(cand), (1,), device=self.cfg.device)]
        return a, b

    def _pair_diff_factor(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        which = torch.randint(0, 3, (n,), device=self.cfg.device)
        a = torch.randint(0, self.n_items, (n,), device=self.cfg.device)
        b = torch.zeros_like(a)
        for i in range(n):
            if which[i] == 0:
                cand = torch.arange(0, self.n_items, device=self.cfg.device)[self.parity != self.parity[a[i]]]
            elif which[i] == 1:
                cand = torch.arange(0, self.n_items, device=self.cfg.device)[self.mod3 != self.mod3[a[i]]]
            else:
                cand = torch.arange(0, self.n_items, device=self.cfg.device)[self.magnitude != self.magnitude[a[i]]]
            b[i] = cand[torch.randint(0, len(cand), (1,), device=self.cfg.device)]
        return a, b

    def _render_batch(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = idx.shape[0]
        max_len = self.cfg.max_len
        device = self.cfg.device
        PAD_ID = self.cfg.max_base + 1
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
                # Ignore random remap; keep canonical digits for stability
                tokens = [d + 1 for d in digits]
            else:
                remapped = [remap[d] for d in digits]
                tokens = [d + 1 for d in remapped]
            L = min(len(tokens), max_len)
            seq[i, max_len - L : max_len] = torch.tensor(tokens[-L:], device=device)
            attn[i, max_len - L : max_len] = True
        return seq, attn, base

    def _render_batch_with_fixed_base(self, idx: torch.Tensor, base: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = idx.shape[0]
        max_len = self.cfg.max_len
        device = self.cfg.device
        PAD_ID = self.cfg.max_base + 1
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
        return seq, attn, base

    @staticmethod
    def _to_base(x: int, base: int) -> List[int]:
        if x == 0:
            return [0]
        out = []
        while x > 0:
            out.append(x % base)
            x //= base
        return list(reversed(out))

    # ----- Hard negative sampler -----
    def _corrupt_o_hard(self, s: int, r: int, o_true: int) -> int:
        while True:
            cand = int(torch.randint(0, self.n_items, (1,), device=self.cfg.device).item())
            if r == 0:
                if self.mod3[cand] == self.mod3[s] and self.parity[cand] != self.parity[s] and cand != o_true:
                    return cand
            elif r == 2:
                if self.parity[cand] == self.parity[s] and self.mod3[cand] != self.mod3[s] and cand != o_true:
                    return cand
            else:
                if cand != (s + 1) % self.n_items and abs(cand - s) <= 3 and cand != o_true:
                    return cand
