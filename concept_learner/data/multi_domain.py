from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from .episode_gen import EpisodeGenerator


class MultiDomainEpisodeGenerator:
    """
    Wrap multiple per-domain EpisodeGenerators and produce mixed-domain batches.
    Each sampler returns concatenated tensors with a 'domain' LongTensor.
    """

    def __init__(self, gens: Sequence[EpisodeGenerator], domain_batch_mix: str = "uniform"):
        assert len(gens) >= 2, "Use MultiDomainEpisodeGenerator for >= 2 domains"
        self.gens: List[EpisodeGenerator] = list(gens)
        self.mix = domain_batch_mix

    @property
    def cfg(self):
        # Expose first generator cfg for convenience
        return self.gens[0].cfg

    def _cat_with_domain(self, parts: List[Dict[str, torch.Tensor]], domain_ids: List[int]) -> Dict[str, torch.Tensor]:
        keys = parts[0].keys()
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            if k == "domain":
                continue
            out[k] = torch.cat([p[k] for p in parts], dim=0)
        B = out[next(iter(out))].shape[0]
        dom_list = []
        for p, d in zip(parts, domain_ids):
            dom_list.append(torch.full((p[next(iter(p))].shape[0],), d, dtype=torch.long, device=self.cfg.device))
        out["domain"] = torch.cat(dom_list, dim=0)
        return out

    def _split_batch(self, batch: int) -> List[int]:
        n = len(self.gens)
        if self.mix == "uniform":
            base = batch // n
            sizes = [base] * n
            sizes[0] += batch - base * n
            return sizes
        else:
            # Fallback: single-domain per batch, chosen uniformly
            sizes = [0] * n
            import random

            j = random.randrange(n)
            sizes[j] = batch
            return sizes

    def sample_views(self, batch: int, **kwargs) -> Dict[str, torch.Tensor]:
        sizes = self._split_batch(batch)
        outs = []
        doms = []
        for i, (g, b) in enumerate(zip(self.gens, sizes)):
            if b <= 0:
                continue
            outs.append(g.sample_views(b, **kwargs))
            doms.append(i)
        return self._cat_with_domain(outs, doms)

    def sample_triples(self, batch: int) -> Dict[str, torch.Tensor]:
        sizes = self._split_batch(batch)
        outs = []
        doms = []
        for i, (g, b) in enumerate(zip(self.gens, sizes)):
            if b <= 0:
                continue
            outs.append(g.sample_triples(b))
            doms.append(i)
        return self._cat_with_domain(outs, doms)

    def sample_analogies(self, batch: int, **kwargs) -> Dict[str, torch.Tensor]:
        sizes = self._split_batch(batch)
        outs = []
        doms = []
        for i, (g, b) in enumerate(zip(self.gens, sizes)):
            if b <= 0:
                continue
            outs.append(g.sample_analogies(b, **kwargs))
            doms.append(i)
        return self._cat_with_domain(outs, doms)

    def sample_posneg_pairs(self, batch: int) -> Dict[str, torch.Tensor]:
        sizes = self._split_batch(batch)
        outs = []
        doms = []
        for i, (g, b) in enumerate(zip(self.gens, sizes)):
            if b <= 0:
                continue
            outs.append(g.sample_posneg_pairs(b))
            doms.append(i)
        return self._cat_with_domain(outs, doms)

    def sample_equivalent_pairs(self, batch: int) -> Dict[str, torch.Tensor]:
        """
        Pairs (x_d1, x_d2) representing the same underlying item across two domains.
        For toy numbers domains, the 'same idx' heuristic works.
        """
        assert len(self.gens) >= 2
        import random

        B = batch
        d1 = random.randrange(len(self.gens))
        d2 = (d1 + 1) % len(self.gens)
        g1, g2 = self.gens[d1], self.gens[d2]
        idx = torch.randint(0, g1.n_items, (B,), device=self.cfg.device)
        x1_desc, x1_mask, _ = g1._render_batch(idx)
        x2_desc, x2_mask, _ = g2._render_batch(idx)
        dom1 = torch.full((B,), d1, dtype=torch.long, device=self.cfg.device)
        dom2 = torch.full((B,), d2, dtype=torch.long, device=self.cfg.device)
        return {
            "x1_desc": x1_desc,
            "x1_mask": x1_mask,
            "x1_domain": dom1,
            "x2_desc": x2_desc,
            "x2_mask": x2_mask,
            "x2_domain": dom2,
        }

