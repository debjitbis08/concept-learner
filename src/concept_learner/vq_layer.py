import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize


class ResidualVQLayer(nn.Module):
    """
    Factorized VQ layer with parallel heads and optional serial, non-residual
    refiners. Keeps the same public interface as the previous RVQ wrapper:
    - forward accepts (B, D) or (B, L, D)
    - returns (z_q, indices_list, vq_loss)

    Design:
      Parallel heads: z_i = P_i h, quantized independently with small codebooks.
      Serial refiners: tiny MLPs on the concatenated code embeddings, each
      followed by its own quantizer. No residual numeric passing between stages.

    The number of returned index tensors equals `num_quantizers` for
    compatibility with existing callers and tests. By default, we allocate
    up to 2 parallel heads; remaining stages (if any) are serial refiners.
    """

    def __init__(
        self,
        in_dim: int,
        rvq_dim: int = 64,  # total concat dim of parallel heads
        codebook_size: int = 32,
        num_quantizers: int = 4,
        out_dim: int | None = None,
        # Optional knobs
        num_parallel_heads: int | None = None,
        serial_codebook_size: int | None = None,
        commitment_weight: float = 0.25,
        pre_vq_noise_std: float = 0.05,
        orth_weight: float = 0.0,
        entropy_weight: float = 0.0,
        # Unused legacy kwargs accepted for compatibility
        **_kwargs,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.total_parallel_dim = int(rvq_dim)
        self.codebook_size = int(codebook_size)
        self.num_quantizers = int(num_quantizers)
        self.num_parallel_heads = (
            int(num_parallel_heads)
            if num_parallel_heads is not None
            else min(2, self.num_quantizers)
        )
        self.num_serial = max(0, self.num_quantizers - self.num_parallel_heads)
        self.serial_codebook_size = int(serial_codebook_size or codebook_size)
        self.commitment_weight = float(commitment_weight)
        self.pre_vq_noise_std = float(pre_vq_noise_std)
        self.orth_weight = float(orth_weight)
        self.entropy_weight = float(entropy_weight)

        # Split total_parallel_dim across heads
        head_dims = [
            self.total_parallel_dim // self.num_parallel_heads
        ] * self.num_parallel_heads
        if self.num_parallel_heads > 0:
            head_dims[0] += self.total_parallel_dim - sum(head_dims)
        self.head_dims = head_dims

        # allowed kwargs for VectorQuantize to keep compatibility with tests
        allowed_kw = {
            "decay",
            "commitment_weight",
            "kmeans_init",
            "kmeans_iters",
            "use_cosine_sim",
            "threshold_ema_dead_code",
        }
        self._vq_kwargs = {k: v for k, v in _kwargs.items() if k in allowed_kw}

        # Parallel projections and quantizers
        self.parallel_proj = nn.ModuleList([nn.Linear(in_dim, d) for d in head_dims])
        self.parallel_vq = nn.ModuleList(
            [
                VectorQuantize(
                    dim=d,
                    codebook_size=self.codebook_size,
                    commitment_weight=self.commitment_weight,
                    **self._vq_kwargs,
                )
                for d in head_dims
            ]
        )

        concat_dim = sum(head_dims)
        self.concat_dim = concat_dim

        # Serial refiners: MLP (to base dim) + VQ (base dim).
        # Stage j sees input of size (1 + j) * concat_dim and outputs concat_dim.
        serial_modules = []
        for j in range(self.num_serial):
            inp_dim = (1 + j) * concat_dim
            hidden = max(32, inp_dim // 2)
            mlp = nn.Sequential(
                nn.Linear(inp_dim, hidden), nn.GELU(), nn.Linear(hidden, concat_dim)
            )
            vq = VectorQuantize(
                dim=concat_dim,
                codebook_size=self.serial_codebook_size,
                commitment_weight=self.commitment_weight,
                **self._vq_kwargs,
            )
            serial_modules.append(nn.ModuleDict({"mlp": mlp, "vq": vq}))
        self.serial = nn.ModuleList(serial_modules)

        # Final projection back to out_dim
        out_dim = int(out_dim or in_dim)
        final_in_dim = concat_dim * (1 + self.num_serial)
        self.proj_out = nn.Linear(final_in_dim, out_dim)

    def _entropy_bonus(self, indices_list, K_list) -> torch.Tensor:
        if self.entropy_weight <= 0:
            return torch.tensor(0.0, device=indices_list[0].device)
        H_target = 0.7 * torch.log(
            torch.tensor(float(max(K_list)), device=indices_list[0].device)
        )
        bonus = 0.0
        for idx, K in zip(indices_list, K_list):
            hist = torch.bincount(idx.view(-1), minlength=K).float()
            p = hist / (hist.sum() + 1e-8)
            ent = -(p * (p + 1e-8).log()).sum()
            over = (ent - H_target).clamp(min=0)
            bonus = bonus + over * over
        return self.entropy_weight * bonus

    def _orth_penalty(self) -> torch.Tensor:
        if self.orth_weight <= 0 or self.num_parallel_heads < 2:
            return torch.tensor(0.0, device=self.parallel_proj[0].weight.device)
        pen = 0.0
        for i in range(self.num_parallel_heads):
            for j in range(i + 1, self.num_parallel_heads):
                Pi = self.parallel_proj[i].weight
                Pj = self.parallel_proj[j].weight
                Pi = Pi / (Pi.norm(dim=1, keepdim=True) + 1e-8)
                Pj = Pj / (Pj.norm(dim=1, keepdim=True) + 1e-8)
                C = Pi @ Pj.t()
                pen = pen + (C**2).sum()
        return self.orth_weight * pen

    def forward(self, x: torch.Tensor):
        # x: (B, D) or (B, L, D)
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True

        # Parallel heads: project and quantize independently
        parallel_indices: list[torch.Tensor] = []
        parallel_embs: list[torch.Tensor] = []
        total_loss = torch.tensor(0.0, device=x.device)

        for proj, vq in zip(self.parallel_proj, self.parallel_vq):
            z = proj(x)  # (B, L, d_i)
            if self.training and self.pre_vq_noise_std > 0:
                z = z + torch.randn_like(z) * self.pre_vq_noise_std
            e, idx, loss = vq(z)
            parallel_embs.append(e)
            parallel_indices.append(idx)
            total_loss = total_loss + loss

        c = (
            torch.cat(parallel_embs, dim=-1)
            if len(parallel_embs) > 1
            else parallel_embs[0]
        )

        # Serial refiners
        serial_indices: list[torch.Tensor] = []
        serial_embs: list[torch.Tensor] = []
        current = c
        for stage in self.serial:
            u = stage["mlp"](current)
            if self.training and self.pre_vq_noise_std > 0:
                u = u + torch.randn_like(u) * self.pre_vq_noise_std
            e, idx, loss = stage["vq"](u)
            serial_embs.append(e)
            serial_indices.append(idx)
            total_loss = total_loss + loss
            current = torch.cat([current, e], dim=-1)

        # Final representation and projection back to out_dim
        all_feats = [c] + serial_embs
        final_repr = torch.cat(all_feats, dim=-1)
        z_q = self.proj_out(final_repr)
        # Expose the unprojected concatenated quantized features for downstream consumers
        self._last_all = final_repr

        # Regularizers
        total_loss = total_loss + self._orth_penalty()
        total_loss = total_loss + self._entropy_bonus(
            parallel_indices + serial_indices,
            [self.codebook_size] * len(parallel_indices)
            + [self.serial_codebook_size] * len(serial_indices),
        )

        # Normalize indices to list-of-tensors (one per quantizer)
        indices_list = parallel_indices + serial_indices
        if squeeze:
            z_q = z_q.squeeze(1)
            self._last_all = self._last_all.squeeze(1)
            indices_list = [
                t.squeeze(1) if t.ndim == 2 and t.shape[1] == 1 else t
                for t in indices_list
            ]

        # Ensure scalar loss (0-dim) for compatibility with tests
        if total_loss.ndim != 0:
            total_loss = total_loss.mean()
        return z_q, indices_list, total_loss
