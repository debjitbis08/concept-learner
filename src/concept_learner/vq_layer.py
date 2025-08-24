import torch
import torch.nn as nn

# Try to import lucidrains' ResidualVQ. Provide a minimal fallback if absent.
try:
    from vector_quantize_pytorch import ResidualVQ  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal CI env
    class ResidualVQ(nn.Module):
        """
        Minimal stand-in for vector-quantize-pytorch's ResidualVQ.
        It does NOT perform real quantization; it simply passes inputs through
        and returns zero indices and a small commitment-style loss so training
        and shapes work in tests without the dependency.
        """

        def __init__(
            self,
            dim: int,
            codebook_size: int = 16,
            num_quantizers: int = 2,
            **kwargs,
        ):
            super().__init__()
            self.dim = dim
            self.num_quantizers = int(num_quantizers)

        def forward(self, x: torch.Tensor):
            # Accept (B, L, D) or (B, D)
            squeeze = False
            if x.ndim == 2:
                x = x.unsqueeze(1)
                squeeze = True
            B, L, D = x.shape
            z_q = x  # identity
            # fake indices: zeros per quantizer
            all_indices = torch.zeros(B, L, self.num_quantizers, dtype=torch.long, device=x.device)
            # small commitment-like loss with gradient
            vq_loss = (z_q - z_q.detach()).pow(2).mean()
            if squeeze:
                z_q = z_q.squeeze(1)
            return z_q, all_indices, vq_loss


class ResidualVQLayer(nn.Module):
    """
    Wraps lucidrains ResidualVQ with in/out projections.
    Accepts (B, D) or (B, L, D), returns same rank.
    """

    def __init__(
        self,
        in_dim: int,  # encoder hidden size (e.g., 128)
        rvq_dim: int = 64,  # internal dim for RVQ
        codebook_size: int = 32,
        num_quantizers: int = 4,
        out_dim: int | None = None,
        **rvq_kwargs,  # decay, commitment_weight, etc.
    ):
        super().__init__()
        self.proj_in = (
            nn.Linear(in_dim, rvq_dim) if in_dim != rvq_dim else nn.Identity()
        )
        self.rvq = ResidualVQ(
            dim=rvq_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            **rvq_kwargs,
        )
        out_dim = out_dim or in_dim
        self.proj_out = (
            nn.Linear(rvq_dim, out_dim) if out_dim != rvq_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        # x: (B, D) or (B, L, D)
        squeeze_back = False
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B,1,D)
            squeeze_back = True

        x_proj = self.proj_in(x)  # (B, L, rvq_dim)
        z_q, all_indices, vq_loss = self.rvq(x_proj)  # all_indices: (B, L, Q)

        # --- make loss a scalar for training/tests ---
        # choose sum (across batch and levels). Can use .mean().
        if isinstance(vq_loss, torch.Tensor) and vq_loss.ndim > 0:
            vq_loss = vq_loss.sum()

        z_q = self.proj_out(z_q)  # (B, L, out_dim)

        # normalize indices to list-of-tensors (one per quantizer)
        if isinstance(all_indices, torch.Tensor):
            # ensure 3D: (B, L, Q)
            if all_indices.ndim == 2:
                all_indices = all_indices.unsqueeze(1)
            # split last dim into per-level tensors
            indices_list = list(all_indices.unbind(dim=-1))  # Q tensors of shape (B, L)
            if squeeze_back:
                indices_list = [t.squeeze(1) for t in indices_list]  # (B,)
            all_indices = indices_list

        if squeeze_back:
            z_q = z_q.squeeze(1)  # (B, out_dim)

        return z_q, all_indices, vq_loss
