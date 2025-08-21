from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Simple VQ-VAE layer with straight-through estimator.
    """

    def __init__(self, num_codes: int = 128, code_dim: int = 64, commitment_cost: float = 0.25):
        super().__init__()
        self.code_dim = code_dim
        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)
        self.beta = commitment_cost

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, D)
        with torch.no_grad():
            codebook = self.codebook.weight  # (K, D)
            # compute distances to codebook vectors
            d = (
                torch.sum(z ** 2, dim=1, keepdim=True)
                + torch.sum(codebook ** 2, dim=1)
                - 2 * torch.matmul(z, codebook.t())
            )  # (B, K)
            indices = torch.argmin(d, dim=1)  # (B,)
        z_q = self.codebook(indices)
        # Losses
        commit_loss = self.beta * F.mse_loss(z.detach(), z_q)
        codebook_loss = F.mse_loss(z, z_q.detach())
        # Straight-through trick
        z_st = z + (z_q - z).detach()
        return z_st, indices, commit_loss + codebook_loss


class EmaVectorQuantizer(nn.Module):
    """
    EMA-based Vector Quantizer (VQ-VAE) for stability and better codebook usage.
    Updates codebook via exponential moving averages of assignments and encoder outputs.
    """

    def __init__(self, num_codes: int = 128, code_dim: int = 64, commitment_cost: float = 0.25, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = commitment_cost
        self.decay = decay
        self.eps = eps
        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", torch.zeros(num_codes, code_dim))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, D)
        with torch.no_grad():
            codebook = self.codebook.weight  # (K, D)
            d = (
                torch.sum(z ** 2, dim=1, keepdim=True)
                + torch.sum(codebook ** 2, dim=1)
                - 2 * torch.matmul(z, codebook.t())
            )  # (B, K)
            indices = torch.argmin(d, dim=1)  # (B,)

        z_q = self.codebook(indices)  # (B, D)

        if self.training:
            with torch.no_grad():
                # one-hot assignments
                encodings = torch.zeros(indices.size(0), self.num_codes, device=z.device)
                encodings.scatter_(1, indices.view(-1, 1), 1)
                # update moving averages
                cluster_size = encodings.sum(0)  # (K,)
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                dw = torch.matmul(encodings.t(), z)  # (K, D)
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                # Laplace smoothing of cluster size
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.num_codes * self.eps)
                    * n
                )
                # normalize to get new codebook
                normalized = self.ema_w / cluster_size.unsqueeze(1).clamp_min(self.eps)
                self.codebook.weight.data.copy_(normalized)
                # Re-seed dead/unused codes to diversify
                dead = (self.ema_cluster_size < 1.0).nonzero(as_tuple=False).flatten()  # very low assignment
                if dead.numel() > 0:
                    # pick random z rows to reinitialize these codes
                    rand_idx = torch.randint(0, z.size(0), (dead.numel(),), device=z.device)
                    new_codes = z[rand_idx].detach() + 0.01 * torch.randn_like(z[rand_idx])
                    self.codebook.weight.data[dead] = new_codes

        # Loss: commitment only
        commit_loss = self.beta * F.mse_loss(z.detach(), z_q)
        z_st = z + (z_q - z).detach()
        return z_st, indices, commit_loss
