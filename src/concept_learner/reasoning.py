import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Reasoner(nn.Module):
    """
    Multi-step reasoning layer.
      - pools token states -> typed state
      - updates state with delta + gate
      - repeats N steps or until STOP
      - broadcasts final state back to tokens
    """

    def __init__(self, d_model: int, max_steps: int = 4):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_steps

        # per-token embedding φ
        self.phi = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # update modules
        self.to_delta = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.to_gate = nn.Linear(2 * d_model, d_model)

        # stop classifier (decides if reasoning should halt at this step)
        self.to_stop = nn.Sequential(
            nn.Linear(2 * d_model, 1),
        )

        # broadcast back to tokens
        self.broadcast = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self, H: torch.Tensor, z: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        H:    (B, T, d)
        z:    (B, d)       # RVQ vector
        mask: (B, T)       # 1=real, 0=pad

        returns: H_reasoned (B, T, d), s_final (B, d)
        """
        B, T, d = H.shape
        m = mask.float().unsqueeze(-1)  # (B, T, 1)

        # initial state from DeepSets pool
        H_phi = self.phi(H)
        s = (H_phi * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  # (B, d)

        stop_probs = []
        stop_logits_all = []
        for step in range(self.max_steps):
            # update state
            x = torch.cat([s, z], dim=-1)  # (B, 2d)
            delta = self.to_delta(x)
            gate = torch.sigmoid(self.to_gate(x))
            s = s + gate * delta

            # predict STOP probability
            stop_logit = self.to_stop(x)  # (B,1)
            stop_logits_all.append(stop_logit)  # keep logits (not sigmoid)
            stop_p = torch.sigmoid(stop_logit)  # (B,1)
            stop_probs.append(stop_p)

            if (not self.training) and (torch.sigmoid(stop_logit) > 0.5).all():
                break

        H_reasoned = H + self.broadcast(
            torch.cat([H, s.unsqueeze(1).expand(B, T, d)], dim=-1)
        )
        stop_logits = torch.cat(stop_logits_all, dim=1)  # (B, num_steps)
        return H_reasoned, s, stop_logits
