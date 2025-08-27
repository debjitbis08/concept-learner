import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from concept_learner.reasoning_ops import (
    TypedState,
    OpBase,
    OpFilterMLP,
    OpCompareGeneric,
    OpCount,
    OpAdd,
)


# --- ReasonerV2 ------------------------------------------------------------


class ReasonerV2(nn.Module):
    """
    Multi-step reasoning with a soft mixture-of-operators per step.
    Keeps your STOP head and broadcast.
    """

    def __init__(
        self,
        d_model: int,
        max_steps: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()
        ops = [
            OpFilterMLP(d_model),  # Filter(p)
            OpCount(d_model, learn_phi=False),  # Count()
            OpAdd(d_model, use_z=True),  # Generic Add, k from z
            OpCompareGeneric(d_model, use_z=True),  # Unified Compare
        ]
        assert len(ops) >= 1, "Need at least one operator"
        self.d = d_model
        self.ops = nn.ModuleList(ops)
        self.max_steps = max_steps
        self.temperature = float(temperature)

        # token featurizer for initial state
        self.phi = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        # gating + stop + broadcast
        self.to_action = nn.Linear(2 * d_model, len(ops))  # logits over ops
        self.to_stop = nn.Linear(2 * d_model, 1)
        self.broadcast = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        # optional state stabilizer (delta & gate, like your v1)
        self.to_delta = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.to_gate = nn.Linear(2 * d_model, d_model)

        # small head to read an initial scalar value from the question tokens
        # val0 = read_number_from_tokens(H, mask)
        self.read_number = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(
        self, H: torch.Tensor, z: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
          H_reasoned:  (B,T,d)
          s_final:     (B,d)
          stop_logits: (B, max_steps)
          action_logits: (B, max_steps, num_ops)
        """
        B, T, d = H.shape
        m = mask.float().unsqueeze(-1)

        # initial typed state (DeepSets pool)
        H_phi = self.phi(H)
        s = (H_phi * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        # initialize typed scalar streams
        mask_float = mask.float().clamp(0.0, 1.0)  # start with the input padding mask
        # read initial number (scalar) from the question tokens
        h_pool = (H_phi * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        val0 = self.read_number(h_pool)  # (B,1)
        val = val0
        boolean = H.new_zeros(B, 1)  # no boolean decision yet

        stop_logits_all = []
        action_logits_all = []
        # track per-example STOP
        done = torch.zeros(B, dtype=torch.bool, device=H.device)

        for step in range(self.max_steps):
            x = torch.cat([s, z], dim=-1)  # (B,2d)

            # ----- action logits over ops -----
            logits_action = self.to_action(x)  # (B, num_ops)
            # Discrete operator selection per step
            if self.training:
                pick = F.gumbel_softmax(
                    logits_action, tau=self.temperature, hard=True, dim=-1
                )  # (B, num_ops) one-hot
            else:
                idx = logits_action.argmax(-1)  # (B,)
                pick = F.one_hot(idx, num_classes=len(self.ops)).float()

            # ----- run all typed-state ops -----
            ts = TypedState(H=H, mask=mask_float, val=val, boolean=boolean)

            cand_masks, cand_vals, cand_bools = [], [], []
            for op in self.ops:
                m_i, v_i, b_i = op(ts, z)  # m_i:(B,T), v_i:(B,1), b_i:(B,1)
                cand_masks.append(m_i)
                cand_vals.append(v_i)
                cand_bools.append(b_i)

            M = torch.stack(cand_masks, dim=-1)  # (B, T, num_ops)
            V = torch.stack(cand_vals, dim=-1)  # (B, 1, num_ops)
            Bv = torch.stack(cand_bools, dim=-1)  # (B, 1, num_ops)

            # ----- select operator outputs (discrete one-hot) -----
            new_mask = torch.einsum("btn,bn->bt", M, pick)  # (B, T)
            new_val = torch.einsum("bin,bn->bi", V, pick)  # (B, 1)
            new_bool = torch.einsum("bin,bn->bi", Bv, pick)  # (B, 1)

            # freeze items that are already done
            if done.any():
                mask_float = torch.where(done.unsqueeze(-1), mask_float, new_mask)
                val = torch.where(done.unsqueeze(-1), val, new_val)
                boolean = torch.where(done.unsqueeze(-1), boolean, new_bool)
            else:
                mask_float = new_mask
                val = new_val
                boolean = new_bool

            # ----- recompute pooled state s from updated mask -----
            H_phi = self.phi(H)  # (B, T, d)
            denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            s_new = (H_phi * mask_float.unsqueeze(-1)).sum(dim=1) / denom  # (B, d)
            if done.any():
                s = torch.where(done.unsqueeze(-1), s, s_new)
            else:
                s = s_new

            # STOP head (per-example)
            stop_logit = self.to_stop(torch.cat([s, z], dim=-1))  # (B,1)
            stop_logits_all.append(stop_logit)
            action_logits_all.append(logits_action.unsqueeze(1))  # (B,1,num_ops)

            p_stop = torch.sigmoid(stop_logit).squeeze(-1)  # (B,)
            done = done | (p_stop > 0.5)

        stop_logits = torch.cat(stop_logits_all, dim=1)  # (B, max_steps)
        action_logits = torch.cat(action_logits_all, dim=1)  # (B, max_steps, num_ops)

        # broadcast final state to tokens
        s_rep = s.unsqueeze(1).expand(B, T, d)
        H_reasoned = H + self.broadcast(torch.cat([H, s_rep], dim=-1))

        # expose final scalar value stream for downstream decoders
        self._last_val = val

        return H_reasoned, s, stop_logits, action_logits
