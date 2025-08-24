import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from concept_learner.reasoning_ops import (
    TypedState,
    OpBase,
    OpFilterMLP,
    OpCompare,
    OpCount,
    OpAddConst,
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
            OpAddConst(+1),  # Add(+1)
            OpCompare(k=1.0, op="gt"),  # Compare(>, 1)
        ]
        assert len(ops) >= 1, "Need at least one operator"
        self.d = d_model
        self.ops = nn.ModuleList(ops)
        self.max_steps = max_steps
        self.temperature = temperature

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

        # initial typed state scalars
        mask_float = mask.float().clamp(0.0, 1.0)  # start with the input padding mask
        val = H.new_zeros(B, 1)  # no count yet
        boolean = H.new_zeros(B, 1)  # no boolean decision yet

        stop_logits_all = []
        action_logits_all = []

        for step in range(self.max_steps):
            x = torch.cat([s, z], dim=-1)  # (B,2d)

            # operator mixture (softmax over ops)
            # ----- action gating over ops -----
            logits_action = self.to_action(x)  # (B, num_ops)
            probs = F.softmax(logits_action / self.temperature, dim=-1)  # (B, num_ops)

            # ----- run all typed-state ops -----
            # maintain typed scalars; start the first step with neutral values
            if step == 0:
                mask_float = mask.float().clamp(0.0, 1.0)  # (B, T)
                val = H.new_zeros(B, 1)  # (B, 1)
                boolean = H.new_zeros(B, 1)  # (B, 1)

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

            # ----- mix operator outputs -----
            mix_mask = torch.einsum("btn,bn->bt", M, probs)  # (B, T)
            mix_val = torch.einsum("bin,bn->bi", V, probs)  # (B, 1)
            mix_bool = torch.einsum("bin,bn->bi", Bv, probs)  # (B, 1)

            # monotone mask update (cannot increase selection)
            mask_float = torch.minimum(mask_float, mix_mask).clamp(0.0, 1.0)
            val = mix_val
            boolean = mix_bool

            # ----- recompute pooled state s from updated mask -----
            H_phi = self.phi(H)  # (B, T, d)
            denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            s = (H_phi * mask_float.unsqueeze(-1)).sum(dim=1) / denom  # (B, d)

            # STOP head
            stop_logit = self.to_stop(torch.cat([s, z], dim=-1))  # (B,1)
            stop_logits_all.append(stop_logit)
            action_logits_all.append(logits_action.unsqueeze(1))  # (B,1,num_ops)

            if (not self.training) and (torch.sigmoid(stop_logit) > 0.5).all():
                break

        stop_logits = torch.cat(stop_logits_all, dim=1)  # (B, steps_done)
        action_logits = torch.cat(action_logits_all, dim=1)  # (B, steps_done, num_ops)

        # pad to max_steps if we broke early in eval
        if stop_logits.size(1) < self.max_steps:
            pad_s = self.max_steps - stop_logits.size(1)
            stop_logits = torch.cat(
                [stop_logits, stop_logits.new_zeros(B, pad_s)], dim=1
            )
            action_logits = torch.cat(
                [
                    action_logits,
                    action_logits.new_zeros(B, pad_s, action_logits.size(-1)),
                ],
                dim=1,
            )

        # broadcast final state to tokens
        s_rep = s.unsqueeze(1).expand(B, T, d)
        H_reasoned = H + self.broadcast(torch.cat([H, s_rep], dim=-1))

        return H_reasoned, s, stop_logits, action_logits
