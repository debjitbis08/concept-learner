from __future__ import annotations

import torch


def select_pair_columns_safe(
    logits: torch.Tensor,
    idx_a: int,
    idx_b: int,
    order: str = "ab",
) -> torch.Tensor:
    """
    Safely select two class columns from `logits` with fallback.

    - logits: (B, C) tensor
    - idx_a, idx_b: desired class indices
    - order: "ab" (return [a,b]) or "ba" (return [b,a])

    If either index is out of range, falls back to the last two columns in the
    requested order when C >= 2; otherwise returns zeros of shape (B, 2).
    """
    if not isinstance(logits, torch.Tensor) or logits.dim() != 2:
        raise ValueError("logits must be a (B,C) tensor")
    B, C = logits.shape
    # Both in-range: select directly
    if 0 <= idx_a < C and 0 <= idx_b < C:
        if order == "ab":
            return logits[:, [idx_a, idx_b]]
        else:
            return logits[:, [idx_b, idx_a]]
    # Fallback: use last two columns if available
    if C >= 2:
        if order == "ab":
            return logits[:, [C - 2, C - 1]]
        else:
            return logits[:, [C - 1, C - 2]]
    # Degenerate case: no valid columns; return zeros
    return logits.new_zeros(B, 2)

