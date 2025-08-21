from __future__ import annotations

from typing import Iterable

import torch


class EMA:
    """
    Simple EMA helper for model parameters.
    Usage:
        ema = EMA(model, decay=0.999)
        for step ...: ema.update(model)
        # swap-in for eval if desired:
        ema.apply_to(model); eval(...); ema.restore(model)
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        self.backup = None

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            i = 0
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                self.shadow[i].data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
                i += 1

    def apply_to(self, model: torch.nn.Module) -> None:
        self.backup = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            i = 0
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                p.data.copy_(self.shadow[i].data)
                i += 1

    def restore(self, model: torch.nn.Module) -> None:
        if self.backup is None:
            return
        with torch.no_grad():
            i = 0
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                p.data.copy_(self.backup[i].data)
                i += 1
        self.backup = None

    # Serialization helpers
    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": [t.clone().cpu() for t in self.shadow],
        }

    def load_state_dict(self, state) -> None:
        self.decay = float(state.get("decay", self.decay))
        sh = state.get("shadow", None)
        if sh is not None:
            # Move to current device of first param
            device = None
            for t in self.shadow:
                device = t.device
                break
            self.shadow = [t.to(device or "cpu") for t in sh]
