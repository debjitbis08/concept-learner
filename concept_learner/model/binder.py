from typing import Dict, Optional

import torch
import torch.nn as nn


class FastBinder(nn.Module):
    """
    Simple fast-weight label binder. Maintains EMA of label -> concept embedding.
    """

    def __init__(self, concept_dim: int, decay: float = 0.9):
        super().__init__()
        self.decay = decay
        self.store: Dict[int, torch.Tensor] = {}
        self.concept_dim = concept_dim

    @torch.no_grad()
    def bind(self, label: int, concept_embed: torch.Tensor):
        # concept_embed: (D,)
        if label in self.store:
            self.store[label] = self.decay * self.store[label] + (1 - self.decay) * concept_embed.detach().cpu()
        else:
            self.store[label] = concept_embed.detach().cpu()

    @torch.no_grad()
    def query(self, label: int) -> Optional[torch.Tensor]:
        return self.store.get(label, None)

    @torch.no_grad()
    def decay_all(self, factor: float = 0.99):
        for k in list(self.store.keys()):
            self.store[k] = factor * self.store[k]

