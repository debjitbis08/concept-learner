from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from .train import ConceptLearner, TrainConfig


@dataclass
class AnalogyResult:
    prediction: Any
    scores: Dict[str, float]
    concepts: Dict[str, Any]


class ConceptAPI:
    """
    Lightweight API around the ConceptLearner for inference and tooling.
    """

    def __init__(self, model: ConceptLearner, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.device = device

    @classmethod
    def load(cls, ckpt_path: str, device: str = "cpu") -> "ConceptAPI":
        cfg = TrainConfig(device=device)
        model = ConceptLearner(cfg)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model", state))
        return cls(model, device=device)

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            return self.model.encode(tokens.to(self.device), mask.to(self.device) if mask is not None else None)

    def complete_analogy(self, A: Any, B: Any, C: Any) -> AnalogyResult:
        # Placeholder: wire up using EpisodeGenerator rendering + model.analogy later.
        return AnalogyResult(prediction=None, scores={}, concepts={})

    def explain(self, x: Any) -> Dict[str, Any]:
        # Placeholder: return nearest codes, prototypes, etc.
        return {}

    def record_feedback(self, A: Any, B: Any, C: Any, D: Any, ok: bool) -> None:
        # Placeholder: append to a small buffer for replay learning.
        return None

