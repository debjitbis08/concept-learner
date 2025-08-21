import glob
import os
from typing import Any, Optional

import torch


class CheckpointManager:
    def __init__(self, dir: str = "checkpoints", keep: int = 5):
        self.dir, self.keep = dir, keep
        os.makedirs(dir, exist_ok=True)

    def save(self, step: int, model: torch.nn.Module, optim: Optional[torch.optim.Optimizer] = None, ema: Any = None, extra: Optional[dict] = None) -> str:
        state = {
            "step": step,
            "model": model.state_dict(),
            "optim": optim.state_dict() if optim else None,
            "ema": ema.state_dict() if ema else None,
            "extra": extra or {},
        }
        path = os.path.join(self.dir, f"ckpt_{step:07d}.pt")
        torch.save(state, path)
        self._gc()
        return path

    def load_latest(self, model: torch.nn.Module, optim: Optional[torch.optim.Optimizer] = None, ema: Any = None):
        files = sorted(glob.glob(os.path.join(self.dir, "ckpt_*.pt")))
        assert files, "No checkpoints found"
        ckpt = torch.load(files[-1], map_location="cpu")
        model.load_state_dict(ckpt["model"])  # type: ignore[arg-type]
        if optim and ckpt.get("optim") is not None:
            optim.load_state_dict(ckpt["optim"])  # type: ignore[arg-type]
        if ema and ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])  # type: ignore[arg-type]
        return ckpt["step"], ckpt.get("extra", {})

    def _gc(self) -> None:
        files = sorted(glob.glob(os.path.join(self.dir, "ckpt_*.pt")))
        for f in files[:-self.keep]:
            try:
                os.remove(f)
            except OSError:
                pass

