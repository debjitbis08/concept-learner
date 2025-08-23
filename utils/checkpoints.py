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
        self._gc(protected=path)
        return path

    def update_latest(self, target_path: str) -> str:
        """
        Create/refresh a 'latest.pt' pointer inside the ckpt dir.
        Tries to create a symlink and falls back to copying if symlinks are not supported.
        Returns the path of the latest file.
        """
        latest = os.path.join(self.dir, "latest.pt")
        # Try to create/replace a symlink first
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                try:
                    os.remove(latest)
                except OSError:
                    pass
            os.symlink(os.path.abspath(target_path), latest)
            return latest
        except Exception:
            # Fall back to copying the checkpoint to latest.pt
            import shutil

            try:
                shutil.copy2(target_path, latest)
            except Exception:
                # As a last resort, write a tiny pointer file with the path
                try:
                    with open(latest, "w") as f:
                        f.write(target_path)
                except Exception:
                    pass
            return latest

    def load_latest(self, model: torch.nn.Module, optim: Optional[torch.optim.Optimizer] = None, ema: Any = None):
        files = sorted(glob.glob(os.path.join(self.dir, "ckpt_*.pt")))
        assert files, "No checkpoints found"
        ckpt = torch.load(files[-1], map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)  # type: ignore[arg-type]
        if missing or unexpected:
            print(f"[ckpt] Warning: non-strict load. missing={len(missing)} unexpected={len(unexpected)}")
        if optim and ckpt.get("optim") is not None:
            optim.load_state_dict(ckpt["optim"])  # type: ignore[arg-type]
        if ema and ckpt.get("ema") is not None:
            ema.load_state_dict(ckpt["ema"])  # type: ignore[arg-type]
        return ckpt["step"], ckpt.get("extra", {})

    def _gc(self, protected: Optional[str] = None) -> None:
        files = sorted(glob.glob(os.path.join(self.dir, "ckpt_*.pt")))
        to_delete = files[:-self.keep]
        for f in to_delete:
            if protected is not None and os.path.abspath(f) == os.path.abspath(protected):
                continue
            try:
                os.remove(f)
            except OSError:
                pass
