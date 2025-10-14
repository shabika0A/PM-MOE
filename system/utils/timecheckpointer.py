# =====================================================================
# File: system/utils/timecheckpointer.py
# =====================================================================
from __future__ import annotations
import os, io, time, json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple
import torch, random, numpy as np

def _atomic_write_bytes(data: bytes, path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

@dataclass
class TrainProgress:
    epoch: int = 0           # 0-based
    batch_idx: int = -1      # last COMPLETED batch in current epoch
    global_step: int = 0     # completed optimizer steps
    stage: str = "train"     # optional tag

@dataclass
class SaveConfig:
    ckpt_dir: str
    run_name: str = "default"
    save_every_secs: int = 300   # 5 minutes
    keep_last: int = 5
    with_amp: bool = True
    verbose: bool = True

class Checkpointer:
    """Time-based autosave for Jupyter: save every N seconds, resume mid-batch."""
    def __init__(self, cfg: SaveConfig):
        self.cfg = cfg
        self._last_saved_wall = 0.0
        os.makedirs(self._run_dir, exist_ok=True)

    @property
    def _run_dir(self) -> str:
        return os.path.join(self.cfg.ckpt_dir, self.cfg.run_name)

    def latest(self) -> Optional[str]:
        files = [f for f in os.listdir(self._run_dir) if f.endswith(".pt")]
        if not files: return None
        files.sort()
        return os.path.join(self._run_dir, files[-1])

    def _rotate(self) -> None:
        files = sorted(os.path.join(self._run_dir, f) for f in os.listdir(self._run_dir) if f.endswith(".pt"))
        while len(files) > self.cfg.keep_last:
            old = files.pop(0)
            try:
                os.remove(old)
                meta = old + ".json"
                if os.path.exists(meta): os.remove(meta)
            except FileNotFoundError:
                pass

    def should_save(self) -> bool:
        now = time.time()
        return (self._last_saved_wall == 0.0) or (now - self._last_saved_wall >= self.cfg.save_every_secs)

    @staticmethod
    def capture_rng() -> Dict[str, Any]:
        state = {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def restore_rng(rng: Dict[str, Any]) -> None:
        try:
            if "py" in rng: random.setstate(rng["py"])
            if "np" in rng: np.random.set_state(rng["np"])
            if "torch_cpu" in rng: torch.random.set_rng_state(rng["torch_cpu"])
            if torch.cuda.is_available() and "torch_cuda_all" in rng:
                torch.cuda.set_rng_state_all(rng["torch_cuda_all"])
        except Exception:
            pass  # tolerate partial restoration

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        scaler: Optional[torch.cuda.amp.GradScaler],
        progress: TrainProgress,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        fname = f"ep{progress.epoch:04d}_b{progress.batch_idx:06d}_gs{progress.global_step:09d}.pt"
        path = os.path.join(self._run_dir, fname)
        state = {
            "progress": asdict(progress),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if (scaler and self.cfg.with_amp) else None,
            "rng": self.capture_rng(),
            "extra": extra or {},
            "saved_at": time.time(),
            "torch": torch.__version__,
        }
        buf = io.BytesIO(); torch.save(state, buf); _atomic_write_bytes(buf.getvalue(), path)
        with open(path + ".json", "w", encoding="utf-8") as f:
            meta = {**state["progress"], "path": path, "saved_at": state["saved_at"]}
            json.dump(meta, f, indent=2)
        self._last_saved_wall = state["saved_at"]
        self._rotate()
        if self.cfg.verbose: print(f"[autosave] {path}")
        return path

    def load(
        self,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        scaler: Optional[torch.cuda.amp.GradScaler],
        path: Optional[str] = None,
        map_location: str | torch.device = "cpu",
    ) -> TrainProgress:
        if path is None:
            path = self.latest()
        if path is None:
            if self.cfg.verbose: print("[autosave] no checkpoint found; starting fresh")
            return TrainProgress()
        state = torch.load(path, map_location=map_location)
        model.load_state_dict(state["model"], strict=False)
        if optimizer and state.get("optimizer") is not None:
            try: optimizer.load_state_dict(state["optimizer"])
            except Exception: pass
        if scheduler and state.get("scheduler") is not None:
            try: scheduler.load_state_dict(state["scheduler"])
            except Exception: pass
        if scaler and state.get("scaler") is not None and self.cfg.with_amp:
            try: scaler.load_state_dict(state["scaler"])
            except Exception: pass
        self.restore_rng(state.get("rng", {}))
        prog = TrainProgress(**state["progress"])
        if self.cfg.verbose:
            print(f"[autosave] loaded {path} -> epoch={prog.epoch} batch={prog.batch_idx} step={prog.global_step}")
        # Reset wall clock so next save triggers in ~save_every_secs
        self._last_saved_wall = time.time()
        return prog

