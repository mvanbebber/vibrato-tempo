"""Device selection (MPS -> CPU fallback) and memory housekeeping."""
from __future__ import annotations

import gc
import logging
import os

# Must be set before torch is imported: lets ops MPS lacks run on the CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

log = logging.getLogger(__name__)


def pick_device(requested: str = "auto") -> str:
    import torch
    if requested in ("auto", "mps") and torch.backends.mps.is_available():
        return "mps"
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if requested not in ("auto", "cpu"):
        log.warning("device %r not available, using cpu", requested)
    return "cpu"


def free_memory() -> None:
    """Release cached GPU memory between stages (matters on 8 GB machines)."""
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_with_fallback(fn, device: str, what: str):
    """Run fn(device); if it fails on a GPU, retry once on the CPU."""
    try:
        return fn(device)
    except (RuntimeError, NotImplementedError) as e:
        if device == "cpu":
            raise
        log.warning("%s failed on %s (%s); retrying on cpu", what, device, str(e).splitlines()[0])
        free_memory()
        return fn("cpu")
