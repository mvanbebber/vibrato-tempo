"""Audio loading and writing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load(path: Path, sr: int | None, mono: bool = True, offset: float = 0.0,
         duration: float | None = None) -> tuple[np.ndarray, int]:
    """Load audio as float32. Shape (n,) if mono else (channels, n)."""
    import librosa
    y, sr_out = librosa.load(str(path), sr=sr, mono=mono, offset=offset, duration=duration)
    if not mono and y.ndim == 1:
        y = np.stack([y, y])
    return y.astype(np.float32), sr_out


def write(path: Path, y: np.ndarray, sr: int, subtype: str = "PCM_24") -> None:
    from .corpus import atomic_write
    data = y.T if y.ndim == 2 else y
    atomic_write(Path(path), lambda p: sf.write(str(p), data, sr, subtype=subtype,
                                               format=Path(path).suffix[1:].upper()))
