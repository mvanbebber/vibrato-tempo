"""Load config.yaml, resolve the hardware profile, and expose project paths."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def total_ram_gb() -> float:
    import psutil
    return psutil.virtual_memory().total / 1024**3


@dataclass
class Config:
    raw: dict            # merged config (profile applied)
    root: Path
    profile: str

    def __getitem__(self, key):
        return self.raw[key]

    def path(self, key: str) -> Path:
        """Absolute path for a paths.* entry (stored relative to the project root)."""
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else self.root / p

    def rel(self, p: Path) -> str:
        """Path relative to the project root, for anything written to CSV/JSON."""
        try:
            return str(Path(p).resolve().relative_to(self.root))
        except ValueError:
            return str(p)

    @property
    def workers(self) -> int:
        return max(1, int(self.raw.get("workers", 1)))


def load_config(path: str | Path | None = None, profile: str | None = None,
                overrides: dict | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    root = path.resolve().parent

    choice = profile or os.environ.get("VIBTEMPO_PROFILE") or raw["hardware"]["profile"]
    if choice == "auto":
        choice = "light" if total_ram_gb() < raw["hardware"]["light_below_gb"] else "full"
    if choice not in raw.get("profiles", {}):
        raise ValueError(f"unknown hardware profile {choice!r}")

    merged = _deep_merge(raw, raw["profiles"][choice])
    if overrides:
        merged = _deep_merge(merged, overrides)
    return Config(raw=merged, root=root, profile=choice)


def reuse_candidates(configured: str, ranked: list[str]) -> list[str]:
    """Models whose cached results may stand in for `configured`: itself, then
    the models ranked above it (the list is best first). A lower-quality
    result is never reused in place of a higher-quality one."""
    if configured not in ranked:
        return [configured]
    better = ranked[: ranked.index(configured)]
    return [configured] + better


def params_hash(*sections: dict, n: int = 10) -> str:
    """Short stable hash of parameter dicts, used in cache keys."""
    blob = json.dumps(sections, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:n]
