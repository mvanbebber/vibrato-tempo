"""Find audio files, join metadata, and manage the on-disk cache."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aif", ".aiff", ".ogg"}
META_COLUMNS = ["title", "artist", "soloist", "genre", "instrument", "bpm", "stem"]


@dataclass
class Song:
    song_id: str          # file name without extension; the join key everywhere
    path: Path
    relpath: str
    sha: str              # content hash (first 12 hex chars) -> cache key
    meta: dict = field(default_factory=dict)

    @property
    def bpm_override(self) -> float | None:
        v = self.meta.get("bpm")
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if v > 0 else None


# ---------------------------------------------------------------- atomic I/O

def atomic_write(path: Path, write_fn, suffix: str = "") -> None:
    """Write via a temp file + rename so an interrupted run never leaves a
    half-written cache entry behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=suffix or path.suffix)
    os.close(fd)
    try:
        write_fn(Path(tmp))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_json(path: Path, obj) -> None:
    atomic_write(path, lambda p: p.write_text(json.dumps(obj, indent=2, default=float)))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- hashing

def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hashes(cfg: Config, paths: list[Path]) -> dict[Path, str]:
    """Content hashes, memoised in cache/hashes.json by (relpath, size, mtime).
    The hash, not the path, keys the cache, so renaming or moving a file between
    machines does not invalidate stems or pitch tracks."""
    index_path = cfg.path("cache_dir") / "hashes.json"
    index = read_json(index_path) if index_path.exists() else {}
    out, changed = {}, False
    for p in paths:
        st = p.stat()
        key = cfg.rel(p)
        rec = index.get(key)
        if not rec or rec["size"] != st.st_size or abs(rec["mtime"] - st.st_mtime) > 1:
            rec = {"size": st.st_size, "mtime": st.st_mtime, "sha1": _sha1(p)}
            index[key] = rec
            changed = True
        out[p] = rec["sha1"][:12]
    if changed:
        write_json(index_path, index)
    return out


# ---------------------------------------------------------------- corpus

def load_metadata(cfg: Config) -> pd.DataFrame | None:
    p = cfg.path("metadata_csv")
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]
    if "file" in df.columns:
        df["song_id"] = df["file"].map(lambda s: Path(s.strip()).stem)
    elif "song_id" not in df.columns:
        raise ValueError(f"{p}: needs a 'file' or 'song_id' column")
    return df.set_index("song_id")


def discover(cfg: Config, only: list[str] | None = None) -> list[Song]:
    audio_dir = cfg.path("audio_dir")
    paths = sorted(p for p in audio_dir.rglob("*")
                   if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."))
    seen: dict[str, Path] = {}
    for p in paths:
        if p.stem in seen:
            raise ValueError(f"duplicate song id {p.stem!r}: {seen[p.stem]} and {p}")
        seen[p.stem] = p
    if only:
        missing = set(only) - set(seen)
        if missing:
            raise ValueError(f"not found in {audio_dir}: {sorted(missing)}")
        paths = [seen[s] for s in only]

    meta = load_metadata(cfg)
    if meta is not None:
        unknown = set(meta.index) - set(seen)
        if unknown:
            log.warning("metadata rows with no audio file: %s", sorted(unknown))
    hashes = file_hashes(cfg, paths)
    songs = []
    for p in paths:
        m = {}
        if meta is not None and p.stem in meta.index:
            m = {k: v for k, v in meta.loc[p.stem].to_dict().items() if v != ""}
        songs.append(Song(p.stem, p, cfg.rel(p), hashes[p], m))
    return songs


# ---------------------------------------------------------------- cache paths

def cache_dir(cfg: Config, kind: str, song: Song) -> Path:
    return cfg.path("cache_dir") / kind / song.sha
