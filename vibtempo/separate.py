"""Stage 1: isolate the soloist with Demucs. Stems are cached as mono FLAC."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import audio
from .config import Config, reuse_candidates
from .corpus import Song, cache_dir
from .hardware import free_memory, pick_device, run_with_fallback

log = logging.getLogger(__name__)


def target_stem(cfg: Config, song: Song) -> str:
    sc = cfg["separation"]
    if song.meta.get("stem"):
        return song.meta["stem"].strip().lower()
    inst = song.meta.get("instrument", "").strip().lower()
    return sc["stem_for_instrument"].get(inst, sc["default_stem"])


def stem_path(cfg: Config, song: Song, model: str, stem: str) -> Path:
    return cache_dir(cfg, "stems", song) / model / f"{stem}.flac"


def find_stem(cfg: Config, song: Song) -> tuple[Path, str] | None:
    """Cached stem from the configured model, else from a model ranked above
    it in separation.reuse_models. Returns (path, model) or None."""
    stem = target_stem(cfg, song)
    sc = cfg["separation"]
    for model in reuse_candidates(sc["model"], sc["reuse_models"]):
        p = stem_path(cfg, song, model, stem)
        if p.exists():
            return p, model
    return None


class Separator:
    """Holds one Demucs model in memory across songs; call close() to free it."""

    def __init__(self, cfg: Config):
        from demucs.pretrained import get_model
        self.cfg = cfg
        self.sc = cfg["separation"]
        self.device = pick_device(cfg["hardware"]["device"])
        self.model = get_model(self.sc["model"])
        self.model.eval()
        log.info("demucs %s on %s (segment %.1fs, shifts %d)", self.sc["model"],
                 self.device, self.sc["segment"], self.sc["shifts"])

    def _apply(self, wav, device):
        import torch
        from demucs.apply import apply_model
        segment = min(float(self.sc["segment"]), float(getattr(self.model, "segment", 1e9)))
        with torch.no_grad():
            # The mix stays on the CPU; Demucs moves one chunk at a time to the
            # device, so device memory use is bounded by the segment length.
            return apply_model(self.model, wav[None], shifts=int(self.sc["shifts"]),
                               split=True, overlap=float(self.sc["overlap"]),
                               segment=segment, device=device, progress=False)[0]

    def separate(self, song: Song) -> Path:
        import torch
        sr = self.model.samplerate
        y, _ = audio.load(song.path, sr=sr, mono=False)
        wav = torch.from_numpy(y[: self.model.audio_channels])
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        out = run_with_fallback(lambda d: self._apply((wav - mean) / std, d),
                                self.device, f"demucs on {song.song_id}")
        out = (out * std + mean).cpu().numpy()

        wanted = target_stem(self.cfg, song)
        if wanted not in self.model.sources:
            raise ValueError(f"{song.song_id}: stem {wanted!r} not produced by "
                             f"{self.sc['model']} (has {self.model.sources})")
        keep = self.model.sources if self.sc["keep_all_stems"] else [wanted]
        for name in keep:
            mono = out[self.model.sources.index(name)].mean(0)
            audio.write(stem_path(self.cfg, song, self.sc["model"], name), mono, sr)
        return stem_path(self.cfg, song, self.sc["model"], wanted)

    def close(self):
        del self.model
        free_memory()


def run(cfg: Config, songs: list[Song], force: bool = False) -> None:
    todo = [s for s in songs if force or find_stem(cfg, s) is None]
    log.info("separation: %d cached, %d to do", len(songs) - len(todo), len(todo))
    if not todo:
        return
    sep = Separator(cfg)
    try:
        for i, s in enumerate(todo, 1):
            log.info("[%d/%d] separating %s", i, len(todo), s.song_id)
            sep.separate(s)
            if cfg.profile == "light":
                free_memory()
    finally:
        sep.close()
