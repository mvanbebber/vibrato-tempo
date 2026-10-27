"""Stage 3: f0 tracking on the isolated stem (torchcrepe, or pYIN fallback).

Raw f0 and periodicity are cached unthresholded, so voicing and note
parameters can be tuned without re-running the tracker."""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import audio
from .config import Config, params_hash, reuse_candidates
from .corpus import Song, atomic_write, cache_dir
from .hardware import free_memory, pick_device, run_with_fallback
from .separate import find_stem

log = logging.getLogger(__name__)
CREPE_SR = 16000


def pitch_source(cfg: Config, song: Song) -> tuple[Path, str] | None:
    """(audio path, source tag) to track: the cached stem, or the mix if
    separation is disabled. None if the stem has not been separated yet."""
    if not cfg["separation"]["enabled"]:
        return song.path, "mix"
    found = find_stem(cfg, song)
    if found is None:
        return None
    path, model = found
    return path, f"{model}-{path.stem}"


def _key(pc: dict, model: str | None) -> str:
    keep = {k: pc[k] for k in ("method", "hop_ms", "fmin_hz", "fmax_hz")}
    if pc["method"] == "crepe":
        keep["crepe_model"] = model
    return f"{pc['method']}{'-' + model if model else ''}-{params_hash(keep, n=8)}"


def pitch_path(cfg: Config, song: Song, source_tag: str, model: str | None = None) -> Path:
    pc = cfg["pitch"]
    if pc["method"] == "crepe" and model is None:
        model = pc["crepe_model"]
    return cache_dir(cfg, "pitch", song) / f"{source_tag}__{_key(pc, model)}.npz"


def find_pitch(cfg: Config, song: Song) -> Path | None:
    src = pitch_source(cfg, song)
    if src is None:
        return None
    pc = cfg["pitch"]
    models = [None]
    if pc["method"] == "crepe":
        models = reuse_candidates(pc["crepe_model"], pc["reuse_crepe_models"])
    for m in models:
        p = pitch_path(cfg, song, src[1], m)
        if p.exists():
            return p
    return None


def load_track(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _save(path: Path, t, f0, periodicity, **meta):
    arrays = dict(t=t.astype(np.float64), f0=f0.astype(np.float32),
                  periodicity=periodicity.astype(np.float32),
                  **{k: np.array(v) for k, v in meta.items()})
    atomic_write(path, lambda p: np.savez_compressed(p, **arrays), suffix=".npz")


# ---------------------------------------------------------------- CREPE

class Crepe:
    def __init__(self, cfg: Config):
        import torchcrepe  # noqa: F401
        self.pc = cfg["pitch"]
        self.device = pick_device(cfg["hardware"]["device"])
        log.info("torchcrepe %s on %s", self.pc["crepe_model"], self.device)

    def _predict(self, x, device):
        import torch
        import torchcrepe
        hop = int(round(CREPE_SR * self.pc["hop_ms"] / 1000))
        # weighted_argmax keeps sub-bin resolution; plain viterbi snaps to
        # 20-cent bins, which would quantise vibrato depth.
        decoder = getattr(torchcrepe.decode, "weighted_viterbi", torchcrepe.decode.weighted_argmax)
        with torch.no_grad():
            f0, per = torchcrepe.predict(torch.from_numpy(x)[None], CREPE_SR, hop,
                                         self.pc["fmin_hz"], self.pc["fmax_hz"],
                                         self.pc["crepe_model"], decoder=decoder,
                                         return_periodicity=True,
                                         batch_size=int(self.pc["batch_size"]),
                                         device=device, pad=True)
        return f0[0].cpu().numpy(), per[0].cpu().numpy(), hop

    def track(self, path: Path, out: Path, source_tag: str):
        x, _ = audio.load(path, sr=CREPE_SR, mono=True)
        f0, per, hop = run_with_fallback(lambda d: self._predict(x, d), self.device,
                                         f"crepe on {path.name}")
        t = np.arange(len(f0)) * hop / CREPE_SR
        _save(out, t, f0, per, method="crepe", model=self.pc["crepe_model"], source=source_tag)

    def close(self):
        free_memory()


# ---------------------------------------------------------------- pYIN

def _pyin_job(args):
    import librosa
    path, out, pc, source_tag = args
    sr = 22050
    x, _ = audio.load(Path(path), sr=sr, mono=True)
    hop = int(round(sr * pc["hop_ms"] / 1000))
    f0, _, vprob = librosa.pyin(x, fmin=pc["fmin_hz"], fmax=pc["fmax_hz"], sr=sr,
                                hop_length=hop, frame_length=2048)
    t = librosa.times_like(f0, sr=sr, hop_length=hop)
    _save(Path(out), t, np.nan_to_num(f0), vprob, method="pyin", model="", source=source_tag)
    return out


# ---------------------------------------------------------------- stage

def run(cfg: Config, songs: list[Song], force: bool = False) -> None:
    jobs = []
    for s in songs:
        src = pitch_source(cfg, s)
        if src is None:
            log.warning("pitch: %s has no separated stem yet; skipping", s.song_id)
            continue
        if force or find_pitch(cfg, s) is None:
            jobs.append((s, src, pitch_path(cfg, s, src[1])))
    log.info("pitch: %d to do", len(jobs))
    if not jobs:
        return
    pc = cfg["pitch"]
    if pc["method"] == "crepe":
        tracker = Crepe(cfg)
        try:
            for i, (s, (path, tag), out) in enumerate(jobs, 1):
                log.info("[%d/%d] crepe %s", i, len(jobs), s.song_id)
                tracker.track(path, out, tag)
        finally:
            tracker.close()
    elif pc["method"] == "pyin":
        args = [(str(path), str(out), pc, tag) for s, (path, tag), out in jobs]
        if cfg.workers > 1:
            with ProcessPoolExecutor(cfg.workers) as ex:
                list(ex.map(_pyin_job, args))
        else:
            for a in args:
                _pyin_job(a)
    else:
        raise ValueError(f"unknown pitch method {pc['method']!r}")
