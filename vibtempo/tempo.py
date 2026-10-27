"""Stage 2: song tempo from the full mix (overridable via metadata 'bpm')."""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import audio
from .config import Config, params_hash
from .corpus import Song, read_json, write_json

log = logging.getLogger(__name__)


def tempo_path(cfg: Config, song: Song) -> Path:
    return cfg.path("cache_dir") / "tempo" / f"{song.sha}__{params_hash(cfg['tempo'])}.json"


def estimate(path: Path, tc: dict) -> dict:
    import librosa
    y, sr = audio.load(path, sr=22050, mono=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    if tc["method"] == "librosa_beat":
        bpm, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr,
                                             start_bpm=tc["start_bpm"], units="time")
        bpm = float(np.atleast_1d(bpm)[0])
        # Median inter-beat interval is steadier than the global estimate when
        # the tracker has locked on.
        if len(beats) > 8:
            bpm = 60.0 / float(np.median(np.diff(beats)))
    elif tc["method"] == "librosa_tempo":
        bpm = float(librosa.feature.rhythm.tempo(onset_envelope=onset, sr=sr,
                                                  start_bpm=tc["start_bpm"])[0])
        beats = np.array([])
    else:
        raise ValueError(f"unknown tempo method {tc['method']!r}")
    # Octave errors are the main failure mode of tempo estimators; keep the
    # alternatives so they can be checked against the metadata override.
    return {"bpm": bpm, "bpm_half": bpm / 2, "bpm_double": bpm * 2,
            "n_beats": int(len(beats)), "method": tc["method"],
            "in_range": bool(tc["min_bpm"] <= bpm <= tc["max_bpm"])}


def _job(args):
    path, tc, out = args
    write_json(Path(out), estimate(Path(path), tc))
    return out


def run(cfg: Config, songs: list[Song], force: bool = False, pool: ProcessPoolExecutor | None = None):
    """Estimate tempo for songs lacking a cached result. With a pool, returns
    futures so tempo can run on the CPU while separation uses the GPU."""
    todo = [s for s in songs if force or not tempo_path(cfg, s).exists()]
    log.info("tempo: %d cached, %d to do", len(songs) - len(todo), len(todo))
    jobs = [(str(s.path), cfg["tempo"], str(tempo_path(cfg, s))) for s in todo]
    if pool is not None:
        return [pool.submit(_job, j) for j in jobs]
    for j in jobs:
        _job(j)
    return []


def song_tempo(cfg: Config, song: Song) -> dict:
    """Tempo used for analysis: metadata override if present, else the estimate."""
    p = tempo_path(cfg, song)
    est = read_json(p) if p.exists() else {}
    out = {"bpm_estimated": est.get("bpm")}
    if song.bpm_override:
        out.update(bpm=song.bpm_override, bpm_source="metadata")
    else:
        out.update(bpm=est.get("bpm"), bpm_source="estimated" if est else "missing")
    return out
