"""Stage 4: notes + vibrato per song, then the per-note and per-song tables."""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import notes as notes_mod
from . import vibrato
from .config import Config, params_hash
from .corpus import Song, atomic_write, cache_dir
from .pitch import find_pitch, load_track
from .tempo import song_tempo

log = logging.getLogger(__name__)

NOTE_COLUMNS = ["song_id", "note", "start_s", "end_s", "duration_s", "pitch_hz",
                "pitch_name", "rate_hz", "rate_cpm", "depth_cents", "confidence",
                "cycles", "vibrato", "reject_reason"]


def analysis_key(cfg: Config) -> str:
    return params_hash(cfg["notes"], cfg["vibrato"], cfg["pitch"]["voicing_threshold"])


def analysis_path(cfg: Config, song: Song, pitch_file: Path) -> Path:
    return cache_dir(cfg, "analysis", song) / f"{pitch_file.stem}__{analysis_key(cfg)}.csv"


def analyse_track(track: dict, cfg_raw: dict) -> pd.DataFrame:
    import librosa
    nc, vc, pc = cfg_raw["notes"], cfg_raw["vibrato"], cfg_raw["pitch"]
    t = track["t"]
    cents, hop_s = notes_mod.prepare(t, track["f0"], track["periodicity"], nc, pc)
    rows = []
    for i, (a, b) in enumerate(notes_mod.segment(cents, hop_s, nc)):
        m = vibrato.fit(t[a:b], cents[a:b], vc)
        mean_c = float(np.mean(cents[a:b]))
        hz = 440.0 * 2 ** ((mean_c - 6900) / 1200)
        reason = vibrato.judge(m, vc)
        rows.append({"note": i, "start_s": t[a], "end_s": t[b - 1] + hop_s,
                     "duration_s": (b - a) * hop_s, "pitch_hz": hz,
                     "pitch_name": librosa.hz_to_note(hz), "rate_hz": m["rate_hz"],
                     "rate_cpm": m["rate_hz"] * 60, "depth_cents": m["depth_cents"],
                     "confidence": m["confidence"], "cycles": m["cycles"],
                     "vibrato": reason == "", "reject_reason": reason})
    return pd.DataFrame(rows, columns=NOTE_COLUMNS[1:])


def _job(args):
    pitch_file, out, cfg_raw = args
    df = analyse_track(load_track(Path(pitch_file)), cfg_raw)
    atomic_write(Path(out), lambda p: df.to_csv(p, index=False))
    return out


def run(cfg: Config, songs: list[Song], force: bool = False) -> None:
    jobs = []
    for s in songs:
        pf = find_pitch(cfg, s)
        if pf is None:
            log.warning("analysis: %s has no pitch track yet; skipping", s.song_id)
            continue
        out = analysis_path(cfg, s, pf)
        if force or not out.exists():
            jobs.append((str(pf), str(out), cfg.raw))
    log.info("analysis: %d to do", len(jobs))
    if cfg.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(min(cfg.workers, len(jobs))) as ex:
            list(ex.map(_job, jobs))
    else:
        for j in jobs:
            _job(j)


# ---------------------------------------------------------------- tables

def zone_for(bpm: float | None, zones: list[dict]) -> dict | None:
    if bpm is None or not np.isfinite(bpm):
        return None
    for z in zones:
        if z["bpm_min"] <= bpm < z["bpm_max"]:
            return z
    return None


def song_notes(cfg: Config, song: Song) -> pd.DataFrame | None:
    pf = find_pitch(cfg, song)
    if pf is None or not analysis_path(cfg, song, pf).exists():
        return None
    df = pd.read_csv(analysis_path(cfg, song, pf), keep_default_na=False,
                     na_values=[""], dtype={"reject_reason": str})
    df["reject_reason"] = df["reject_reason"].fillna("")
    df.insert(0, "song_id", song.song_id)
    return df


def provenance(cfg: Config, song: Song) -> dict:
    """Which stem model and pitch model produced this song's numbers. Mixing
    models across a corpus is a confound, so it is recorded per song."""
    pf = find_pitch(cfg, song)
    if pf is None:
        return {"stem_source": "", "pitch_track": ""}
    tag, key = pf.stem.split("__", 1)
    return {"stem_source": tag, "pitch_track": key}


def build_tables(cfg: Config, songs: list[Song]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_notes, summary = [], []
    lo, hi = cfg["target_band_bpm"]
    for s in songs:
        df = song_notes(cfg, s)
        tempo = song_tempo(cfg, s)
        row = {"song_id": s.song_id, "file": s.relpath,
               **{k: s.meta.get(k, "") for k in ("title", "artist", "soloist", "genre", "instrument")},
               "bpm": tempo["bpm"], "bpm_source": tempo["bpm_source"],
               "bpm_estimated": tempo["bpm_estimated"], **provenance(cfg, s)}
        if df is None:
            row["status"] = "not_analysed"
            summary.append(row)
            continue
        all_notes.append(df)
        v = df[df["vibrato"]]
        row.update(n_sustained_notes=len(df), n_vibrato_notes=len(v),
                   vibrato_fraction=len(v) / len(df) if len(df) else np.nan)
        if len(v):
            q1, med, q3 = np.percentile(v["rate_hz"], [25, 50, 75])
            row.update(rate_hz_median=med, rate_hz_q1=q1, rate_hz_q3=q3,
                       rate_cpm_median=med * 60, depth_cents_median=v["depth_cents"].median(),
                       confidence_median=v["confidence"].median())
            if row["bpm"]:
                ratio = med * 60 / row["bpm"]
                row.update(rate_to_tempo=ratio, log2_rate_to_tempo=np.log2(ratio))
            row["in_target_band"] = bool(lo <= med * 60 <= hi)
        z = zone_for(row["bpm"], cfg["zones"])
        row.update(zone=z["name"] if z else "outside_zones",
                   zone_subdivision=z["subdivision"] if z else np.nan)
        row["status"] = ("ok" if len(v) >= cfg["summary"]["min_notes_per_song"]
                         else "too_few_vibrato_notes")
        summary.append(row)
    notes_df = pd.concat(all_notes, ignore_index=True) if all_notes else pd.DataFrame(columns=NOTE_COLUMNS)
    return notes_df, pd.DataFrame(summary)


def write_tables(cfg: Config, songs: list[Song]) -> tuple[pd.DataFrame, pd.DataFrame]:
    notes_df, summary = build_tables(cfg, songs)
    out = cfg.path("output_dir")
    out.mkdir(parents=True, exist_ok=True)
    rnd = lambda d: d.round(4)
    atomic_write(out / "notes_all.csv", lambda p: rnd(notes_df).to_csv(p, index=False))
    atomic_write(out / "notes.csv", lambda p: rnd(notes_df[notes_df["vibrato"].astype(bool)]
                                               .drop(columns=["vibrato", "reject_reason"])).to_csv(p, index=False))
    atomic_write(out / "songs.csv", lambda p: rnd(summary).to_csv(p, index=False))
    return notes_df, summary
