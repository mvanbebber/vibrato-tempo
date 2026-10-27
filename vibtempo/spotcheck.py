"""Spot-check: audio clip + pitch-curve plot for individual notes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import audio, notes as notes_mod, vibrato
from .config import Config
from .corpus import Song
from .pitch import find_pitch, load_track, pitch_source


def _fit_curve(t, cents, m, vc):
    """Trend + sinusoid evaluated on the note's frames, for overlaying."""
    tt = t - t[0]
    q = vibrato._trend_basis(len(t), int(vc["trend_degree"]))
    X = np.column_stack([q, np.sin(2 * np.pi * m["rate_hz"] * tt), np.cos(2 * np.pi * m["rate_hz"] * tt)])
    beta, *_ = np.linalg.lstsq(X, cents, rcond=None)
    return q @ beta[: q.shape[1]], X @ beta


def export(cfg: Config, song: Song, note_row: pd.Series, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plots import INK, INK_2, MUTED, SERIES, SURFACE, _style

    pad = cfg["spotcheck"]["pad_s"]
    t0 = max(0.0, note_row["start_s"] - pad)
    dur = note_row["end_s"] + pad - t0
    tag = f"{song.song_id}_note{int(note_row['note']):04d}_{note_row['start_s']:.2f}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    src = pitch_source(cfg, song)
    for label, path in (("solo", src[0] if src else None), ("mix", song.path)):
        if path is None:
            continue
        y, sr = audio.load(path, sr=None, mono=False, offset=t0, duration=dur)
        p = out_dir / f"{tag}_{label}.wav"
        audio.write(p, y, sr, subtype="PCM_16")
        written.append(p)

    tr = load_track(find_pitch(cfg, song))
    cents, hop_s = notes_mod.prepare(tr["t"], tr["f0"], tr["periodicity"], cfg["notes"], cfg["pitch"])
    t = tr["t"]
    win = (t >= t0) & (t <= t0 + dur)
    sel = (t >= note_row["start_s"] - 1e-6) & (t < note_row["end_s"] - 1e-6) & np.isfinite(cents)
    ref = np.nanmean(cents[sel])
    m = vibrato.fit(t[sel], cents[sel], cfg["vibrato"])
    trend, full = _fit_curve(t[sel], cents[sel], m, cfg["vibrato"])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.5), dpi=150, facecolor=SURFACE, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    for ax in (a1, a2):
        _style(ax)
        ax.axvspan(note_row["start_s"], note_row["end_s"], color="#eaf2fc", lw=0, zorder=0.5)
    a1.plot(t[win], cents[win] - ref, color=MUTED, lw=1, label="f0 (context)")
    a1.plot(t[sel], cents[sel] - ref, color=SERIES, lw=2, label="f0 (note)")
    a1.plot(t[sel], full - ref, color=INK, lw=1, ls="--", label="fit: trend + sinusoid")
    a1.set_ylabel(f"cents re {note_row['pitch_name']}")
    a1.legend(frameon=False, fontsize=8, loc="upper right")
    status = "vibrato" if note_row["vibrato"] else f"rejected: {note_row['reject_reason']}"
    a1.set_title(f"{song.song_id} · note {int(note_row['note'])} · {note_row['start_s']:.2f}s · "
                 f"{note_row['rate_hz']:.2f} Hz ({note_row['rate_cpm']:.0f}/min) · "
                 f"{note_row['depth_cents']:.0f} c p-p · conf {note_row['confidence']:.2f} · {status}",
                 loc="left", fontsize=9, color=INK)
    a2.plot(t[sel], cents[sel] - trend, color=SERIES, lw=2, label="detrended f0")
    a2.plot(t[sel], full - trend, color=INK, lw=1, ls="--", label="sinusoid")
    a2.axhline(0, color=INK_2, lw=0.5)
    a2.set_ylabel("cents")
    a2.set_xlabel("time in song (s)")
    a2.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    p = out_dir / f"{tag}.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    written.append(p)
    return written
