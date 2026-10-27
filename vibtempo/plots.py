"""Corpus plots: vibrato rate vs tempo with zone boundaries."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e6e5e0"
SERIES = "#2a78d6"
BAND = "#eaf2fc"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _zones(ax, cfg: Config, label_y: float):
    zones = cfg["zones"]
    edges = sorted({z["bpm_min"] for z in zones} | {z["bpm_max"] for z in zones})
    for e in edges:
        ax.axvline(e, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    for z in zones:
        mid = np.sqrt(z["bpm_min"] * z["bpm_max"])
        ax.text(mid, label_y, z["name"], ha="center", va="bottom", fontsize=8, color=INK_2)


def rate_vs_tempo(cfg: Config, summary: pd.DataFrame, out: Path) -> Path:
    df = summary.dropna(subset=["bpm", "rate_cpm_median"]) if "rate_cpm_median" in summary else summary.iloc[:0]
    lo, hi = cfg["target_band_bpm"]
    zones = cfg["zones"]
    xmin = min(z["bpm_min"] for z in zones) * 0.85
    xmax = max(z["bpm_max"] for z in zones) * 1.12

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.set_xscale("log", base=2)
    ax.axhspan(lo, hi, color=BAND, zorder=0.5, lw=0)
    ax.text(xmax / 1.02, hi, f"target band {lo}–{hi}", ha="right", va="bottom",
            fontsize=8, color=INK_2)
    # The theory's prediction inside each zone: vibrato = subdivision x tempo.
    for z in zones:
        x = np.array([z["bpm_min"], z["bpm_max"]])
        ax.plot(x, x * z["subdivision"], color=INK_2, lw=1, alpha=0.6, zorder=1.5)
        ax.text(x[1] / 1.03, x[1] * z["subdivision"], f"×{z['subdivision']:g}",
                fontsize=7, color=INK_2, ha="right", va="bottom")
    if len(df):
        ok = df["status"] == "ok"
        yerr = np.vstack([df["rate_cpm_median"] - df["rate_hz_q1"] * 60,
                          df["rate_hz_q3"] * 60 - df["rate_cpm_median"]])
        ax.errorbar(df["bpm"], df["rate_cpm_median"], yerr=yerr, fmt="none",
                    ecolor=SERIES, elinewidth=1, alpha=0.5, zorder=2)
        ax.scatter(df.loc[ok, "bpm"], df.loc[ok, "rate_cpm_median"], s=36, color=SERIES,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3, label="song (median, IQR)")
        if (~ok).any():
            ax.scatter(df.loc[~ok, "bpm"], df.loc[~ok, "rate_cpm_median"], s=36,
                       facecolor="none", edgecolor=SERIES, linewidth=1.2, zorder=3,
                       label=f"< {cfg['summary']['min_notes_per_song']} vibrato notes")
        if len(df) <= 12:
            for _, r in df.iterrows():
                ax.annotate(r["song_id"], (r["bpm"], r["rate_cpm_median"]), xytext=(5, 4),
                            textcoords="offset points", fontsize=7, color=INK_2)
        ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK_2)
    ymax = max(hi * 1.6, float(df["rate_hz_q3"].max() * 60 * 1.1) if len(df) else 0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    _zones(ax, cfg, ymax * 0.955)
    ticks = [z["bpm_min"] for z in zones] + [zones[-1]["bpm_max"]]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Song tempo (BPM)")
    ax.set_ylabel("Vibrato rate (cycles per minute)")
    ax.set_title(f"Vibrato rate vs. tempo  (n = {len(df)} songs)", loc="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def per_song_rates(cfg: Config, notes: pd.DataFrame, summary: pd.DataFrame, out: Path) -> Path:
    """Per-note vibrato rates for each song, songs ordered by tempo."""
    v = notes[notes["vibrato"].astype(bool)]
    order = summary.dropna(subset=["bpm"]).sort_values("bpm")
    order = order[order["song_id"].isin(v["song_id"])]
    lo, hi = cfg["target_band_bpm"]
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(order) + 1.2)), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.axvspan(lo, hi, color=BAND, zorder=0.5, lw=0)
    rng = np.random.default_rng(0)
    for i, (_, r) in enumerate(order.iterrows()):
        x = v.loc[v["song_id"] == r["song_id"], "rate_cpm"]
        ax.scatter(x, i + rng.uniform(-0.15, 0.15, len(x)), s=10, color=SERIES, alpha=0.55,
                   linewidth=0, zorder=2)
        ax.plot([np.median(x)] * 2, [i - 0.3, i + 0.3], color=INK, lw=2, zorder=3)
    ax.set_yticks(range(len(order)), [f"{s}  ({b:.0f} BPM)" for s, b in zip(order["song_id"], order["bpm"])])
    ax.set_xlabel("Vibrato rate per note (cycles per minute); bar = median")
    ax.set_title("Per-note vibrato rates, songs ordered by tempo", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def make_all(cfg: Config, notes: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    d = cfg.path("output_dir") / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return [rate_vs_tempo(cfg, summary, d / "rate_vs_tempo.png"),
            per_song_rates(cfg, notes, summary, d / "per_song_rates.png")]
