"""Validation: agreement between the tool and human raters.

Each ratings CSV holds one rater: columns `song_id`, a rate column (`rate_hz`,
or `rate_cpm` / `rate_bpm` in cycles per minute), and optionally `time_s` (a
time inside the rated note). Files with `time_s` are compared note by note; all
files are also compared at song level (median rate per song)."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config

TOOL = "tool"


# ---------------------------------------------------------------- ICC

def icc(Y: np.ndarray, alpha: float = 0.05) -> dict:
    """ICC(1,1), ICC(2,1) and ICC(3,1) with 95% CIs for an n-subjects x k-raters
    matrix (Shrout & Fleiss 1979; CIs per McGraw & Wong 1996). ICC(2,1) — two-way
    random effects, absolute agreement, single rater — is the one to report when
    the tool is meant to stand in for a human rater."""
    Y = np.asarray(Y, float)
    n, k = Y.shape
    if n < 3 or k < 2:
        return {}
    g = Y.mean()
    ssr = k * ((Y.mean(1) - g) ** 2).sum()
    ssc = n * ((Y.mean(0) - g) ** 2).sum()
    sse = ((Y - g) ** 2).sum() - ssr - ssc
    msr, msc = ssr / (n - 1), ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    msw = (ssc + sse) / (n * (k - 1))
    q = 1 - alpha / 2
    out = {}

    icc1 = (msr - msw) / (msr + (k - 1) * msw)
    f1 = msr / msw
    fl, fu = f1 / stats.f.ppf(q, n - 1, n * (k - 1)), f1 * stats.f.ppf(q, n * (k - 1), n - 1)
    out["ICC1"] = (icc1, (fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1))

    icc2 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    a = k * icc2 / (n * (1 - icc2))
    b = 1 + k * icc2 * (n - 1) / (n * (1 - icc2))
    v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    fl, fu = stats.f.ppf(q, n - 1, v), stats.f.ppf(q, v, n - 1)
    lo = n * (msr - fl * mse) / (fl * (k * msc + (k * n - k - n) * mse) + n * msr)
    hi = n * (fu * msr - mse) / (k * msc + (k * n - k - n) * mse + n * fu * msr)
    out["ICC2"] = (icc2, lo, hi)

    icc3 = (msr - mse) / (msr + (k - 1) * mse)
    f3 = msr / mse
    fl, fu = f3 / stats.f.ppf(q, n - 1, (n - 1) * (k - 1)), f3 * stats.f.ppf(q, (n - 1) * (k - 1), n - 1)
    out["ICC3"] = (icc3, (fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1))
    return out


def pair_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = b - a
    ic = icc(np.column_stack([a, b]))
    row = {"n": len(a)}
    for name in ("ICC2", "ICC3"):
        if name in ic:
            row[name], row[name + "_lo"], row[name + "_hi"] = ic[name]
    if len(a) >= 3:
        row["pearson_r"] = stats.pearsonr(a, b)[0]
        row["spearman_rho"] = stats.spearmanr(a, b)[0]
    row.update(bias=d.mean(), sd_diff=d.std(ddof=1) if len(d) > 1 else np.nan,
               mae=np.abs(d).mean())
    row["loa_lo"] = row["bias"] - 1.96 * row["sd_diff"]
    row["loa_hi"] = row["bias"] + 1.96 * row["sd_diff"]
    return row


# ---------------------------------------------------------------- loading

def load_ratings(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "rate_hz" in df:
        df["rate_hz"] = df["rate_hz"].astype(float)
    elif "rate_cpm" in df or "rate_bpm" in df:
        df["rate_hz"] = df.get("rate_cpm", df.get("rate_bpm")).astype(float) / 60
    else:
        raise ValueError(f"{path}: needs rate_hz, rate_cpm or rate_bpm")
    if "song_id" not in df:
        raise ValueError(f"{path}: needs a song_id column")
    df["song_id"] = df["song_id"].astype(str).map(lambda s: Path(s).stem)
    df["rater"] = name
    keep = ["song_id", "rater", "rate_hz"] + (["time_s"] if "time_s" in df else [])
    return df[keep].dropna(subset=["rate_hz"])


def match_notes(rated: pd.DataFrame, tool_notes: pd.DataFrame, tol: float) -> pd.DataFrame:
    """For each rated note, the tool note containing time_s (else the nearest
    within tol seconds). Returns columns tool (rate Hz, NaN if unmatched) and
    key, which identifies the note so different raters' ratings line up."""
    rates, keys = [], []
    for _, r in rated.iterrows():
        cand = tool_notes[tool_notes["song_id"] == r["song_id"]]
        rate, key = np.nan, f"{r['song_id']}@{round(r['time_s'] / tol)}"
        if not cand.empty:
            dist = np.maximum(0, np.maximum(cand["start_s"] - r["time_s"], r["time_s"] - cand["end_s"]))
            i = dist.idxmin()
            if dist[i] <= tol:
                rate, key = cand.at[i, "rate_hz"], f"{r['song_id']}#{int(cand.at[i, 'note'])}"
        rates.append(rate)
        keys.append(key)
    return pd.DataFrame({"tool": rates, "key": keys}, index=rated.index)


# ---------------------------------------------------------------- report

def _agreement(wide: pd.DataFrame, level: str) -> list[dict]:
    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        sub = wide[[a, b]].dropna()
        if len(sub) >= 2:
            rows.append({"level": level, "rater_a": a, "rater_b": b,
                         **pair_stats(sub[a].values, sub[b].values)})
    groups = [("all raters", list(wide.columns))]
    humans = [c for c in wide.columns if c != TOOL]
    if len(humans) >= 2:
        groups.append(("humans only", humans))
    for label, cols in groups:
        if len(cols) > 2:
            sub = wide[cols].dropna()
            ic = icc(sub.values)
            if ic:
                rows.append({"level": level, "rater_a": label, "rater_b": "", "n": len(sub),
                             "ICC2": ic["ICC2"][0], "ICC2_lo": ic["ICC2"][1], "ICC2_hi": ic["ICC2"][2],
                             "ICC3": ic["ICC3"][0], "ICC3_lo": ic["ICC3"][1], "ICC3_hi": ic["ICC3"][2]})
    return rows


def run(cfg: Config, rating_files: list[Path], names: list[str] | None,
        notes_all: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    vcfg = cfg["validation"]
    min_conf = vcfg["min_confidence"]
    names = names or [Path(p).stem for p in rating_files]
    ratings = [load_ratings(Path(p), n) for p, n in zip(rating_files, names)]
    tool_notes = notes_all[notes_all["vibrato"].astype(bool)]
    if min_conf is not None:
        tool_notes = tool_notes[tool_notes["confidence"] >= min_conf]

    rows, coverage = [], []
    # Note level: only raters who gave time stamps.
    timed = [r for r in ratings if "time_s" in r]
    if timed:
        tol = vcfg["match_tolerance_s"]
        frames = []
        for r in timed:
            r = pd.concat([r, match_notes(r, tool_notes, tol)], axis=1)
            coverage.append({"rater": r["rater"].iat[0], "rated_notes": len(r),
                             "matched_to_tool_vibrato_note": int(r["tool"].notna().sum())})
            frames.append(r)
        long = pd.concat(frames)
        wide = long.pivot_table(index="key", columns="rater", values="rate_hz", aggfunc="mean")
        wide[TOOL] = long.groupby("key")["tool"].first()
        rows += _agreement(wide[[TOOL] + [c for c in wide.columns if c != TOOL]], "note")

    # Song level: each rater's median per song vs the tool's song median.
    per_song = pd.concat(ratings).groupby(["song_id", "rater"])["rate_hz"].median().unstack()
    tool_song = summary.set_index("song_id").get("rate_hz_median")
    if tool_song is not None:
        per_song[TOOL] = tool_song.reindex(per_song.index)
    rows += _agreement(per_song[[TOOL] + [c for c in per_song.columns if c != TOOL]], "song")

    res = pd.DataFrame(rows)
    out = cfg.path("output_dir") / "validation"
    out.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(out / "agreement.csv", index=False)
    per_song.round(4).to_csv(out / "song_level_values.csv")
    if timed:
        long.drop(columns="key").round(4).to_csv(out / "note_level_matches.csv", index=False)
    _plots(per_song, out)
    report = _report(res, coverage, out)
    return res, report


def _plots(per_song: pd.DataFrame, out: Path):
    from .plots import INK_2, MUTED, SERIES, SURFACE, _style
    import matplotlib.pyplot as plt
    humans = [c for c in per_song.columns if c != TOOL]
    if TOOL not in per_song or not humans:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=200, facecolor=SURFACE)
    for ax in axes:
        _style(ax)
    markers = ["o", "s", "^", "D"]
    for i, h in enumerate(humans):
        sub = per_song[[h, TOOL]].dropna() * 60
        m = markers[i % len(markers)]
        axes[0].scatter(sub[h], sub[TOOL], s=30, marker=m, color=SERIES, alpha=0.6 + 0.4 * (i == 0),
                        edgecolor=SURFACE, label=h)
        mean, diff = (sub[h] + sub[TOOL]) / 2, sub[TOOL] - sub[h]
        axes[1].scatter(mean, diff, s=30, marker=m, color=SERIES, edgecolor=SURFACE, label=h)
        if len(diff) > 1:
            b, sd = diff.mean(), diff.std(ddof=1)
            for y, ls in ((b, "-"), (b - 1.96 * sd, "--"), (b + 1.96 * sd, "--")):
                axes[1].axhline(y, color=MUTED, lw=0.8, ls=ls)
    lim = [np.nanmin(per_song.values) * 60 * 0.95, np.nanmax(per_song.values) * 60 * 1.05]
    axes[0].plot(lim, lim, color=MUTED, lw=0.8)
    axes[0].set(xlim=lim, ylim=lim, xlabel="Human rating (cycles/min)", ylabel="Tool (cycles/min)")
    axes[0].set_title("Song-level medians", loc="left", fontsize=10)
    axes[1].axhline(0, color=INK_2, lw=0.5)
    axes[1].set(xlabel="Mean of tool and human (cycles/min)", ylabel="Tool − human (cycles/min)")
    axes[1].set_title("Bland–Altman (lines: bias, ±1.96 SD)", loc="left", fontsize=10)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "agreement.png", facecolor=SURFACE)
    plt.close(fig)


def _report(res: pd.DataFrame, coverage: list[dict], out: Path) -> Path:
    lines = ["# Agreement report", "",
             "ICC2 = ICC(2,1), two-way random, absolute agreement, single rater.",
             "ICC3 = ICC(3,1), two-way mixed, consistency. Rates in Hz; bias = rater_b − rater_a.", ""]
    for level in ("note", "song"):
        sub = res[res["level"] == level] if len(res) else res
        if sub.empty:
            continue
        lines += [f"## {level.capitalize()} level", "",
                  "| comparison | n | ICC(2,1) [95% CI] | ICC(3,1) | r | bias Hz | MAE Hz |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            comp = f"{r['rater_a']} vs {r['rater_b']}" if r["rater_b"] else r["rater_a"]
            f = lambda k, p=3: "" if pd.isna(r.get(k)) else f"{r[k]:.{p}f}"
            lines.append(f"| {comp} | {int(r['n'])} | {f('ICC2')} [{f('ICC2_lo')}, {f('ICC2_hi')}] | "
                         f"{f('ICC3')} | {f('pearson_r')} | {f('bias')} | {f('mae')} |")
        lines.append("")
    if coverage:
        lines += ["## Coverage", "", "| rater | rated notes | matched to a tool vibrato note |", "|---|---|---|"]
        lines += [f"| {c['rater']} | {c['rated_notes']} | {c['matched_to_tool_vibrato_note']} |" for c in coverage]
    p = out / "report.md"
    p.write_text("\n".join(lines) + "\n")
    return p
