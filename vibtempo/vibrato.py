"""Per-note vibrato measurement by sinusoid fitting.

For each candidate rate f, the note's pitch (cents) is regressed on a low-order
polynomial trend plus sin/cos at f. The rate is the f that best explains the
detrended pitch; confidence is the fraction of detrended variance it explains
(an R^2 in [0, 1]). Unlike autocorrelation or FFT peak picking, this works on
notes holding only ~1.5 vibrato cycles."""
from __future__ import annotations

import numpy as np


def _trend_basis(n: int, degree: int) -> np.ndarray:
    x = np.linspace(-1, 1, n)
    q, _ = np.linalg.qr(np.vander(x, degree + 1, increasing=True))
    return q  # orthonormal columns spanning the trend


def fit(t: np.ndarray, cents: np.ndarray, vc: dict) -> dict:
    t = t - t[0]
    n = len(t)
    q = _trend_basis(n, int(vc["trend_degree"]))
    r = cents - q @ (q.T @ cents)                 # detrended pitch
    ss_r = float(r @ r)
    freqs = np.arange(vc["rate_min_hz"], vc["rate_max_hz"] + 1e-9, vc["rate_step_hz"])

    def r2_at(fs: np.ndarray):
        ph = 2 * np.pi * np.outer(fs, t)
        s, c = np.sin(ph), np.cos(ph)
        s -= (s @ q) @ q.T                        # project the trend out (Frisch-Waugh)
        c -= (c @ q) @ q.T
        ss, cc, sc = (s * s).sum(1), (c * c).sum(1), (s * c).sum(1)
        sr, cr = s @ r, c @ r
        det = ss * cc - sc**2
        det = np.where(det > 1e-12, det, np.nan)
        a = (cc * sr - sc * cr) / det
        b = (ss * cr - sc * sr) / det
        explained = a * sr + b * cr
        return explained / ss_r if ss_r > 0 else np.zeros_like(fs), a, b

    r2, _, _ = r2_at(freqs)
    r2 = np.nan_to_num(r2)
    k = int(np.argmax(r2))
    at_edge = k == 0 or k == len(freqs) - 1
    f = freqs[k]
    if not at_edge:                              # parabolic refinement
        y0, y1, y2 = r2[k - 1], r2[k], r2[k + 1]
        denom = y0 - 2 * y1 + y2
        if denom < 0:
            f += 0.5 * vc["rate_step_hz"] * (y0 - y2) / denom
    conf, a, b = r2_at(np.array([f]))
    amp = float(np.hypot(a[0], b[0]))
    duration = n * float(np.median(np.diff(t))) if n > 1 else 0.0
    return {"rate_hz": float(f), "depth_cents": 2 * amp,   # peak-to-peak
            "confidence": float(np.clip(np.nan_to_num(conf[0]), 0, 1)),
            "cycles": float(f * duration), "at_edge": bool(at_edge),
            "residual_sd_cents": float(np.sqrt(ss_r / n))}


def judge(m: dict, vc: dict, min_conf: float | None = None) -> str:
    """'' if the note counts as vibrato, else the first failed criterion."""
    if m["at_edge"]:
        return "rate_at_search_limit"
    if m["cycles"] < vc["min_cycles"]:
        return "too_few_cycles"
    if m["confidence"] < (vc["min_confidence"] if min_conf is None else min_conf):
        return "low_confidence"
    if m["depth_cents"] < vc["min_depth_cents"]:
        return "too_shallow"
    if m["depth_cents"] > vc["max_depth_cents"]:
        return "too_deep"
    return ""
