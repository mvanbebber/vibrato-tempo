"""Note segmentation: sustained, roughly steady-pitch regions of the f0 track."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d


def hz_to_cents(f0: np.ndarray) -> np.ndarray:
    """Cents on the MIDI scale (6900 = A4 = 440 Hz); NaN where f0 <= 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f0 > 0, 1200 * np.log2(f0 / 440.0) + 6900, np.nan)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index ranges where mask is True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def vibrato_free_contour(cents: np.ndarray, hop_s: float) -> np.ndarray:
    """Pitch contour with vibrato removed: two box filters of 250 ms and 167 ms
    have spectral nulls at 4, 6, 8 and 12 Hz, together suppressing 3-10 Hz
    modulation while following note-to-note steps."""
    out = cents
    for w_s in (0.250, 0.167):
        w = max(1, int(round(w_s / hop_s)))
        out = uniform_filter1d(out, w, mode="nearest")
    return out


def prepare(t: np.ndarray, f0: np.ndarray, periodicity: np.ndarray, nc: dict, pc: dict):
    """Voicing decision, gap bridging and glitch removal.
    Returns (cents with NaN where unvoiced, hop seconds)."""
    hop_s = float(np.median(np.diff(t)))
    cents = hz_to_cents(f0)
    voiced = (periodicity >= pc["voicing_threshold"]) & np.isfinite(cents)
    max_gap = int(round(nc["max_gap_ms"] / 1000 / hop_s))
    idx = np.arange(len(cents))
    for a, b in _runs(~voiced):
        if 0 < a and b < len(cents) and b - a <= max_gap:
            voiced[a:b] = True
            cents[a:b] = np.interp(idx[a:b], [a - 1, b], [cents[a - 1], cents[b]])
    cents[~voiced] = np.nan
    k = int(nc["median_filter_frames"])
    if k > 1:
        for a, b in _runs(voiced):
            if b - a >= k:
                cents[a:b] = median_filter(cents[a:b], size=k, mode="nearest")
    return cents, hop_s


def segment(cents: np.ndarray, hop_s: float, nc: dict) -> list[tuple[int, int]]:
    """Split voiced runs into notes wherever the vibrato-free contour leaves the
    current note's running median by more than split_cents; trim note edges and
    keep notes of at least min_note_ms."""
    min_len = int(np.ceil(nc["min_note_ms"] / 1000 / hop_s))
    trim = int(round(nc["edge_trim_ms"] / 1000 / hop_s))
    split = float(nc["split_cents"])
    notes = []
    for a, b in _runs(np.isfinite(cents)):
        if b - a < min_len:
            continue
        smooth = vibrato_free_contour(cents[a:b], hop_s)
        start = 0
        for i in range(1, len(smooth) + 1):
            if i == len(smooth) or abs(smooth[i] - np.median(smooth[start:i])) > split:
                s, e = a + start + trim, a + i - trim
                if e - s >= min_len:
                    notes.append((s, e))
                start = i
    return notes
