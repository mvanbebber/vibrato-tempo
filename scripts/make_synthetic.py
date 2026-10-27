"""Generate synthetic test songs with known tempo and per-note vibrato.

Writes audio/<song>.wav, metadata.csv, and test_data/ground_truth_notes.csv.
Also writes two *simulated* rater files (ground truth + noise) to exercise
validation mode; they are not real human ratings."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SR = 44100
ROOT = Path(__file__).resolve().parent.parent

SONGS = [  # song_id, bpm, mean vibrato Hz, mean depth (cents p-p), key MIDI, genre
    ("synth_slow", 66.0, 5.6, 90, 64, "test-ballad"),
    ("synth_mid", 112.0, 6.2, 70, 67, "test-pop"),
    ("synth_fast", 176.0, 6.8, 110, 62, "test-uptempo"),
]
FORMANTS = [(800, 80, 1.0), (1150, 90, 0.5), (2900, 120, 0.25)]  # vowel /a/


def voice(f0: np.ndarray, rng) -> np.ndarray:
    """Harmonic voice-like tone with an /a/ formant envelope."""
    phase = 2 * np.pi * np.cumsum(f0) / SR
    out = np.zeros_like(f0)
    fmean = np.mean(f0[f0 > 0])
    for h in range(1, int(5000 / fmean)):
        fh = h * fmean
        env = sum(g / (1 + ((fh - fc) / bw) ** 2) for fc, bw, g in FORMANTS) + 0.02
        out += env * h ** -0.7 * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
    return out


def melody(bpm, rate, depth, key, dur_s, rng):
    beat = 60 / bpm
    scale = np.array([0, 2, 4, 5, 7, 9, 11, 12])
    # Durations in beats: long notes carry vibrato, short ones should be skipped.
    choices = [0.5, 0.5, 1, 1.5, 2, 3, 4]
    t, notes = 1.0, []
    while t < dur_s - 2:
        nb = rng.choice(choices)
        d = nb * beat
        # Keep sustained notes at least ~0.9 s so most can carry vibrato.
        if d >= 0.6:
            d = max(d, 0.9)
        has_vib = d >= 0.6 and rng.random() > 0.2
        notes.append(dict(start_s=t, end_s=t + d * 0.92, midi=key + rng.choice(scale) - 12 * (key > 66),
                          vibrato=has_vib,
                          rate_hz=rate + rng.uniform(-0.35, 0.35) if has_vib else np.nan,
                          depth_cents=depth * rng.uniform(0.8, 1.2) if has_vib else np.nan))
        t += d + (0.12 if rng.random() < 0.3 else 0.0)
    return notes


def render(song_id, bpm, rate, depth, key, dur_s, seed):
    rng = np.random.default_rng(seed)
    n = int(dur_s * SR)
    tt = np.arange(n) / SR
    notes = melody(bpm, rate, depth, key, dur_s, rng)

    f0 = np.zeros(n)
    amp = np.zeros(n)
    for nt in notes:
        a, b = int(nt["start_s"] * SR), int(nt["end_s"] * SR)
        local = tt[a:b] - tt[a]
        cents = np.zeros(b - a)
        if nt["vibrato"]:
            onset = np.clip(local / 0.15, 0, 1)  # vibrato depth ramps in over 150 ms
            cents = onset * nt["depth_cents"] / 2 * np.sin(2 * np.pi * nt["rate_hz"] * local)
        f0[a:b] = 440 * 2 ** ((nt["midi"] - 69 + cents / 100) / 12)
        env = np.minimum(1, np.minimum(local / 0.03, (local[-1] - local) / 0.05))
        amp[a:b] = env
    # Render each note separately so phase restarts cleanly.
    vox = np.zeros(n)
    for nt in notes:
        a, b = int(nt["start_s"] * SR), int(nt["end_s"] * SR)
        vox[a:b] = voice(f0[a:b], rng) * amp[a:b]
    vox += 0.003 * rng.normal(size=n) * (amp > 0)  # breath

    # Accompaniment: kick, snare, hi-hat, bass, soft pad.
    beat = 60 / bpm
    drums = np.zeros(n)
    k = np.arange(int(0.25 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-k * 30)) * k) * np.exp(-k * 12)
    snare = rng.normal(size=len(k)) * np.exp(-k * 25) * 0.5
    hat = rng.normal(size=int(0.05 * SR)) * np.exp(-np.arange(int(0.05 * SR)) / SR * 90) * 0.15
    bass = np.zeros(n)
    pad = np.zeros(n)
    roots = [key - 24, key - 24 + 5, key - 24 + 7, key - 24 - 3]
    for i, bt in enumerate(np.arange(0, dur_s, beat)):
        a = int(bt * SR)
        hit = kick if i % 2 == 0 else snare
        m = min(len(hit), n - a)
        drums[a:a + m] += hit[:m]
        for h in (0, beat / 2):
            ha = int((bt + h) * SR)
            mh = min(len(hat), n - ha)
            if mh > 0:
                drums[ha:ha + mh] += hat[:mh]
        bar = (i // 4) % 4
        fb = 440 * 2 ** ((roots[bar] - 69) / 12)
        bl = min(int(beat * 0.9 * SR), n - a)
        lt = np.arange(bl) / SR
        bass[a:a + bl] += (np.sin(2 * np.pi * fb * lt) + 0.3 * np.sin(4 * np.pi * fb * lt)) * np.exp(-lt * 3)
        if i % 4 == 0:
            pl = min(int(4 * beat * SR), n - a)
            pt = np.arange(pl) / SR
            for iv in (12, 16, 19):
                fp = fb * 2 ** (iv / 12)
                pad[a:a + pl] += np.sin(2 * np.pi * fp * pt) * 0.5 + 0.15 * np.sin(4 * np.pi * fp * pt)
    norm = lambda x: x / (np.max(np.abs(x)) + 1e-9)
    mix = 0.55 * norm(vox) + 0.45 * norm(drums) + 0.35 * norm(bass) + 0.12 * norm(pad)
    mix = 0.9 * mix / np.max(np.abs(mix))
    stereo = np.stack([mix, mix], 1)
    return stereo, notes, 0.9 * 0.55 * norm(vox) / np.max(np.abs(0.55 * norm(vox) + 0.45 * norm(drums) + 0.35 * norm(bass) + 0.12 * norm(pad)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40)
    args = ap.parse_args()
    (ROOT / "audio").mkdir(exist_ok=True)
    (ROOT / "test_data").mkdir(exist_ok=True)
    truth, meta = [], []
    for i, (sid, bpm, rate, depth, key, genre) in enumerate(SONGS):
        y, notes, solo = render(sid, bpm, rate, depth, key, args.seconds, seed=i)
        sf.write(ROOT / "audio" / f"{sid}.wav", y, SR, subtype="PCM_16")
        # The isolated voice, for testing downstream stages with ideal separation.
        (ROOT / "test_data" / "solo").mkdir(exist_ok=True)
        sf.write(ROOT / "test_data" / "solo" / f"{sid}.wav", solo, SR, subtype="PCM_16")
        for j, nt in enumerate(notes):
            truth.append({"song_id": sid, "note": j, **nt})
        meta.append({"file": f"{sid}.wav", "title": sid, "artist": "synthetic", "soloist": "synth voice",
                     "genre": genre, "instrument": "voice",
                     # Override only one song, to exercise both tempo paths.
                     "bpm": bpm if sid == "synth_fast" else ""})
        print(f"{sid}: {bpm} BPM, {sum(n['vibrato'] for n in notes)} vibrato notes / {len(notes)}")
    pd.DataFrame(meta).to_csv(ROOT / "metadata.csv", index=False)
    gt = pd.DataFrame(truth)
    gt.round(4).to_csv(ROOT / "test_data" / "ground_truth_notes.csv", index=False)

    rng = np.random.default_rng(42)
    v = gt[gt["vibrato"]].copy()
    v["time_s"] = ((v["start_s"] + v["end_s"]) / 2).round(2)
    for name, bias, sd in (("simulated_rater_A", 0.0, 0.12), ("simulated_rater_B", 0.08, 0.2)):
        r = v[["song_id", "time_s"]].copy()
        r["rate_hz"] = (v["rate_hz"] + bias + rng.normal(0, sd, len(v))).round(3)
        r.to_csv(ROOT / "test_data" / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
