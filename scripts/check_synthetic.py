"""Compare tool output with the synthetic ground truth (after `vibtempo run`)."""
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "output")
TRUE_BPM = {"synth_slow": 66.0, "synth_mid": 112.0, "synth_fast": 176.0}

gt = pd.read_csv(ROOT / "test_data" / "ground_truth_notes.csv")
tool = pd.read_csv(OUT / "notes_all.csv")
songs = pd.read_csv(OUT / "songs.csv")

rows = []
for _, g in gt.iterrows():
    cand = tool[tool["song_id"] == g["song_id"]]
    ov = (np.minimum(cand["end_s"], g["end_s"]) - np.maximum(cand["start_s"], g["start_s"])).clip(lower=0)
    best = ov.idxmax() if len(ov) and ov.max() > 0 else None
    r = {"song_id": g["song_id"], "true_vibrato": g["vibrato"], "dur": g["end_s"] - g["start_s"],
         "true_rate": g["rate_hz"], "true_depth": g["depth_cents"], "segmented": best is not None}
    if best is not None:
        r.update(tool_vibrato=bool(cand.at[best, "vibrato"]), tool_rate=cand.at[best, "rate_hz"],
                 tool_depth=cand.at[best, "depth_cents"], conf=cand.at[best, "confidence"],
                 reason=cand.at[best, "reject_reason"])
    rows.append(r)
m = pd.DataFrame(rows)

print("Tempo")
for _, s in songs.iterrows():
    print(f"  {s['song_id']:11s} true {TRUE_BPM[s['song_id']]:6.1f}  estimated {s['bpm_estimated']:6.1f}"
          f"  used {s['bpm']:6.1f} ({s['bpm_source']})")

print("\nNote detection (ground-truth notes)")
for sid, d in m.groupby("song_id", sort=False):
    vib = d[d["true_vibrato"]]
    straight = d[~d["true_vibrato"] & (d["dur"] >= 0.3)]
    hit = vib["tool_vibrato"].fillna(False).astype(bool)
    fp = straight["tool_vibrato"].fillna(False).astype(bool)
    print(f"  {sid:11s} vibrato notes found {hit.sum()}/{len(vib)}   "
          f"straight sustained notes wrongly accepted {fp.sum()}/{len(straight)}")

ok = m[m["true_vibrato"] & m["tool_vibrato"].fillna(False).astype(bool)]
err = ok["tool_rate"] - ok["true_rate"]
derr = (ok["tool_depth"] - ok["true_depth"]) / ok["true_depth"]
print(f"\nRate error over {len(ok)} detected notes: mean {err.mean():+.3f} Hz, "
      f"MAE {err.abs().mean():.3f} Hz, max |err| {err.abs().max():.3f} Hz")
print(f"Depth error: median {100 * derr.median():+.1f}%, MAE {100 * derr.abs().mean():.1f}%")
miss = m[m["true_vibrato"] & ~m["tool_vibrato"].fillna(False).astype(bool)]
if len(miss):
    print("\nMissed vibrato notes:", miss["reason"].fillna("not segmented").value_counts().to_dict())

print("\nSong medians (Hz): true vs tool")
for sid, d in gt[gt["vibrato"]].groupby("song_id", sort=False):
    t = songs.set_index("song_id").at[sid, "rate_hz_median"]
    print(f"  {sid:11s} true {d['rate_hz'].median():.3f}  tool {t:.3f}")
