# vibtempo

Measures vibrato rate against song tempo across a corpus of recordings:
source separation (Demucs), then tempo (librosa), pitch tracking (torchcrepe or pYIN),
note segmentation, and per-note vibrato fitting. Outputs per-note and per-song tables,
corpus plots, agreement statistics against human raters, and spot-check clips.

## Setup (once per Mac)

```bash
./setup_mac.sh          # creates .venv, installs requirements, reports MPS
. .venv/bin/activate
python -m vibtempo info # shows RAM, chosen profile, device, cache status
```

The `.venv` is machine-specific, so don't sync it. Everything else syncs.

## Layout

```
config.yaml        all parameters (one place)
metadata.csv       optional: file,title,artist,soloist,genre,instrument,bpm,stem
audio/             input recordings (wav/mp3/flac/m4a/aiff)
cache/             stems, tempo, pitch tracks, per-song analysis (keyed by file content hash)
output/            notes.csv, notes_all.csv, songs.csv, plots/, validation/, spotcheck/
```

All paths in `config.yaml` are relative to the project root.

## Running

```bash
python -m vibtempo run                         # all stages, all songs
python -m vibtempo run song_a song_b           # a subset (file names without extension)
python -m vibtempo run --stages separate,tempo # e.g. only the GPU-heavy part, on the Pro
python -m vibtempo run --stages analyze,report # re-analyse from cached pitch tracks
python -m vibtempo run --force analyze         # recompute a stage despite the cache
python -m vibtempo --profile light run         # override profile auto-detection
python -m vibtempo --device cpu run            # force CPU
```

Every stage writes its results atomically (temp file, then rename) and skips work that
is already cached, so a run can be stopped with Ctrl-C and restarted at any point.

| Stage | Cached at | Recomputed when |
|---|---|---|
| separate | `cache/stems/<hash>/<model>/<stem>.flac` | audio changes; none of `separation.reuse_models` cached |
| tempo | `cache/tempo/<hash>__<params>.json` | `tempo:` settings change |
| pitch | `cache/pitch/<hash>/<stem>__<tracker>.npz` | tracker, hop, fmin/fmax change (raw f0 + periodicity; thresholds are applied later) |
| analyze | `cache/analysis/<hash>/...csv` | any `notes:` / `vibrato:` setting or `voicing_threshold` changes |
| report | `output/` | every run (fast) |

Tuning `notes:` and `vibrato:` parameters only re-runs `analyze`, which takes seconds.

## Hardware profiles

`hardware.profile: auto` picks **light** when RAM is under 16 GB, otherwise **full**.

| | light (Air, 8 GB) | full (Max, 64 GB) |
|---|---|---|
| Demucs | `htdemucs`, 5 s segments, no shifts | `htdemucs_ft`, 7.8 s segments, 1 shift |
| CREPE | `tiny`, batch 256 | `full`, batch 2048 |
| CPU stages | one file at a time | 6 worker processes; tempo runs during separation |
| Memory | models freed and MPS cache emptied between stages and files | models freed between stages |

The device is MPS when available. If a Demucs or CREPE call fails on MPS, it is
retried on the CPU, and ops MPS lacks fall back to the CPU through `PYTORCH_ENABLE_MPS_FALLBACK=1`.

### Split work between the two Macs

Run separation and pitch on the Pro, then analyse anywhere:

```bash
# on the Pro
python -m vibtempo run --stages separate,tempo,pitch
# copy the project to the Air (or use a shared Dropbox/iCloud folder)
rsync -a --exclude .venv pro.local:vibrato-tempo/ ~/vibrato-tempo/
# on the Air: reuses the Pro's htdemucs_ft stems and CREPE-full tracks
python -m vibtempo run --stages analyze,report
```

Lookups try the configured model first and then the `reuse_models` /
`reuse_crepe_models` lists, so the Air uses the Pro's higher-quality results
instead of recomputing. `songs.csv` records `stem_source` and `pitch_track` for every
song. **Make the final corpus numbers come from a single model combination.**
Check those two columns before reporting.

## Metadata CSV

Columns (all optional except `file`): `file` (name in `audio/`), `title`, `artist`,
`soloist`, `genre`, `instrument`, `bpm` (overrides the estimated tempo), `stem`
(forces a Demucs stem). `instrument` picks the stem via `separation.stem_for_instrument`:
voices go to `vocals`, and other instruments default to `other`. Note that `other` also
contains guitars, keys and so on, so instrumental soloists are tracked less cleanly than singers.

## Outputs

- `output/notes.csv`: one row per accepted vibrato note: song, start/end, pitch,
  duration, `rate_hz`, `rate_cpm` (cycles per minute, comparable to BPM),
  `depth_cents` (peak-to-peak), `confidence`, `cycles`.
- `output/notes_all.csv`: every sustained note, with `vibrato` (True/False) and `reject_reason`.
- `output/songs.csv`: per song: `bpm`, `bpm_source` (metadata/estimated), `bpm_estimated`,
  note counts, median and IQR rate, `rate_to_tempo` (vibrato cycles per beat),
  `log2_rate_to_tempo`, `zone`, `in_target_band`, `status`, provenance.
- `output/plots/rate_vs_tempo.png`: song medians (IQR bars) on a log-2 tempo axis
  with zone boundaries, the target band, and each zone's predicted `subdivision × tempo` line.
- `output/plots/per_song_rates.png`: per-note rates for each song, ordered by tempo.

## How vibrato is measured

1. **Voicing**: frames with periodicity ≥ `pitch.voicing_threshold`; dropouts ≤ `max_gap_ms` are bridged.
2. **Notes**: the pitch contour is smoothed with two box filters (250 ms and 167 ms; their
   nulls at 4/6/8/12 Hz remove vibrato but keep note steps). A new note starts when this
   contour moves more than `split_cents` from the note's running median. Edges are trimmed by
   `edge_trim_ms`, and notes shorter than `min_note_ms` are dropped.
3. **Fit**: for each candidate rate in `[rate_min_hz, rate_max_hz]`, the note's pitch
   (cents) is regressed on a degree-`trend_degree` polynomial plus a sinusoid. The rate
   is the best-fitting frequency, refined by parabolic interpolation. **Confidence** is the
   share of the detrended pitch variance that the sinusoid explains (R², 0–1).
   **Depth** is the sinusoid's peak-to-peak amplitude in cents. A sinusoid fit works on
   short notes (~1.5 cycles), where autocorrelation and FFT peak picking break down.
4. **Accept** a note if it has ≥ `min_cycles` cycles, confidence ≥ `min_confidence`,
   depth within `[min_depth_cents, max_depth_cents]`, and a rate that is not at the edge of the search range.

## Validation

One CSV per rater: `song_id`, `rate_hz` (or `rate_cpm`/`rate_bpm`), and optionally
`time_s` (any time inside the rated note).

```bash
python -m vibtempo validate ratings_me.csv ratings_r2.csv --names MvB R2
```

Writes `output/validation/report.md`, `agreement.csv`, `agreement.png` (scatter and Bland–Altman),
and the matched values. For every pair (tool–A, tool–B, A–B) and for the whole group it reports
ICC(2,1) with a 95% CI (two-way random, absolute agreement, single rater: the usual choice
when the tool stands in for a rater), ICC(3,1), Pearson r, Spearman ρ, bias, MAE, and limits of agreement.
Files with `time_s` are compared note by note (each rating is matched to the tool note containing
that time, within `validation.match_tolerance_s`). All files are also compared at song level.
The ICC code is checked against Shrout & Fleiss (1979) in `tests/`.

## Spot-check

```bash
python -m vibtempo spotcheck song_a --note 12
python -m vibtempo spotcheck song_a --time 63.2
python -m vibtempo spotcheck song_a --random 5            # random accepted notes
python -m vibtempo spotcheck song_a --random 5 --rejected # random rejected notes
```

Each note produces `<song>_noteNNNN_<t>s_solo.wav`, `_mix.wav` (the note ± `spotcheck.pad_s`)
and a `.png` showing the pitch curve with the fitted trend + sinusoid and the detrended residual.

## Tests and synthetic data

```bash
pytest -q                                   # ICC vs Shrout & Fleiss, fit, segmentation
python scripts/make_synthetic.py            # 3 x 40 s songs, known tempo + per-note vibrato
python scripts/make_test_configs.py         # config_test_mix.yaml / config_test_oracle.yaml
python scripts/install_oracle_stems.py      # true isolated voice as an 'oracle' stem
vibtempo --config config_test_oracle.yaml run
python scripts/check_synthetic.py output_test_oracle
vibtempo --config config_test_oracle.yaml validate \
    test_data/simulated_rater_A.csv test_data/simulated_rater_B.csv
```

The rater files in `test_data/` are **simulated** (ground truth plus noise), and exist only to
exercise validation mode.

Results on the synthetic set (47 vibrato notes, 45 straight sustained notes):

| Pitch source | Tracker | Vibrato notes found | False accepts | Rate MAE | Depth bias |
|---|---|---|---|---|---|
| ideal stem | CREPE full | 46/47 | 0 | 0.014 Hz | −9% |
| ideal stem | CREPE tiny | 39/47 | 0 | 0.012 Hz | −21% |
| ideal stem | pYIN | 41/47 | 0 | 0.063 Hz | −44% |
| full mix, no separation | CREPE tiny | 8/47 | 1 | 0.033 Hz | −27% |

Depth reads low because CREPE quantises pitch into 20-cent bins (vibrato comes out as a
staircase) and pYIN's HMM smooths it. Rate is unaffected. **Report depths only from a single
tracker setting.** Treat them as relative rather than absolute.
