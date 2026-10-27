"""Command-line interface: python -m vibtempo <command> ..."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import hardware  # noqa: F401  (sets the MPS fallback env var before torch loads)
from .config import load_config, total_ram_gb

log = logging.getLogger("vibtempo")
STAGES = ["separate", "tempo", "pitch", "analyze", "report"]


def _songs(cfg, args):
    from .corpus import discover
    songs = discover(cfg, args.songs or None)
    if not songs:
        sys.exit(f"no audio files in {cfg.path('audio_dir')}")
    return songs


def cmd_info(cfg, args):
    from .hardware import pick_device
    from .pitch import find_pitch
    from .separate import find_stem
    from .tempo import tempo_path
    print(f"project root : {cfg.root}")
    print(f"RAM          : {total_ram_gb():.1f} GB -> profile '{cfg.profile}'")
    print(f"device       : {pick_device(cfg['hardware']['device'])}")
    print(f"demucs       : {cfg['separation']['model']} (segment {cfg['separation']['segment']}s)")
    print(f"pitch        : {cfg['pitch']['method']} {cfg['pitch'].get('crepe_model', '')}")
    print(f"workers      : {cfg.workers}")
    songs = _songs(cfg, args)
    print(f"\n{'song':30s} {'stem':22s} {'tempo':6s} {'pitch'}")
    for s in songs:
        st = find_stem(cfg, s)
        pf = find_pitch(cfg, s)
        print(f"{s.song_id[:30]:30s} {(st[1] + '/' + st[0].stem) if st else '-':22s} "
              f"{'yes' if tempo_path(cfg, s).exists() else '-':6s} {pf.stem.split('__')[1] if pf else '-'}")


def cmd_run(cfg, args):
    from . import analyze, pitch, plots, separate, tempo
    songs = _songs(cfg, args)
    stages = args.stages.split(",") if args.stages else STAGES
    bad = set(stages) - set(STAGES)
    if bad:
        sys.exit(f"unknown stage(s): {sorted(bad)}")
    force = set(args.force.split(",")) if args.force else set()
    log.info("profile %s, %d songs, stages %s", cfg.profile, len(songs), ",".join(stages))
    t0 = time.time()

    # In the full profile, tempo (CPU) runs in a process pool while Demucs uses
    # the GPU; in the light profile everything runs one file at a time.
    pool = ProcessPoolExecutor(cfg.workers) if cfg.workers > 1 and "tempo" in stages else None
    try:
        futures = tempo.run(cfg, songs, "tempo" in force, pool) if "tempo" in stages else []
        if "separate" in stages and cfg["separation"]["enabled"]:
            separate.run(cfg, songs, "separate" in force)
        for f in futures:
            f.result()
    finally:
        if pool:
            pool.shutdown()
    if "pitch" in stages:
        pitch.run(cfg, songs, "pitch" in force)
    if "analyze" in stages:
        analyze.run(cfg, songs, "analyze" in force)
    if "report" in stages:
        notes_all, summary = analyze.write_tables(cfg, songs)
        figs = plots.make_all(cfg, notes_all, summary)
        out = cfg.path("output_dir")
        log.info("wrote %s, %s, %s", cfg.rel(out / "notes.csv"), cfg.rel(out / "songs.csv"),
                 ", ".join(cfg.rel(f) for f in figs))
    log.info("done in %.1fs", time.time() - t0)


def cmd_validate(cfg, args):
    from . import analyze, validate
    from .corpus import discover
    songs = discover(cfg)
    notes_all, summary = analyze.build_tables(cfg, songs)
    res, report = validate.run(cfg, args.ratings, args.names, notes_all, summary)
    print(report.read_text())
    print(f"(written to {cfg.rel(report.parent)})")


def cmd_spotcheck(cfg, args):
    import numpy as np
    from . import analyze, spotcheck
    from .corpus import discover
    song = discover(cfg, [args.song])[0]
    df = analyze.song_notes(cfg, song)
    if df is None:
        sys.exit(f"{args.song} has not been analysed yet; run `run` first")
    if args.note is not None:
        rows = df[df["note"] == args.note]
    elif args.time is not None:
        d = np.maximum(0, np.maximum(df["start_s"] - args.time, args.time - df["end_s"]))
        rows = df.loc[[d.idxmin()]]
    else:
        pool = df[~df["vibrato"]] if args.rejected else df[df["vibrato"]]
        rows = pool.sample(min(args.random, len(pool)), random_state=args.seed)
    if rows.empty:
        sys.exit("no matching note")
    out = cfg.path("output_dir") / "spotcheck"
    for _, r in rows.sort_values("start_s").iterrows():
        for p in spotcheck.export(cfg, song, r, out):
            print(cfg.rel(p))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vibtempo", description=__doc__)
    ap.add_argument("--config", default=None, help="config file (default: config.yaml in the project)")
    ap.add_argument("--profile", choices=["auto", "light", "full"], default=None,
                    help="override hardware.profile")
    ap.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="show profile, device and what is cached")
    p.add_argument("songs", nargs="*")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("run", help="run pipeline stages (each is cached and resumable)")
    p.add_argument("songs", nargs="*", help="song ids (file names without extension); default all")
    p.add_argument("--stages", help=f"comma-separated subset of {','.join(STAGES)}")
    p.add_argument("--force", help="comma-separated stages to recompute even if cached")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("validate", help="agreement with human ratings")
    p.add_argument("ratings", nargs="+", type=Path, help="one CSV per rater")
    p.add_argument("--names", nargs="+", help="rater names (default: file names)")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("spotcheck", help="export audio clip + pitch plot for notes")
    p.add_argument("song")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--note", type=int)
    g.add_argument("--time", type=float, help="seconds; picks the note at/nearest this time")
    g.add_argument("--random", type=int, default=3, help="sample this many notes (default)")
    p.add_argument("--rejected", action="store_true", help="sample from rejected notes")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_spotcheck)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    over = {"hardware": {"device": args.device}} if args.device else None
    cfg = load_config(args.config, args.profile, over)
    try:
        args.fn(cfg, args)
    except KeyboardInterrupt:
        sys.exit("\ninterrupted; finished items are cached, rerun the same command to resume")


if __name__ == "__main__":
    main()
