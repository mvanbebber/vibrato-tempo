"""TEST ONLY: install the synthetic songs' true isolated voice as if it were a
Demucs stem from a model called 'oracle', to test downstream stages with ideal
separation. Use with:  vibtempo --config config_test_oracle.yaml run"""
from pathlib import Path

import soundfile as sf

from vibtempo.config import load_config
from vibtempo.corpus import discover
from vibtempo.separate import stem_path

ROOT = Path(__file__).resolve().parent.parent
cfg = load_config(ROOT / "config.yaml")
for s in discover(cfg):
    src = ROOT / "test_data" / "solo" / f"{s.song_id}.wav"
    if src.exists():
        y, sr = sf.read(src)
        dst = stem_path(cfg, s, "oracle", "vocals")
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, y, sr, subtype="PCM_24")
        print(cfg.rel(dst))
