#!/usr/bin/env bash
# One-time setup on each Mac. The .venv is machine-specific: create it on each
# machine and do not sync it (everything else in the project syncs fine).
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "warning: not running natively on Apple Silicon (uname -m = $(uname -m))."
  echo "         Use an arm64 Python, or MPS will be unavailable."
fi

PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10"'
echo "using $("$PY" --version) at $(command -v "$PY")"

command -v ffmpeg >/dev/null || echo "note: ffmpeg not found; needed only for .m4a/.aac input (brew install ffmpeg)"

[[ -d .venv ]] || "$PY" -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # PyTorch for macOS arm64 includes MPS support

python - <<'PY'
import torch
print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
PY
python -m vibtempo info || true
echo
echo "Demucs downloads its model on first use (htdemucs ~80 MB; htdemucs_ft ~320 MB)."
