#!/usr/bin/env bash
# Minimal end-to-end reproduction of arXiv 2602.14486 (Aristotelian / null calibration).
# CPU-only. Installs just the dep the standalone package needs (torch), then runs
# the synthetic width/depth/power demonstration, which writes EVAL.md + artifacts.
set -euo pipefail

echo "=== Python / pip ==="
python --version || python3 --version
PY=$(command -v python || command -v python3)
echo "Using interpreter: $PY"

# Ensure pip exists.
"$PY" -m ensurepip --upgrade 2>/dev/null || true
"$PY" -m pip --version || { echo "pip unavailable"; exit 1; }

echo "=== Installing torch (CPU) if needed ==="
if ! "$PY" -c "import torch" 2>/dev/null; then
  "$PY" -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    || "$PY" -m pip install --no-cache-dir torch
fi
"$PY" -c "import torch; print('torch', torch.__version__)"

echo "=== Installing standalone calibrated_similarity package (editable, no deps) ==="
"$PY" -m pip install --no-cache-dir -e . --no-deps || true
# Fallback: the package is importable directly from repo root regardless.
"$PY" -c "import calibrated_similarity as c; print('calibrated_similarity', c.__version__)"

echo "=== Running minimal reproduction ==="
export REPRO_DEVICE="${REPRO_DEVICE:-cpu}"
export REPRO_SEED="${REPRO_SEED:-0}"
export REPRO_K="${REPRO_K:-200}"
export REPRO_TRIALS="${REPRO_TRIALS:-30}"
"$PY" minimal_repro.py

echo "=== EVAL.md ==="
cat EVAL.md
