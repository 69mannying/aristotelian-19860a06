#!/usr/bin/env bash
# Minimal single-GPU reproduction of the PRH real-data headline of arXiv 2602.14486.
# Extracts vision (timm ViT) + language (Bloomz) features on a WIT slice, applies
# the repo's null calibration, and checks that global CKA convergence collapses
# after calibration while local mutual-kNN survives. Writes EVAL.md + artifacts.
set -euo pipefail

PY=$(command -v python || command -v python3)
echo "Using interpreter: $PY"
"$PY" --version
"$PY" -m ensurepip --upgrade 2>/dev/null || true

echo "=== nvidia-smi ==="
nvidia-smi || echo "(no nvidia-smi; will fall back to CPU)"

echo "=== Installing deps needed for the PRH experiment ==="
# torch/torchvision usually preinstalled on GPU images; install only if missing.
if ! "$PY" -c "import torch" 2>/dev/null; then
  "$PY" -m pip install --no-cache-dir torch torchvision
fi
"$PY" -m pip install --no-cache-dir \
  "timm" "transformers" "datasets" "accelerate" "sentencepiece" "protobuf" \
  "scikit-learn" "numpy" "loguru" "tqdm" "Pillow"

# Standalone calibration package (no extra deps).
"$PY" -m pip install --no-cache-dir -e . --no-deps || true

"$PY" -c "import torch,timm,transformers,datasets,sklearn; \
print('torch',torch.__version__,'cuda',torch.cuda.is_available()); \
print('timm',timm.__version__,'transformers',transformers.__version__)"

echo "=== Running minimal PRH reproduction ==="
export PRH_MODELSET="${PRH_MODELSET:-min}"
export PRH_MAX_SAMPLES="${PRH_MAX_SAMPLES:-256}"
export PRH_PERMS="${PRH_PERMS:-200}"
export PRH_BATCH="${PRH_BATCH:-8}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
"$PY" prh_min.py

echo "=== EVAL.md ==="
cat EVAL.md
