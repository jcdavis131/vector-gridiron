#!/usr/bin/env bash
# run_local_gpu.sh — Gridiron — local GPU pickup
set -euo pipefail
EPOCHS="${1:-50}"
cd "$(dirname "$0")/.."
echo "[gridiron] epochs=$EPOCHS $(date -u)"
DEVICE="cpu"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then DEVICE="cuda"; echo "[gridiron] CUDA -> $DEVICE"; else echo "[gridiron] -> cpu"; fi

if [ ! -f pipeline/data/train_matrix.npz ]; then
  echo "[gridiron] Missing pipeline/data/train_matrix.npz - fetching nflverse or synthetic"
  python3 pipeline/fetch_nflverse.py --seasons 2021 2022 2023 2>&1 || python3 pipeline/train_mtnn.py --synthetic --epochs "$EPOCHS" | tee -a pipeline/cache/train_gridiron_${EPOCHS}ep.log
  if [ -f pipeline/data/train_matrix.npz ]; then
    python3 pipeline/train_mtnn.py --epochs "$EPOCHS" 2>&1 | tee -a pipeline/cache/train_gridiron_${EPOCHS}ep.log || true
  fi
  exit 0
fi
python3 pipeline/train_mtnn.py --epochs "$EPOCHS" 2>&1 | tee -a pipeline/cache/train_gridiron_${EPOCHS}ep.log || true
echo "[gridiron] done"
