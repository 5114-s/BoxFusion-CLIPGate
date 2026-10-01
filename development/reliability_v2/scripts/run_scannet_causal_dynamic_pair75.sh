#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
RUN_DIR="${BOXFUSION_DYNAMIC_PAIR_DIR:-$ROOT/reports/causal_dynamic_pair75_20260908}"
GPUS="${BOXFUSION_GPUS:-0,1}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

exec "$PYTHON" -u "$ROOT/tools/run_causal_dynamic_pair75.py" \
    --run-dir "$RUN_DIR" --gpus "$GPUS" "$@"
