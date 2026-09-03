#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENE_LIST="${BOXFUSION_B6_BOXER_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
PYTHON="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}/bin/python"

if [[ ! -f "$SCENE_LIST" ]]; then
    echo "Missing scene list: $SCENE_LIST" >&2
    exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Python interpreter: $PYTHON" >&2
    exit 1
fi

list_sha="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
list_scope="$(basename "$SCENE_LIST" .txt)-${list_sha:0:12}"
base="$ROOT/results/b6_selective_boxer"
diagnostics="$ROOT/diagnostics/b6_selective_boxer"
report="$ROOT/reports/b6_selective_boxer/$list_scope/contract_audit.json"

exec "$PYTHON" "$ROOT/tools/audit_b6_selective_boxer.py" \
    --scene-list "$SCENE_LIST" \
    --control-root "$base/s0_control/$list_scope" \
    --observer-root "$base/s0_observer/$list_scope" \
    --observer-diagnostics "$diagnostics/s0_observer/$list_scope/boxer" \
    --active-root "$base/s1_selective/$list_scope" \
    --active-diagnostics "$diagnostics/s1_selective/$list_scope/boxer" \
    --observer-box-atol 1e-4 \
    --observer-score-atol 1e-6 \
    --output "$report"
