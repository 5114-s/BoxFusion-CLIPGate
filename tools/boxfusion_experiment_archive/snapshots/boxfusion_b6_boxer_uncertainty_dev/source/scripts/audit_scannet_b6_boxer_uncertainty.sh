#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_ROOT="${BOXFUSION_SELECTIVE_SOURCE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev}"
SCENE_LIST="${BOXFUSION_B6_BOXER_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
PYTHON="${BOXFUSION_AUDIT_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"

if [[ ! -f "$SCENE_LIST" ]]; then
    echo "Missing scene list: $SCENE_LIST" >&2
    exit 1
fi
list_sha="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
list_scope="$(basename "$SCENE_LIST" .txt)-${list_sha:0:12}"
base="$ROOT/results/b6_boxer_uncertainty"
diagnostics="$ROOT/diagnostics/b6_boxer_uncertainty"
report="$ROOT/reports/b6_boxer_uncertainty/${list_scope}_audit.json"

"$PYTHON" "$ROOT/tools/audit_b6_boxer_uncertainty.py" \
    --scene-list "$SCENE_LIST" \
    --source-g0-root "$SOURCE_ROOT/results/b6_selective_boxer/s1_selective/$list_scope" \
    --control-root "$base/u0_control/$list_scope" \
    --observer-root "$base/u1_observer/$list_scope" \
    --active-root "$base/u2_active/$list_scope" \
    --observer-diagnostics "$diagnostics/u1_observer/$list_scope/uncertainty" \
    --active-diagnostics "$diagnostics/u2_active/$list_scope/uncertainty" \
    --control-boxer-diagnostics "$diagnostics/u0_control/$list_scope/boxer" \
    --observer-boxer-diagnostics "$diagnostics/u1_observer/$list_scope/boxer" \
    --active-boxer-diagnostics "$diagnostics/u2_active/$list_scope/boxer" \
    --output "$report"

echo "Uncertainty-fusion audit completed: $report"
