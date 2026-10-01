#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENE_LIST="${BOXFUSION_B6_BOXER_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
PYTHON="${BOXFUSION_AUDIT_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"

list_sha="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
list_scope="$(basename "$SCENE_LIST" .txt)-${list_sha:0:12}"
logs="$ROOT/logs/b6_boxer_uncertainty"
reports="$ROOT/reports/b6_boxer_uncertainty"

"$PYTHON" "$ROOT/tools/report_b6_boxer_uncertainty.py" \
    --control-log "$logs/u0_control/$list_scope/eval_stdout.log" \
    --observer-log "$logs/u1_observer/$list_scope/eval_stdout.log" \
    --active-log "$logs/u2_active/$list_scope/eval_stdout.log" \
    --audit "$reports/${list_scope}_audit.json" \
    --output "$reports/${list_scope}_effectiveness.json"

echo "Effectiveness report completed: $reports/${list_scope}_effectiveness.json"
