#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCOPE="${BOXFUSION_P_FULL100:-0}"
if [[ "$SCOPE" == 1 ]]; then scope=full100; else scope=ablation10; fi
P0_TAG="${BOXFUSION_P0_RUN_TAG:-p0_${scope}_b6frozen_v1}"
P1_TAG="${BOXFUSION_P1_RUN_TAG:-p1_${scope}_b6frozen_v1}"
SCENE_LIST="${BOXFUSION_P_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
if [[ "$SCOPE" == 1 && -z "${BOXFUSION_P_SCENE_LIST:-}" ]]; then
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
fi
P0_ROOT="${BOXFUSION_P0_PRED_ROOT:-$ROOT/results/p_ablation/$P0_TAG}"
P1_ROOT="${BOXFUSION_P1_PRED_ROOT:-$ROOT/results/p_ablation/$P1_TAG}"
DIAG_ROOT="${BOXFUSION_P1_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/p_ablation/$P1_TAG}"
REPORT_ROOT="${BOXFUSION_P_REPORT_ROOT:-$ROOT/reports/p_ablation/$P1_TAG}"
PYTHON="${BOXFUSION_ENV_ROOT:-${CONDA_PREFIX:-/home/admin1/miniconda3/envs/boxfusion2}}/bin/python"
mkdir -p "$REPORT_ROOT"

"$PYTHON" "$ROOT/tools/verify_p1_identity.py" \
    --baseline-root "$P0_ROOT" \
    --observer-root "$P1_ROOT" \
    --diagnostics-root "$DIAG_ROOT" \
    --output "$REPORT_ROOT/identity.json"
"$PYTHON" "$ROOT/tools/report_p1_residual_recall.py" \
    --scene-list "$SCENE_LIST" \
    --prediction-root "$P1_ROOT" \
    --diagnostics-root "$DIAG_ROOT" \
    --gt-root "${BOXFUSION_P1_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}" \
    --scans-root "${BOXFUSION_P1_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}" \
    --output "$REPORT_ROOT/recall.json"
