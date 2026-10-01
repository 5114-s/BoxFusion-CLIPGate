#!/usr/bin/env bash
set -euo pipefail

# CPU-only, read-only-by-default paired B5 report.
#
# Usage:
#   bash scripts/report_scannet_b5_ap50.sh \
#     /path/to/identity_predictions \
#     /path/to/candidate_predictions \
#     /path/to/candidate_diagnostics
#
# Optional writes are explicit:
#   BOXFUSION_B5_REPORT_JSON=/new/report.json
#   BOXFUSION_B5_SCORE_LOCK_ROOT=/new/nonexistent/prediction_root

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 IDENTITY_PRED_ROOT CANDIDATE_PRED_ROOT CANDIDATE_DIAGNOSTICS_ROOT" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
SCENE_LIST="${BOXFUSION_B5_REPORT_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
SCAN_ROOT="${BOXFUSION_SCANNET_SCAN_ROOT:-/extra/ZhaoX/scannet_data/scans}"
GT_ROOT="${BOXFUSION_SCANNET_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Python environment: $PYTHON" >&2
    exit 1
fi

arguments=(
    --identity-pred-root "$1"
    --candidate-pred-root "$2"
    --candidate-diagnostics-root "$3"
    --scene-list "$SCENE_LIST"
    --scan-root "$SCAN_ROOT"
    --gt-root "$GT_ROOT"
)
if [[ -n "${BOXFUSION_B5_REPORT_JSON:-}" ]]; then
    arguments+=(--output-json "$BOXFUSION_B5_REPORT_JSON")
fi
if [[ -n "${BOXFUSION_B5_SCORE_LOCK_ROOT:-}" ]]; then
    arguments+=(--lock-identity-scores "$BOXFUSION_B5_SCORE_LOCK_ROOT")
fi

exec "$PYTHON" "$ROOT/tools/report_b5_ap50_ablation.py" "${arguments[@]}"
