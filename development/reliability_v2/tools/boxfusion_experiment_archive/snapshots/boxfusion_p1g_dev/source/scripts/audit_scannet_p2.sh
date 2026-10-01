#!/usr/bin/env bash
set -euo pipefail

# Validate a completed P2 observer run and generate its proposal-recall report.
# P2 diagnostics contain both candidate streams; a same-code-tree P1 formal
# prediction run is additionally required to verify the observer identity.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FULL100="${BOXFUSION_P_FULL100:-0}"
case "$FULL100" in
    0) SCOPE=ablation10 ;;
    1) SCOPE=full100 ;;
    *) echo "BOXFUSION_P_FULL100 must be 0 or 1" >&2; exit 2 ;;
esac

if [[ -n "${BOXFUSION_P_SCENE_LIST:-}" ]]; then
    SCENE_LIST="$BOXFUSION_P_SCENE_LIST"
    SCOPE=custom
elif [[ "$FULL100" == "1" ]]; then
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
else
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
fi

RUN_TAG="${BOXFUSION_P2_RUN_TAG:-${BOXFUSION_P_RUN_TAG:-p2_${SCOPE}_b6frozen_v1}}"
PRED_ROOT="${BOXFUSION_P2_PRED_ROOT:-${BOXFUSION_P_PRED_ROOT:-$ROOT/results/p_ablation/$RUN_TAG}}"
DIAGNOSTICS_ROOT="${BOXFUSION_P2_DIAGNOSTICS_ROOT:-${BOXFUSION_P_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/p_ablation/$RUN_TAG}}"
REPORT_ROOT="${BOXFUSION_P2_REPORT_ROOT:-${BOXFUSION_P_REPORT_ROOT:-$ROOT/reports/p_ablation/$RUN_TAG}}"
P1_BASELINE_TAG="${BOXFUSION_P2_BASELINE_TAG:-p1_${SCOPE}_b6frozen_v1}"
P1_BASELINE_ROOT="${BOXFUSION_P2_BASELINE_PRED_ROOT:-$ROOT/results/p_ablation/$P1_BASELINE_TAG}"
P1_CHECKPOINT="${BOXFUSION_P1_RESIDUAL_CHECKPOINT:-$ROOT/models/scannet_p1_residual.pt}"
P2_CHECKPOINT="${BOXFUSION_P2_OCCUPANCY_CHECKPOINT:-$ROOT/models/scannet_p2_occupancy_topk.pt}"
GT_ROOT="${BOXFUSION_P2_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCANS_ROOT="${BOXFUSION_P2_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-${CONDA_PREFIX:-/home/admin1/miniconda3/envs/boxfusion2}}"
PYTHON="$ENV_ROOT/bin/python"

for path in \
    "$PYTHON" \
    "$SCENE_LIST" \
    "$P1_CHECKPOINT" \
    "$P2_CHECKPOINT" \
    "$ROOT/tools/validate_p2_run_artifacts.py" \
    "$ROOT/tools/report_p2_occupancy_recall.py"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing P2 audit input: $path" >&2
        exit 1
    fi
done
for directory in \
    "$PRED_ROOT" \
    "$DIAGNOSTICS_ROOT" \
    "$P1_BASELINE_ROOT" \
    "$GT_ROOT" \
    "$SCANS_ROOT"; do
    if [[ ! -d "$directory" ]]; then
        echo "Missing P2 audit directory: $directory" >&2
        exit 1
    fi
done

mkdir -p "$REPORT_ROOT"
echo "P2 audit: scope=$SCOPE, tag=$RUN_TAG"
echo "  predictions: $PRED_ROOT"
echo "  diagnostics: $DIAGNOSTICS_ROOT"
echo "  frozen P1 baseline: $P1_BASELINE_ROOT"
echo "  reports: $REPORT_ROOT"

"$PYTHON" "$ROOT/tools/validate_p2_run_artifacts.py" \
    --scene-list "$SCENE_LIST" \
    --prediction-root "$PRED_ROOT" \
    --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --expected-p1-checkpoint "$P1_CHECKPOINT" \
    --expected-p2-checkpoint "$P2_CHECKPOINT" \
    --baseline-prediction-root "$P1_BASELINE_ROOT" \
    --identity-corner-tolerance \
        "${BOXFUSION_P2_IDENTITY_CORNER_TOLERANCE:-0.02}" \
    --identity-score-tolerance \
        "${BOXFUSION_P2_IDENTITY_SCORE_TOLERANCE:-0.02}" \
    --identity-iou-loss-tolerance \
        "${BOXFUSION_P2_IDENTITY_IOU_LOSS_TOLERANCE:-0.05}" \
    > "$REPORT_ROOT/artifact_validation.json"

"$PYTHON" "$ROOT/tools/report_p2_occupancy_recall.py" \
    --scene-list "$SCENE_LIST" \
    --prediction-root "$PRED_ROOT" \
    --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --gt-root "$GT_ROOT" \
    --scans-root "$SCANS_ROOT" \
    --output "$REPORT_ROOT/recall.json" \
    > "$REPORT_ROOT/recall.stdout.json"

echo "P2 audit completed"
echo "  artifact validation: $REPORT_ROOT/artifact_validation.json"
echo "  recall report: $REPORT_ROOT/recall.json"
