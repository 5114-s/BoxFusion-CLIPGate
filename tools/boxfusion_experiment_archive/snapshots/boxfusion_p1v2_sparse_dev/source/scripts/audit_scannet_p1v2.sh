#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
case "${STAGE^^}" in
    P1R)
        STAGE=P1R
        DEFAULT_CHECKPOINT="$ROOT/models/scannet_p1r_snapshot_inside.pt"
        DEFAULT_REFERENCE="/data/ZhaoX/OVM3D-Dett/boxfusion_p1_dev/reports/p_ablation/p1_ablation10_b6frozen_v1/recall.json"
        ;;
    P1S)
        STAGE=P1S
        DEFAULT_CHECKPOINT="$ROOT/models/scannet_p1s_native_sparse.pt"
        DEFAULT_REFERENCE="$ROOT/reports/p1v2_ablation/p1r_ablation10_b6frozen_v1/recall.json"
        ;;
    *)
        echo "Stage must be P1R or P1S" >&2
        exit 2
        ;;
esac

FULL100="${BOXFUSION_P1V2_FULL100:-0}"
if [[ "$FULL100" == "1" ]]; then
    scope=full100
    DEFAULT_SCENES="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
else
    scope=ablation10
    DEFAULT_SCENES="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
fi
TAG="${BOXFUSION_P1V2_RUN_TAG:-${STAGE,,}_${scope}_b6frozen_v1}"
SCENE_LIST="${BOXFUSION_P1V2_SCENE_LIST:-$DEFAULT_SCENES}"
CHECKPOINT="${BOXFUSION_P1V2_CHECKPOINT:-$DEFAULT_CHECKPOINT}"
PRED_ROOT="${BOXFUSION_P1V2_PRED_ROOT:-$ROOT/results/p1v2_ablation/$TAG}"
DIAG_ROOT="${BOXFUSION_P1V2_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/p1v2_ablation/$TAG}"
REPORT_ROOT="${BOXFUSION_P1V2_REPORT_ROOT:-$ROOT/reports/p1v2_ablation/$TAG}"
REFERENCE="${BOXFUSION_P1V2_REFERENCE_REPORT:-$DEFAULT_REFERENCE}"
P0_ROOT="${BOXFUSION_P0_PRED_ROOT:-$ROOT/results/p_ablation/p0_${scope}_b6frozen_v1}"
PYTHON="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}/bin/python"

for path in "$PYTHON" "$SCENE_LIST" "$CHECKPOINT" "$REFERENCE"; do
    [[ -f "$path" ]] || {
        echo "Missing P1-v2 audit input: $path" >&2
        exit 1
    }
done
for path in "$PRED_ROOT" "$DIAG_ROOT"; do
    [[ -d "$path" ]] || {
        echo "Missing P1-v2 artifact directory: $path" >&2
        exit 1
    }
done
mkdir -p "$REPORT_ROOT"

"$PYTHON" "$ROOT/tools/validate_p1v2_run_artifacts.py" \
    --stage "$STAGE" \
    --scene-list "$SCENE_LIST" \
    --prediction-root "$PRED_ROOT" \
    --diagnostics-root "$DIAG_ROOT" \
    --expected-checkpoint "$CHECKPOINT" \
    --output "$REPORT_ROOT/artifact_validation.json"

# Cross-run byte identity is informative, not the safety gate: the frozen
# upstream B6 CUDA path has a measured P0-repeat drift.  The artifact validator
# above enforces the observer mutation/applied-count contract.
if [[ -d "$P0_ROOT" ]]; then
    set +e
    "$PYTHON" "$ROOT/tools/verify_p1_identity.py" \
        --baseline-root "$P0_ROOT" \
        --observer-root "$PRED_ROOT" \
        --diagnostics-root "$DIAG_ROOT" \
        --output "$REPORT_ROOT/cross_run_identity.json"
    identity_status=$?
    set -e
    echo "P1-v2 cross-run byte identity status: $identity_status (informational)"
fi

set +e
"$PYTHON" "$ROOT/tools/report_p1v2_recall.py" \
    --stage "$STAGE" \
    --scene-list "$SCENE_LIST" \
    --prediction-root "$PRED_ROOT" \
    --diagnostics-root "$DIAG_ROOT" \
    --gt-root "${BOXFUSION_P1_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}" \
    --scans-root "${BOXFUSION_P1_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}" \
    --reference-report "$REFERENCE" \
    --maximum-runtime-seconds-per-scene "${BOXFUSION_P1V2_MAX_RUNTIME_PER_SCENE:-0.80}" \
    --maximum-candidates-per-scene "${BOXFUSION_P1V2_MAX_CANDIDATES_PER_SCENE:-256}" \
    --output "$REPORT_ROOT/recall.json"
report_status=$?
set -e
if [[ "$report_status" -ne 0 && "$report_status" -ne 3 ]]; then
    exit "$report_status"
fi
exit "$report_status"
