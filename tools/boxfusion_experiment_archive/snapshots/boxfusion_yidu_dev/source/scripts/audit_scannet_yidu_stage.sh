#!/usr/bin/env bash
set -euo pipefail

# CPU-only audit for one already completed YiDu observer stage.
# It performs, in order:
#   1) exact structure plus an explicit cross-run numerical envelope;
#   2) ground-truth-free candidate export;
#   3) retrospective ScanNet oracle reporting.
#
# Example:
#   bash scripts/audit_scannet_yidu_stage.sh A1

STAGE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
ENV_ROOT="${BOXFUSION_YIDU_ENV_ROOT:-${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion2}}"
PYTHON="$ENV_ROOT/bin/python"

case "${STAGE^^}" in
    A1|A2|A3|A4|A5|A6) STAGE="${STAGE^^}" ;;
    *) echo "Stage must be one of A1,A2,A3,A4,A5,A6" >&2; exit 2 ;;
esac

B0_TAG="${BOXFUSION_YIDU_B0_TAG:-yidu_b0_ablation10_frozen_v1}"
STAGE_TAG="${BOXFUSION_YIDU_STAGE_TAG:-yidu_${STAGE,,}_ablation10_observer_v1}"
SCENE_LIST="${BOXFUSION_YIDU_AUDIT_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
B0_ROOT="${BOXFUSION_YIDU_B0_PRED_ROOT:-$ROOT/results/yidu_ablation/$B0_TAG}"
STAGE_ROOT="${BOXFUSION_YIDU_STAGE_PRED_ROOT:-$ROOT/results/yidu_ablation/$STAGE_TAG}"
DIAGNOSTICS_ROOT="${BOXFUSION_YIDU_STAGE_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/yidu_ablation/$STAGE_TAG}"
GEOMETRY_ROOT="${BOXFUSION_YIDU_STAGE_GEOMETRY_ROOT:-$ROOT/datasets/yidu_ablation/${STAGE_TAG}_geometry}"
REPORT="${BOXFUSION_YIDU_STAGE_REPORT:-$ROOT/reports/yidu_ablation/${STAGE_TAG}_oracle.json}"
DELTA_REPORT="${BOXFUSION_YIDU_STAGE_DELTA_REPORT:-$ROOT/reports/yidu_ablation/${STAGE_TAG}_candidate_deltas.json}"
GT_ROOT="${BOXFUSION_YIDU_GT_ROOT:-$LIVE_ROOT/evaluation/data_util/scannet_train_detection_data}"
SCAN_ROOT="${BOXFUSION_YIDU_SCAN_ROOT:-/extra/ZhaoX/scannet_data/scans}"
IDENTITY_MODE="${BOXFUSION_YIDU_IDENTITY_MODE:-numeric_envelope}"
IDENTITY_MAX_CORNER_ABS="${BOXFUSION_YIDU_IDENTITY_MAX_CORNER_ABS:-0.02}"
IDENTITY_MAX_SCORE_ABS="${BOXFUSION_YIDU_IDENTITY_MAX_SCORE_ABS:-0.02}"
IDENTITY_MAX_IOU_LOSS="${BOXFUSION_YIDU_IDENTITY_MAX_IOU_LOSS:-0.05}"
REQUIRE_ZERO_WRITE="${BOXFUSION_YIDU_REQUIRE_ZERO_WRITE:-1}"

case "$IDENTITY_MODE" in
    strict_bitwise|numeric_envelope) ;;
    *)
        echo "BOXFUSION_YIDU_IDENTITY_MODE must be strict_bitwise or numeric_envelope" >&2
        exit 2
        ;;
esac
case "$REQUIRE_ZERO_WRITE" in
    0|1) ;;
    *)
        echo "BOXFUSION_YIDU_REQUIRE_ZERO_WRITE must be 0 or 1" >&2
        exit 2
        ;;
esac
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
for value in \
    "$IDENTITY_MAX_CORNER_ABS" \
    "$IDENTITY_MAX_SCORE_ABS" \
    "$IDENTITY_MAX_IOU_LOSS"; do
    "$PYTHON" -c \
        'import math, sys; value=float(sys.argv[1]); assert math.isfinite(value) and value >= 0.0' \
        "$value" \
        >/dev/null 2>&1 \
        || { echo "YiDu identity envelope values must be finite and non-negative" >&2; exit 2; }
done
"$PYTHON" -c \
    'import sys; assert float(sys.argv[1]) <= 1.0' \
    "$IDENTITY_MAX_IOU_LOSS" \
    >/dev/null 2>&1 \
    || { echo "YiDu identity IoU-loss tolerance must not exceed 1" >&2; exit 2; }

for directory in \
    "$B0_ROOT" "$STAGE_ROOT" "$DIAGNOSTICS_ROOT" \
    "$GT_ROOT" "$SCAN_ROOT"; do
    [[ -d "$directory" ]] \
        || { echo "Missing audit input directory: $directory" >&2; exit 1; }
done
[[ -s "$SCENE_LIST" ]] \
    || { echo "Missing audit scene list: $SCENE_LIST" >&2; exit 1; }
[[ ! -e "$GEOMETRY_ROOT" ]] \
    || { echo "Choose a fresh geometry output: $GEOMETRY_ROOT" >&2; exit 1; }
[[ ! -e "$REPORT" ]] \
    || { echo "Choose a fresh report output: $REPORT" >&2; exit 1; }
[[ ! -e "$DELTA_REPORT" ]] \
    || { echo "Choose a fresh delta report: $DELTA_REPORT" >&2; exit 1; }

echo "YiDu $STAGE fixed-stage CPU audit"
echo "  B0 predictions: $B0_ROOT"
echo "  stage predictions: $STAGE_ROOT"
echo "  diagnostics: $DIAGNOSTICS_ROOT"
echo "  candidate output: $GEOMETRY_ROOT"
echo "  oracle report: $REPORT"
echo "  candidate delta report: $DELTA_REPORT"
echo "  cross-run identity mode: $IDENTITY_MODE"
if [[ "$IDENTITY_MODE" == "numeric_envelope" ]]; then
    echo "  numeric envelope: corner=$IDENTITY_MAX_CORNER_ABS, score=$IDENTITY_MAX_SCORE_ABS, IoU-loss=$IDENTITY_MAX_IOU_LOSS"
fi
echo "  require in-process zero-write proof: $REQUIRE_ZERO_WRITE"
if [[ "$STAGE" == "A1" || "$STAGE" == "A2" ]]; then
    echo "  note: $STAGE is a point-cleaning observer and does not emit box candidates; candidates=0 is expected"
fi

IDENTITY_ARGS=(
    --baseline-root "$B0_ROOT"
    --observer-root "$STAGE_ROOT"
    --diagnostics-root "$DIAGNOSTICS_ROOT"
    --expected-stage "$STAGE"
)
if [[ "$IDENTITY_MODE" == "numeric_envelope" ]]; then
    IDENTITY_ARGS+=(
        --max-corner-abs "$IDENTITY_MAX_CORNER_ABS"
        --max-score-abs "$IDENTITY_MAX_SCORE_ABS"
        --max-matched-iou-loss "$IDENTITY_MAX_IOU_LOSS"
    )
fi
if [[ "$REQUIRE_ZERO_WRITE" == "1" ]]; then
    IDENTITY_ARGS+=(--require-zero-write)
fi
"$PYTHON" "$ROOT/tools/verify_yidu_identity.py" "${IDENTITY_ARGS[@]}"

"$PYTHON" "$ROOT/tools/export_yidu_geometry_candidates.py" \
    --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --prediction-root "$STAGE_ROOT" \
    --scene-list "$SCENE_LIST" \
    --output-root "$GEOMETRY_ROOT" \
    --stage "$STAGE"

"$PYTHON" "$ROOT/tools/report_trifusion_oracles.py" \
    --pred-root "$STAGE_ROOT" \
    --scene-list "$SCENE_LIST" \
    --gt-root "$GT_ROOT" \
    --scan-root "$SCAN_ROOT" \
    --geometry-candidates-root "$GEOMETRY_ROOT" \
    --output "$REPORT"

"$PYTHON" "$ROOT/tools/report_yidu_candidate_deltas.py" \
    --geometry-root "$GEOMETRY_ROOT" \
    --prediction-root "$STAGE_ROOT" \
    --scene-list "$SCENE_LIST" \
    --gt-root "$GT_ROOT" \
    --scan-root "$SCAN_ROOT" \
    --output "$DELTA_REPORT"

echo "YiDu $STAGE audit completed. Oracle output is diagnostic only."
