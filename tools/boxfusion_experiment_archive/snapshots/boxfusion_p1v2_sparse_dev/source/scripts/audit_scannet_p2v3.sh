#!/usr/bin/env bash
set -euo pipefail

# Validate and report a completed observer-only P2-v3 run.  Ground truth is
# used only by the offline recall report.  A negative go/no-go decision is an
# experimental result, not a script failure, unless REQUIRE_GO=1 is supplied.

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

RUN_TAG="${BOXFUSION_P2V3_RUN_TAG:-${BOXFUSION_P_RUN_TAG:-p2v3_${SCOPE}_b6frozen_v1}}"
PRED_ROOT="${BOXFUSION_P2V3_PRED_ROOT:-${BOXFUSION_P_PRED_ROOT:-$ROOT/results/p_ablation/$RUN_TAG}}"
DIAGNOSTICS_ROOT="${BOXFUSION_P2V3_DIAGNOSTICS_ROOT:-${BOXFUSION_P_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/p_ablation/$RUN_TAG}}"
REPORT_ROOT="${BOXFUSION_P2V3_REPORT_ROOT:-${BOXFUSION_P_REPORT_ROOT:-$ROOT/reports/p_ablation/$RUN_TAG}}"
P1_CHECKPOINT="${BOXFUSION_P1_RESIDUAL_CHECKPOINT:-$ROOT/models/scannet_p1_residual.pt}"
P2_CHECKPOINT="${BOXFUSION_P2_OCCUPANCY_CHECKPOINT:-$ROOT/models/scannet_p2_occupancy_topk.pt}"
GT_ROOT="${BOXFUSION_P2V3_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCANS_ROOT="${BOXFUSION_P2V3_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
MIN_R25_PP="${BOXFUSION_P2V3_MIN_DELTA_R25_PP:-3.0}"
MIN_R50_PP="${BOXFUSION_P2V3_MIN_DELTA_R50_PP:-1.0}"
REQUIRE_GO="${BOXFUSION_P2V3_REQUIRE_GO:-0}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-${CONDA_PREFIX:-/home/admin1/miniconda3/envs/boxfusion2}}"
PYTHON="$ENV_ROOT/bin/python"

case "$REQUIRE_GO" in
    0|1) ;;
    *) echo "BOXFUSION_P2V3_REQUIRE_GO must be 0 or 1" >&2; exit 2 ;;
esac

for path in \
    "$PYTHON" \
    "$SCENE_LIST" \
    "$P1_CHECKPOINT" \
    "$P2_CHECKPOINT" \
    "$ROOT/tools/validate_p2v3_run_artifacts.py" \
    "$ROOT/tools/report_p2v3_reliability_fusion_recall.py"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing P2-v3 audit input: $path" >&2
        exit 1
    fi
done
for directory in \
    "$PRED_ROOT" \
    "$DIAGNOSTICS_ROOT" \
    "$GT_ROOT" \
    "$SCANS_ROOT"; do
    if [[ ! -d "$directory" ]]; then
        echo "Missing P2-v3 audit directory: $directory" >&2
        exit 1
    fi
done

mkdir -p "$REPORT_ROOT"
echo "P2-v3 audit: scope=$SCOPE, tag=$RUN_TAG"
echo "  predictions: $PRED_ROOT"
echo "  diagnostics: $DIAGNOSTICS_ROOT"
echo "  reports: $REPORT_ROOT"
echo "  gate: delta-R25 >= ${MIN_R25_PP}pp and delta-R50 >= ${MIN_R50_PP}pp"

# Do not compare independent formal prediction runs here.  The diagnostics
# themselves enforce applied_count=0/mutation=false, while independent CUDA
# runs have a separately measured small nondeterminism envelope.
"$PYTHON" "$ROOT/tools/validate_p2v3_run_artifacts.py" \
    --scene-list "$SCENE_LIST" \
    --prediction-root "$PRED_ROOT" \
    --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --expected-p1-checkpoint "$P1_CHECKPOINT" \
    --expected-p2-checkpoint "$P2_CHECKPOINT" \
    > "$REPORT_ROOT/base_artifact_validation.json"

"$PYTHON" "$ROOT/tools/report_p2v3_reliability_fusion_recall.py" \
    --scene-list "$SCENE_LIST" \
    --prediction-root "$PRED_ROOT" \
    --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --gt-root "$GT_ROOT" \
    --scans-root "$SCANS_ROOT" \
    --minimum-delta-r25-pp "$MIN_R25_PP" \
    --minimum-delta-r50-pp "$MIN_R50_PP" \
    --output "$REPORT_ROOT/recall.json" \
    > "$REPORT_ROOT/recall.stdout.json"

"$PYTHON" -c \
    'import json,sys; r=json.load(open(sys.argv[1], encoding="utf-8")); g=r["go_no_go"]; print("P2-v3 decision:", g["decision"]); print("  delta R@0.25: %.4f pp" % g["observed_delta_recall_at_025_percentage_points"]); print("  delta R@0.50: %.4f pp" % g["observed_delta_recall_at_050_percentage_points"]); raise SystemExit(0 if g["passed"] or sys.argv[2] == "0" else 3)' \
    "$REPORT_ROOT/recall.json" "$REQUIRE_GO"

echo "P2-v3 audit completed"
echo "  base artifact validation: $REPORT_ROOT/base_artifact_validation.json"
echo "  recall/go-no-go report: $REPORT_ROOT/recall.json"
