#!/usr/bin/env bash
set -euo pipefail

# Read-only by default:
#   bash scripts/audit_scannet_p1_nondeterminism.sh
#
# Persist a report only when explicitly requested:
#   bash scripts/audit_scannet_p1_nondeterminism.sh reports/p1_nondet.json
#
# All paths/tags can be overridden through the variables below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_ROOT="${BOXFUSION_P1_SOURCE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_p1_dev}"
FULL100="${BOXFUSION_P_FULL100:-0}"
case "$FULL100" in
    0) SCOPE=ablation10 ;;
    1) SCOPE=full100 ;;
    *) echo "BOXFUSION_P_FULL100 must be 0 or 1" >&2; exit 2 ;;
esac

if [[ -n "${BOXFUSION_P_SCENE_LIST:-}" ]]; then
    SCENE_LIST="$BOXFUSION_P_SCENE_LIST"
elif [[ "$FULL100" == 1 ]]; then
    SCENE_LIST="$SOURCE_ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
else
    SCENE_LIST="$SOURCE_ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
fi

P0_TAG="${BOXFUSION_P0_RUN_TAG:-p0_${SCOPE}_b6frozen_v1}"
P0_REPEAT_TAG="${BOXFUSION_P0_REPEAT_RUN_TAG:-p0_${SCOPE}_b6frozen_repeat_v1}"
P1_TAG="${BOXFUSION_P1_RUN_TAG:-p1_${SCOPE}_b6frozen_v1}"

P0_ROOT="${BOXFUSION_P0_PRED_ROOT:-$SOURCE_ROOT/results/p_ablation/$P0_TAG}"
P0_LOG_ROOT="${BOXFUSION_P0_LOG_ROOT:-$SOURCE_ROOT/logs/p_ablation/$P0_TAG}"
P0_REPEAT_ROOT="${BOXFUSION_P0_REPEAT_PRED_ROOT:-$SOURCE_ROOT/results/p_ablation/$P0_REPEAT_TAG}"
P0_REPEAT_LOG_ROOT="${BOXFUSION_P0_REPEAT_LOG_ROOT:-$SOURCE_ROOT/logs/p_ablation/$P0_REPEAT_TAG}"
P1_ROOT="${BOXFUSION_P1_PRED_ROOT:-$SOURCE_ROOT/results/p_ablation/$P1_TAG}"
P1_LOG_ROOT="${BOXFUSION_P1_LOG_ROOT:-$SOURCE_ROOT/logs/p_ablation/$P1_TAG}"
P1_DIAGNOSTICS_ROOT="${BOXFUSION_P1_DIAGNOSTICS_ROOT:-$SOURCE_ROOT/diagnostics/p_ablation/$P1_TAG}"
PYTHON="${BOXFUSION_ENV_ROOT:-${CONDA_PREFIX:-/home/admin1/miniconda3/envs/boxfusion2}}/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Python executable: $PYTHON" >&2
    exit 1
fi

ARGS=(
    "$ROOT/tools/audit_p1_nondeterminism.py"
    --scene-list "$SCENE_LIST"
    --p0-root "$P0_ROOT"
    --p0-manifest "$P0_LOG_ROOT/run_manifest.json"
    --p0-eval-log "$P0_LOG_ROOT/eval_stdout.log"
    --p0-repeat
        "$P0_REPEAT_ROOT"
        "$P0_REPEAT_LOG_ROOT/run_manifest.json"
        "$P0_REPEAT_LOG_ROOT/eval_stdout.log"
    --p1-root "$P1_ROOT"
    --p1-diagnostics-root "$P1_DIAGNOSTICS_ROOT"
    --p1-manifest "$P1_LOG_ROOT/run_manifest.json"
    --p1-eval-log "$P1_LOG_ROOT/eval_stdout.log"
    --match-iou "${BOXFUSION_P_NONDET_MATCH_IOU:-0.25}"
    --iou-thresholds 0.15 0.25 0.50
    --trusted-local-pickles
)

OUTPUT="${1:-${BOXFUSION_P_NONDET_REPORT:-}}"
if [[ -n "$OUTPUT" ]]; then
    ARGS+=(--output "$OUTPUT")
fi
if [[ "${BOXFUSION_P_NONDET_REQUIRE_ENVELOPE:-0}" == 1 ]]; then
    ARGS+=(--require-within-envelope)
fi
if [[ "${BOXFUSION_P_NONDET_REQUIRE_BIT_EXACT:-0}" == 1 ]]; then
    ARGS+=(--require-bit-exact)
fi

exec "$PYTHON" "${ARGS[@]}"
