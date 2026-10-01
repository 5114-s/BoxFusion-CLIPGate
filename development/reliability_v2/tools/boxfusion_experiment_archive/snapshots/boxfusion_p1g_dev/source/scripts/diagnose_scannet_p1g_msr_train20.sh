#!/usr/bin/env bash
set -euo pipefail

# Paired, train-only causal diagnostic for P1G.
#
# The two replays use the exact same frozen B6/P1S candidates.  The second
# replay relaxes both the face clamps and the internal MSR evidence gates.
# Neither replay can write formal BoxFusion predictions.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_ROOT="${BOXFUSION_P1G_SOURCE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_p1_dev}"
SOURCE_TAG="${BOXFUSION_P1G_SOURCE_TAG:-p1_residual_inputs_train100_v1}"
SCENE_LIST="${BOXFUSION_P1G_DIAG_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1g_audit20.txt}"
FORBIDDEN_LIST="${BOXFUSION_P1G_FORBIDDEN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
DIAGNOSTICS_ROOT="${BOXFUSION_P1G_SOURCE_DIAGNOSTICS:-$SOURCE_ROOT/diagnostics/p1_training/$SOURCE_TAG}"
PREDICTION_ROOT="${BOXFUSION_P1G_SOURCE_PREDICTIONS:-$SOURCE_ROOT/results/p1_training/$SOURCE_TAG}"
GT_ROOT="${BOXFUSION_P1G_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCANS_ROOT="${BOXFUSION_P1G_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
FRAMES_ROOT="${BOXFUSION_P1G_FRAMES_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_b6_dev/data/scannet_train}"
P1S_CHECKPOINT="${BOXFUSION_P1G_P1S_CHECKPOINT:-$ROOT/models/scannet_p1s_native_sparse.pt}"
B6_CHECKPOINT="${BOXFUSION_P1G_B6_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
PYTHON="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}/bin/python"
RUN_TAG="${BOXFUSION_P1G_DIAG_TAG:-audit20_pair_v1}"
MINIMUM_CROSS="${BOXFUSION_P1G_MINIMUM_CROSS_IOU50:-5}"

if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "BOXFUSION_P1G_DIAG_TAG contains unsafe path characters" >&2
    exit 2
fi

for path in \
    "$PYTHON" "$SCENE_LIST" "$FORBIDDEN_LIST" "$P1S_CHECKPOINT" \
    "$B6_CHECKPOINT" "$ROOT/tools/replay_p1g_train_msr.py" \
    "$ROOT/tools/diagnose_p1g_train_msr.py"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing P1G diagnostic input: $path" >&2
        exit 1
    fi
done
for directory in \
    "$DIAGNOSTICS_ROOT" "$PREDICTION_ROOT" "$GT_ROOT" \
    "$SCANS_ROOT" "$FRAMES_ROOT"; do
    if [[ ! -d "$directory" ]]; then
        echo "Missing P1G diagnostic directory: $directory" >&2
        exit 1
    fi
done

CONSERVATIVE_OUTPUT="$ROOT/diagnostics/p1g_train_msr/${RUN_TAG}_conservative"
PERMISSIVE_OUTPUT="$ROOT/diagnostics/p1g_train_msr/${RUN_TAG}_permissive"
CONSERVATIVE_SUMMARY="$ROOT/reports/p1g_train_msr_${RUN_TAG}_conservative.json"
PERMISSIVE_SUMMARY="$ROOT/reports/p1g_train_msr_${RUN_TAG}_permissive.json"
DIAGNOSIS="$ROOT/reports/p1g_train_msr_${RUN_TAG}_diagnosis.json"

COMMON_ARGS=(
    --scene-list "$SCENE_LIST"
    --forbidden-scene-list "$FORBIDDEN_LIST"
    --diagnostics-root "$DIAGNOSTICS_ROOT"
    --prediction-root "$PREDICTION_ROOT"
    --gt-root "$GT_ROOT"
    --scans-root "$SCANS_ROOT"
    --frames-root "$FRAMES_ROOT"
    --p1s-checkpoint "$P1S_CHECKPOINT"
    --b6-checkpoint "$B6_CHECKPOINT"
)

echo "P1G paired train-only diagnostic: tag=$RUN_TAG"
echo "  scenes: $SCENE_LIST"
echo "  formal prediction mutation: disabled"

"$PYTHON" "$ROOT/tools/replay_p1g_train_msr.py" \
    "${COMMON_ARGS[@]}" \
    --output-root "$CONSERVATIVE_OUTPUT" \
    --summary-json "$CONSERVATIVE_SUMMARY" \
    --msr-evidence-profile conservative \
    --msr-max-face-shift-ratio 0.18 \
    --msr-min-extent-ratio 0.70 \
    --msr-max-extent-ratio 1.25 \
    --msr-max-center-shift-ratio 0.15

"$PYTHON" "$ROOT/tools/replay_p1g_train_msr.py" \
    "${COMMON_ARGS[@]}" \
    --output-root "$PERMISSIVE_OUTPUT" \
    --summary-json "$PERMISSIVE_SUMMARY" \
    --msr-evidence-profile permissive \
    --msr-max-face-shift-ratio 0.50 \
    --msr-min-extent-ratio 0.50 \
    --msr-max-extent-ratio 1.75 \
    --msr-max-center-shift-ratio 0.50

"$PYTHON" "$ROOT/tools/diagnose_p1g_train_msr.py" \
    --conservative-summary "$CONSERVATIVE_SUMMARY" \
    --permissive-summary "$PERMISSIVE_SUMMARY" \
    --minimum-cross-iou50 "$MINIMUM_CROSS" \
    --output "$DIAGNOSIS"

echo "P1G causal diagnosis: $DIAGNOSIS"
