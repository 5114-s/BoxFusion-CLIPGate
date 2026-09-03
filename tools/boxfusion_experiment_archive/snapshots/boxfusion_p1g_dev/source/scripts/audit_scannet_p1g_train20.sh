#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_ROOT="${BOXFUSION_P1G_SOURCE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_p1_dev}"
SOURCE_TAG="${BOXFUSION_P1G_SOURCE_TAG:-p1_residual_inputs_train100_v1}"
SCENE_LIST="${BOXFUSION_P1G_AUDIT_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1g_audit20.txt}"
P1S_CHECKPOINT="${BOXFUSION_P1G_P1S_CHECKPOINT:-$ROOT/models/scannet_p1s_native_sparse.pt}"
P1G_CHECKPOINT="${BOXFUSION_P1G_CHECKPOINT:-$ROOT/models/scannet_p1g_aligned_geometry.pt}"
DIAGNOSTICS="${BOXFUSION_P1G_DIAGNOSTICS:-$SOURCE_ROOT/diagnostics/p1_training/$SOURCE_TAG}"
PREDICTIONS="${BOXFUSION_P1G_PREDICTIONS:-$SOURCE_ROOT/results/p1_training/$SOURCE_TAG}"
GT_ROOT="${BOXFUSION_P1G_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCANS_ROOT="${BOXFUSION_P1G_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
OUTPUT="${BOXFUSION_P1G_AUDIT_OUTPUT:-$ROOT/reports/p1g_audit/p1g_audit20_v1.json}"
PYTHON="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}/bin/python"

for path in "$PYTHON" "$SCENE_LIST" "$P1S_CHECKPOINT" "$P1G_CHECKPOINT"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing P1G audit input: $path" >&2
        exit 1
    fi
done
for path in "$DIAGNOSTICS" "$PREDICTIONS" "$GT_ROOT" "$SCANS_ROOT"; do
    if [[ ! -d "$path" ]]; then
        echo "Missing P1G audit directory: $path" >&2
        exit 1
    fi
done
mkdir -p "$(dirname "$OUTPUT")"

echo "P1G one-shot train-only audit20"
echo "  scenes: $SCENE_LIST"
echo "  frozen P1S: $P1S_CHECKPOINT"
echo "  frozen P1G: $P1G_CHECKPOINT"
echo "  output: $OUTPUT"
echo "A pass only authorizes the preselected fresh50 audit."

set +e
"$PYTHON" "$ROOT/tools/evaluate_p1g_candidate_audit.py" \
    --stage module20 \
    --scene-list "$SCENE_LIST" \
    --p1s-checkpoint "$P1S_CHECKPOINT" \
    --p1g-checkpoint "$P1G_CHECKPOINT" \
    --source-diagnostics-root "$DIAGNOSTICS" \
    --prediction-root "$PREDICTIONS" \
    --gt-root "$GT_ROOT" \
    --scans-root "$SCANS_ROOT" \
    --device "${BOXFUSION_P1G_AUDIT_DEVICE:-cpu}" \
    --maximum-refiner-seconds-per-scene "${BOXFUSION_P1G_MAX_REFINER_SECONDS_PER_SCENE:-0.15}" \
    --maximum-refiner-p95-seconds-per-scene "${BOXFUSION_P1G_MAX_REFINER_P95_SECONDS_PER_SCENE:-0.30}" \
    --output "$OUTPUT"
status=$?
set -e
if [[ "$status" -ne 0 && "$status" -ne 3 ]]; then
    exit "$status"
fi
exit "$status"
