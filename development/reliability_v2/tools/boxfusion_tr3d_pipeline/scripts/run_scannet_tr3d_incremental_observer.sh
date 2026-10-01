#!/usr/bin/env bash
set -euo pipefail

# Causal incremental TR3D observer followed by a strictly offline GT audit.
# The online pass is immutable: incremental proposals are never written into
# BoxFusion predictions.  Use a new tag for every formal run.

GPU_SPEC="${1:-0,1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TAG="${BOXFUSION_INCREMENTAL_TR3D_RUN_TAG:-incremental_tr3d_observer_fixed10_v1}"
SCENE_LIST="${BOXFUSION_INCREMENTAL_TR3D_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
YOLOE_CHECKPOINT="${BOXFUSION_YOLOE_CHECKPOINT:-/data/ZhaoX/OVM3D-Dett/boxfusion_stage3_dev/models/yoloe-11s-seg-pf.pt}"
GT_ROOT="${BOXFUSION_INCREMENTAL_TR3D_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCANS_ROOT="${BOXFUSION_INCREMENTAL_TR3D_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
INTERVAL="${BOXFUSION_TR3D_INCREMENTAL_EVERY_KEYFRAMES:-5}"

if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]]; then
    echo "Invalid BOXFUSION_INCREMENTAL_TR3D_RUN_TAG: $TAG" >&2
    exit 1
fi
for path in "$SCENE_LIST" "$YOLOE_CHECKPOINT" "$GT_ROOT" "$SCANS_ROOT"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing incremental-TR3D input: $path" >&2
        exit 1
    fi
done

list_sha="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
scope="$(basename "$SCENE_LIST" .txt)-${list_sha:0:12}"
diagnostics="$ROOT/diagnostics/b6_g0_tr3d_terminal/$TAG/$scope/online/tr3d_incremental"
baseline="$ROOT/results/b6_g0_tr3d_terminal_same_run_baseline/$TAG/$scope"
report="$ROOT/reports/tr3d_incremental_online/$TAG/$scope/recall_audit.json"

echo "Causal incremental TR3D observer"
echo "  tag/scenes: $TAG / $SCENE_LIST"
echo "  GPUs/interval: $GPU_SPEC / every $INTERVAL keyframes"
echo "  mutation: disabled (observer-only)"
echo "  diagnostics: $diagnostics"
echo "  offline audit: $report"

BOXFUSION_B6_BOXER_SCENE_LIST="$SCENE_LIST" \
BOXFUSION_TR3D_TERMINAL_RUN_TAG="$TAG" \
BOXFUSION_YOLOE_CHECKPOINT="$YOLOE_CHECKPOINT" \
BOXFUSION_INCREMENTAL_TR3D_OBSERVER=1 \
BOXFUSION_TR3D_INCREMENTAL_EVERY_KEYFRAMES="$INTERVAL" \
    bash "$ROOT/scripts/run_scannet_b6_g0_tr3d_terminal_active.sh" "$GPU_SPEC"

"/home/admin1/miniconda3/envs/boxfusion2/bin/python" \
    "$ROOT/tools/audit_tr3d_incremental_observer.py" \
    --diagnostics-root "$diagnostics" \
    --baseline-root "$baseline" \
    --scene-list "$SCENE_LIST" \
    --ground-truth-root "$GT_ROOT" \
    --scans-root "$SCANS_ROOT" \
    --output "$report"

echo "Incremental TR3D observer and offline audit completed: $report"
