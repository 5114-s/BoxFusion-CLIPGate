#!/usr/bin/env bash
set -euo pipefail

# Live, train-only incremental TR3D diagnostics. No GT is read here.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
GPU_SPEC="${1:-0,1}"
TAG="${BOXFUSION_INCREMENTAL_TRAIN_TAG:-incremental_tr3d_train100_v1}"
EXPECTED_SCENES="${BOXFUSION_INCREMENTAL_TRAIN_EXPECTED_SCENES:-100}"
TRAIN_LIST="${BOXFUSION_INCREMENTAL_TRAIN_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
FRAMES_ROOT="${BOXFUSION_INCREMENTAL_TRAIN_FRAMES_ROOT:-$ROOT/data/scannet_train}"
PREFIX_MANIFEST="${BOXFUSION_INCREMENTAL_TRAIN_PREFIX_MANIFEST:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/data/tr3d_prefix_train100_boxfusion_causal_p100_v1/manifests/trajectory_prefix_train100_boxfusion_causal_p100_v1.jsonl}"
PARENT_CACHE="${BOXFUSION_INCREMENTAL_TRAIN_PARENT_CACHE:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/cache/tr3d_prefix_boxfusion_causal_p100_train100_v1}"
YOLOE="${BOXFUSION_YOLOE_CHECKPOINT:-/data/ZhaoX/OVM3D-Dett/boxfusion_stage3_dev/models/yoloe-11s-seg-pf.pt}"

for path in "$TRAIN_LIST" "$PREFIX_MANIFEST" "$YOLOE"; do
    [[ -f "$path" && ! -L "$path" ]] || { echo "Missing train-only input: $path" >&2; exit 2; }
done
for path in "$FRAMES_ROOT" "$PARENT_CACHE"; do
    [[ -d "$path" && ! -L "$path" ]] || { echo "Missing train-only root: $path" >&2; exit 2; }
done
scene_count="$(awk 'NF && $1 !~ /^#/ {n++} END {print n+0}' "$TRAIN_LIST")"
[[ "$EXPECTED_SCENES" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid expected scene count" >&2; exit 2; }
[[ "$scene_count" == "$EXPECTED_SCENES" ]] || { echo "Expected $EXPECTED_SCENES train scenes, got $scene_count" >&2; exit 2; }

echo "Train-only causal incremental TR3D collection"
echo "  scenes/tag: $scene_count / $TAG"
echo "  frames: $FRAMES_ROOT"
echo "  mutation/GT/evaluation: disabled/none/skipped"

BOXFUSION_B6_BOXER_SCENE_LIST="$TRAIN_LIST" \
BOXFUSION_TR3D_TERMINAL_RUN_TAG="$TAG" \
BOXFUSION_SCANNET_FRAMES_ROOT="$FRAMES_ROOT" \
BOXFUSION_R3_PREFIX_MANIFEST="$PREFIX_MANIFEST" \
BOXFUSION_R3_PARENT_CACHE_ROOT="$PARENT_CACHE" \
BOXFUSION_YOLOE_CHECKPOINT="$YOLOE" \
BOXFUSION_INCREMENTAL_TR3D_OBSERVER=1 \
BOXFUSION_TR3D_INCREMENTAL_EVERY_KEYFRAMES="${BOXFUSION_TR3D_INCREMENTAL_EVERY_KEYFRAMES:-5}" \
BOXFUSION_PROPOSAL_CACHE_MODE_OVERRIDE=disabled \
BOXFUSION_SKIP_EVALUATION=1 \
    bash "$ROOT/scripts/run_scannet_b6_g0_tr3d_terminal_active.sh" "$GPU_SPEC"

list_sha="$(sha256sum "$TRAIN_LIST" | awk '{print $1}')"
scope="$(basename "$TRAIN_LIST" .txt)-${list_sha:0:12}"
diagnostics="$ROOT/diagnostics/b6_g0_tr3d_terminal/$TAG/$scope/online/tr3d_incremental"
count="$(find "$diagnostics" -maxdepth 1 -type f -name 'scene*_tr3d_incremental.json' | wc -l)"
[[ "$count" == "$EXPECTED_SCENES" ]] || { echo "Expected $EXPECTED_SCENES incremental diagnostics, found $count" >&2; exit 1; }
echo "Train-only incremental diagnostics completed: $diagnostics"
