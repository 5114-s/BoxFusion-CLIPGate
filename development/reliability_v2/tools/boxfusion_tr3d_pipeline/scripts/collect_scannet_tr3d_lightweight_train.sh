#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-6}"
GPU_SPEC="${2:-0,1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ "$STAGE" =~ ^[1-6]$ ]] || { echo "stage must be 1..6" >&2; exit 2; }
TAG="${BOXFUSION_LIGHTWEIGHT_TRAIN_TAG:-tr3d_lightweight_l${STAGE}_train100_v1}"
TRAIN_LIST="${BOXFUSION_LIGHTWEIGHT_TRAIN_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
FRAMES_ROOT="${BOXFUSION_LIGHTWEIGHT_TRAIN_FRAMES_ROOT:-$ROOT/data/scannet_train}"
PREFIX_MANIFEST="${BOXFUSION_LIGHTWEIGHT_TRAIN_PREFIX_MANIFEST:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/data/tr3d_prefix_train100_boxfusion_causal_p100_v1/manifests/trajectory_prefix_train100_boxfusion_causal_p100_v1.jsonl}"
PARENT_CACHE="${BOXFUSION_LIGHTWEIGHT_TRAIN_PARENT_CACHE:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/cache/tr3d_prefix_boxfusion_causal_p100_train100_v1}"
YOLOE="${BOXFUSION_YOLOE_CHECKPOINT:-/data/ZhaoX/OVM3D-Dett/boxfusion_stage3_dev/models/yoloe-11s-seg-pf.pt}"

for path in "$TRAIN_LIST" "$PREFIX_MANIFEST" "$YOLOE"; do
    [[ -f "$path" && ! -L "$path" ]] || { echo "Missing train input: $path" >&2; exit 2; }
done
for path in "$FRAMES_ROOT" "$PARENT_CACHE"; do
    [[ -d "$path" && ! -L "$path" ]] || { echo "Missing train root: $path" >&2; exit 2; }
done

echo "Lightweight train-only observer: L$STAGE, tag=$TAG, GPUs=$GPU_SPEC"
BOXFUSION_B6_BOXER_SCENE_LIST="$TRAIN_LIST" \
BOXFUSION_TR3D_TERMINAL_RUN_TAG="$TAG" \
BOXFUSION_SCANNET_FRAMES_ROOT="$FRAMES_ROOT" \
BOXFUSION_R3_PREFIX_MANIFEST="$PREFIX_MANIFEST" \
BOXFUSION_R3_PARENT_CACHE_ROOT="$PARENT_CACHE" \
BOXFUSION_YOLOE_CHECKPOINT="$YOLOE" \
BOXFUSION_INCREMENTAL_TR3D_OBSERVER=1 \
BOXFUSION_TR3D_LIGHTWEIGHT_FUSION=1 \
BOXFUSION_TR3D_LIGHTWEIGHT_STAGE="$STAGE" \
BOXFUSION_TR3D_INCREMENTAL_EVERY_KEYFRAMES="${BOXFUSION_TR3D_INCREMENTAL_EVERY_KEYFRAMES:-5}" \
BOXFUSION_PROPOSAL_CACHE_MODE_OVERRIDE=disabled \
BOXFUSION_SKIP_EVALUATION=1 \
    bash "$ROOT/scripts/run_scannet_b6_g0_tr3d_terminal_active.sh" "$GPU_SPEC"

list_sha="$(sha256sum "$TRAIN_LIST" | awk '{print $1}')"
scope="$(basename "$TRAIN_LIST" .txt)-${list_sha:0:12}"
diagnostics="$ROOT/diagnostics/b6_g0_tr3d_terminal/$TAG/$scope/online/tr3d_incremental"
count="$(find "$diagnostics" -maxdepth 1 -type f -name 'scene*_tr3d_incremental.json' | wc -l)"
[[ "$count" == "100" ]] || { echo "Expected 100 diagnostics, found $count" >&2; exit 1; }
echo "Completed L$STAGE train diagnostics: $diagnostics"
