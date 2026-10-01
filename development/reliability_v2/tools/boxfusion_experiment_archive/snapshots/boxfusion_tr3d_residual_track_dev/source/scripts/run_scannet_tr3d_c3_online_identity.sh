#!/usr/bin/env bash
set -euo pipefail

# C3 online identity observer on top of the frozen
# B6 + Selective-Boxer G0 + terminal-R3 path.
#
# Default: one-scene smoke test.
# Fixed 10:
#   BOXFUSION_C3_ONLINE_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt" \
#   BOXFUSION_C3_ONLINE_RUN_TAG=c3_online_identity_fixed10_v1 \
#     bash scripts/run_scannet_tr3d_c3_online_identity.sh 0,1
# Full 100 (only after fixed10 audit passes):
#   BOXFUSION_C3_ONLINE_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
#   BOXFUSION_C3_ONLINE_RUN_TAG=c3_online_identity_full100_v1 \
#     bash scripts/run_scannet_tr3d_c3_online_identity.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENE_LIST="${BOXFUSION_C3_ONLINE_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_smoke_scene0277_00.txt}"
RUN_TAG="${BOXFUSION_C3_ONLINE_RUN_TAG:-c3_online_identity_smoke1_v1}"
C2_CACHE_ROOT="${BOXFUSION_C3_ONLINE_C2_CACHE_ROOT:-$ROOT/artifacts/tr3d_c2_maskrgbd/c2_c1top10_full100_v1/cache}"
PARENT_CACHE_ROOT="${BOXFUSION_C3_ONLINE_PARENT_CACHE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/cache/tr3d_prefix_boxfusion_causal_p100_full100_v3}"

if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]]; then
    echo "Invalid BOXFUSION_C3_ONLINE_RUN_TAG: $RUN_TAG" >&2
    exit 1
fi
for required_file in "$SCENE_LIST"; do
    if [[ ! -f "$required_file" || -L "$required_file" ]]; then
        echo "Missing/non-regular C3 online input file: $required_file" >&2
        exit 1
    fi
done
for required_root in "$C2_CACHE_ROOT" "$PARENT_CACHE_ROOT"; do
    if [[ ! -d "$required_root" || -L "$required_root" ]]; then
        echo "Missing/non-regular C3 online input root: $required_root" >&2
        exit 1
    fi
done

scene_count="$(awk 'NF && $1 !~ /^#/ {count += 1} END {print count + 0}' "$SCENE_LIST")"
if [[ "$scene_count" -lt 1 ]]; then
    echo "C3 online scene list is empty: $SCENE_LIST" >&2
    exit 1
fi

export BOXFUSION_TR3D_TERMINAL_RUN_TAG="$RUN_TAG"
export BOXFUSION_B6_BOXER_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_C3_ONLINE_ENABLED=1
export BOXFUSION_C3_ONLINE_C2_CACHE_ROOT="$C2_CACHE_ROOT"
export BOXFUSION_C3_ONLINE_PARENT_CACHE_ROOT="$PARENT_CACHE_ROOT"
export BOXFUSION_C3_ONLINE_TMPDIR="${BOXFUSION_C3_ONLINE_TMPDIR:-/dev/shm}"

echo "C3 online prediction-identity observer"
echo "  run tag: $RUN_TAG"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  route: source_rank<=5 AND mask2_depth"
echo "  online evidence: shared YOLOE masks + real metric depth/K/pose"
echo "  frozen evidence: terminal TR3D + C1 depth/DINO Top-5 rank"
echo "  online SAM3/DINO extra forward: disabled/disabled"
echo "  mutation/applied_count: false/0"
echo "  GPUs: $GPU_SPEC"

exec bash "$ROOT/scripts/run_scannet_b6_g0_tr3d_terminal_active.sh" "$GPU_SPEC"

