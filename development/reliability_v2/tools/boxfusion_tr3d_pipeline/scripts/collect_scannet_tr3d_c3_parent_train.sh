#!/usr/bin/env bash
set -euo pipefail

# Collect train-only online C3 features directly from terminal TR3D parent
# proposals.  No validation C1/C2/SAM3 cache is read by this route.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
GPU_SPEC="${1:-0,1}"
TAG="${BOXFUSION_C3_TRAIN_RUN_TAG:-tr3d_c3_online_train100_v1}"
TRAIN_LIST="${BOXFUSION_C3_TRAIN_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
EXPECTED_SCENES="${BOXFUSION_C3_TRAIN_EXPECTED_SCENES:-100}"
PARENT_ROOT="${BOXFUSION_C3_TRAIN_PARENT_CACHE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/cache/tr3d_prefix_boxfusion_causal_p100_train100_v1}"
FRAMES_ROOT="${BOXFUSION_C3_TRAIN_FRAMES_ROOT:-$ROOT/data/scannet_train}"
YOLOE="${BOXFUSION_C3_TRAIN_YOLOE_CHECKPOINT:-/data/ZhaoX/OVM3D-Dett/boxfusion_stage3_dev/models/yoloe-11s-seg-pf.pt}"
PRED_ROOT="$ROOT/results/tr3d_c3_train/$TAG"
ONLINE_DIAGNOSTICS="$ROOT/diagnostics/tr3d_c3_train_online_refinement/$TAG"
C3_DIAGNOSTICS="$ROOT/diagnostics/$TAG"
LOG_ROOT="$ROOT/logs/tr3d_c3_train/$TAG"

[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,95}$ ]] || {
    echo "Invalid train C3 run tag: $TAG" >&2; exit 2;
}
for path in "$TRAIN_LIST" "$YOLOE"; do
    [[ -f "$path" && ! -L "$path" ]] || { echo "Missing train C3 input: $path" >&2; exit 2; }
done
for path in "$PARENT_ROOT" "$FRAMES_ROOT"; do
    [[ -d "$path" && ! -L "$path" ]] || { echo "Missing train C3 root: $path" >&2; exit 2; }
done
scene_count="$(awk 'NF && $1 !~ /^#/ {n++} END {print n+0}' "$TRAIN_LIST")"
parent_count="$(find "$PARENT_ROOT" -mindepth 2 -maxdepth 2 -type f -name p100.npz | wc -l)"
[[ "$EXPECTED_SCENES" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid expected scene count" >&2; exit 2; }
[[ "$scene_count" == "$EXPECTED_SCENES" && "$parent_count" == 100 ]] || {
    echo "Expected scenes=$EXPECTED_SCENES and parent cache=100, got scenes=$scene_count parents=$parent_count" >&2; exit 2;
}
echo "Train-only parent-score C3 diagnostic collection"
echo "  scenes: $scene_count from $TRAIN_LIST"
echo "  route: parent_score_rank<=5 AND online_yoloe_mask2_depth"
echo "  parent cache: $PARENT_ROOT"
echo "  diagnostics: $C3_DIAGNOSTICS"
echo "  GPUs: $GPU_SPEC"

BOXFUSION_ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}" \
BOXFUSION_SCENE_LIST="$TRAIN_LIST" \
BOXFUSION_SCANNET_FRAMES_ROOT="$FRAMES_ROOT" \
BOXFUSION_ONLINE_PRED_ROOT="$PRED_ROOT" \
BOXFUSION_ONLINE_LOG_ROOT="$LOG_ROOT" \
BOXFUSION_DIAGNOSTICS_ROOT="$ONLINE_DIAGNOSTICS" \
BOXFUSION_C3_ONLINE_PARENT_CACHE_ROOT="$PARENT_ROOT" \
BOXFUSION_C3_ONLINE_DIAGNOSTICS_ROOT="$C3_DIAGNOSTICS" \
BOXFUSION_C3_ONLINE_CANDIDATE_SOURCE=parent_score \
BOXFUSION_YOLOE_CHECKPOINT="$YOLOE" \
BOXFUSION_ONLINE_ABLATION_PROFILE=quality_only \
BOXFUSION_SCANNET_MIN_EXTENT=0.40 \
BOXFUSION_SKIP_EVALUATION=1 \
bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"

diagnostic_count="$(find "$C3_DIAGNOSTICS" -maxdepth 1 -type f -name 'scene*_c3_online_identity.json' | wc -l)"
[[ "$diagnostic_count" == "$EXPECTED_SCENES" ]] || {
    echo "Expected $EXPECTED_SCENES C3 diagnostics, found $diagnostic_count" >&2; exit 1;
}
echo "Train-only C3 diagnostic collection completed: $C3_DIAGNOSTICS"
