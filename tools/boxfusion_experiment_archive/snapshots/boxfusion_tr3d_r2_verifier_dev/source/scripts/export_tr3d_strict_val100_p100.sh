#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

PYTHON_BIN="${BOXFUSION_TR3D_EXPORT_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
OUTPUT_ROOT="$ROOT/data/tr3d_prefix_val100_boxfusion_causal_p100_v3"
FRAMES_ROOT="/data/ZhaoX/BoxFusion/upstream_clean/scannet_readme_frames"
FULL_SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
FIXED_SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
SOURCE_INFO="/extra/ZhaoX/scannet_data/scannet_infos_val.pkl"
SOURCE_POINTS="/extra/ZhaoX/scannet_data/points"
FULL_MANIFEST="$OUTPUT_ROOT/manifests/trajectory_prefix_val100_boxfusion_causal_p100_v3.jsonl"
FULL_SUMMARY="$OUTPUT_ROOT/manifests/trajectory_prefix_val100_boxfusion_causal_p100_v3.summary.json"
FULL_INFO="$OUTPUT_ROOT/annotations/scannet_infos_prefix_val100_boxfusion_causal_p100_v3.pkl"
FIXED_ROOT="$ROOT/data/tr3d_prefix_val10_boxfusion_causal_p100_v2"
FIXED_MANIFEST="$FIXED_ROOT/manifests/trajectory_prefix_val10_boxfusion_causal_p100_v2.jsonl"
FIXED_INFO="$FIXED_ROOT/annotations/scannet_infos_prefix_val10_boxfusion_causal_p100_v2.pkl"

for path in \
  "$PYTHON_BIN" "$FRAMES_ROOT" "$FULL_SCENE_LIST" "$FIXED_SCENE_LIST" \
  "$SOURCE_INFO" "$FIXED_MANIFEST" "$FIXED_INFO"; do
  [[ -e "$path" ]] || { echo "Missing strict-export input: $path" >&2; exit 2; }
done
scene_count="$(awk 'NF && $1 !~ /^#/ {count += 1} END {print count + 0}' "$FULL_SCENE_LIST")"
[[ "$scene_count" == "100" ]] || {
  echo "Strict val100 export requires exactly 100 scenes: $FULL_SCENE_LIST" >&2
  exit 2
}

audit_complete_export() {
  for path in "$FULL_MANIFEST" "$FULL_SUMMARY" "$FULL_INFO"; do
    [[ -f "$path" ]] || return 1
  done
  "$PYTHON_BIN" "$ROOT/tools/audit_tr3d_prefix_superset.py" \
    --full-manifest "$FULL_MANIFEST" \
    --fixed-manifest "$FIXED_MANIFEST" \
    --full-info "$FULL_INFO" \
    --fixed-info "$FIXED_INFO" \
    --full-scene-list "$FULL_SCENE_LIST" \
    --fixed-scene-list "$FIXED_SCENE_LIST" \
    --expected-full-scene-count 100
}

if [[ -e "$OUTPUT_ROOT" ]]; then
  if audit_complete_export; then
    echo "Strict val100 p100 export already complete and content-audited."
    echo "No RGB-D decoding or artifact write was performed."
    exit 0
  fi
  echo "Refusing partial/external strict-export root: $OUTPUT_ROOT" >&2
  echo "It may still be produced by another process. This launcher is create-only" >&2
  echo "and will never overwrite or resume a partially visible namespace." >&2
  exit 2
fi

# Atomic namespace claim: a concurrent launcher will observe this directory
# and fail above rather than sharing or overwriting point artifacts.
mkdir "$OUTPUT_ROOT"

echo "Creating strict BoxFusion-clock val100 p100 export"
echo "  output: $OUTPUT_ROOT"
echo "  scenes: 100 from $FULL_SCENE_LIST"
echo "  clock: gap=25, post-frame tail guard"
echo "  pose: previous-valid carry-forward for infinity only"
echo "  RGB-D: pixel_stride=4, depth=[0.1,6.0]m, voxel=0.01m"
echo "  policy: create-only; any failure leaves a quarantined partial namespace"

"$PYTHON_BIN" "$ROOT/tools/export_tr3d_trajectory_prefixes.py" \
  --prepared-root "$OUTPUT_ROOT" \
  --frames-root "$FRAMES_ROOT" \
  --scene-list "$FULL_SCENE_LIST" \
  --source-info "$SOURCE_INFO" \
  --source-points "$SOURCE_POINTS" \
  --output-info-name "$(basename "$FULL_INFO")" \
  --manifest-name "$(basename "$FULL_MANIFEST")" \
  --fractions 1.0 \
  --frame-stride 25 \
  --pixel-stride 4 \
  --voxel-size 0.01 \
  --depth-scale 1000 \
  --min-depth 0.1 \
  --max-depth 6.0 \
  --min-observed-points 20 \
  --min-visibility-fraction 0.0

audit_complete_export
echo "Strict val100 p100 export complete and fixed10-content-identical."

