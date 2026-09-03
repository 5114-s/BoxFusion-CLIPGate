#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

RUN_TAG="${1:-${BOXFUSION_R2_RUN_TAG:-}}"
if [[ -z "$RUN_TAG" || ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$ ]]; then
  echo "Usage: $0 <unique-run-tag>" >&2
  exit 2
fi

PYTHON_BIN="${BOXFUSION_R2_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
PARENT_CACHE_ROOT="${BOXFUSION_R2_PARENT_CACHE_ROOT:-$ROOT/cache/tr3d_prefix_boxfusion_causal_p100_fixed10_v2}"
PREFIX_MANIFEST="${BOXFUSION_R2_PREFIX_MANIFEST:-$ROOT/data/tr3d_prefix_val10_boxfusion_causal_p100_v2/manifests/trajectory_prefix_val10_boxfusion_causal_p100_v2.jsonl}"
FRAMES_ROOT="${BOXFUSION_R2_FRAMES_ROOT:-/data/ZhaoX/BoxFusion/upstream_clean/scannet_readme_frames}"
SCENE_LIST="${BOXFUSION_R2_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
FROZEN_MANIFEST="${BOXFUSION_R2_FROZEN_MANIFEST:-$ROOT/manifests/frozen_g0_selective_boxer_full100.json}"
GT_ROOT="${BOXFUSION_R2_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCANS_ROOT="${BOXFUSION_R2_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
PREFIX_ID="${BOXFUSION_R2_PREFIX_ID:-p100}"
TOP_K="${BOXFUSION_R2_TOP_K:-5}"
PIXEL_STRIDE="${BOXFUSION_R2_PIXEL_STRIDE:-4}"
MARGIN="${BOXFUSION_R2_MARGIN:-0.05}"
RESUME="${BOXFUSION_R2_RESUME:-0}"
R2_CACHE_ROOT="${BOXFUSION_R2_CACHE_ROOT:-$ROOT/cache/tr3d_r2a/$RUN_TAG}"
REPORT_TAG="${BOXFUSION_R2_REPORT_TAG:-$RUN_TAG}"
REPORT_ROOT="${BOXFUSION_R2_REPORT_ROOT:-$ROOT/reports/tr3d_r2a/$REPORT_TAG}"
PARENT_CHECKPOINT_SHA="${BOXFUSION_R2_PARENT_CHECKPOINT_SHA256:-a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448}"
PARENT_CONFIG_SHA="${BOXFUSION_R2_PARENT_CONFIG_SHA256:-709b66d9e244ef4385dfa9bbc89895ad06c78534f9d14bb7149b687fd58da785}"

for path in "$PYTHON_BIN" "$PREFIX_MANIFEST" "$SCENE_LIST" "$FROZEN_MANIFEST"; do
  [[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
[[ -d "$PARENT_CACHE_ROOT" ]] || { echo "Missing parent cache: $PARENT_CACHE_ROOT" >&2; exit 2; }
[[ -d "$FRAMES_ROOT" ]] || { echo "Missing ScanNet frames: $FRAMES_ROOT" >&2; exit 2; }
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || { echo "BOXFUSION_R2_RESUME must be 0 or 1" >&2; exit 2; }
scene_count="$(awk 'NF && $1 !~ /^#/ {count += 1} END {print count + 0}' "$SCENE_LIST")"
[[ "$scene_count" == "10" ]] || { echo "observer10 requires exactly 10 scenes" >&2; exit 2; }
if [[ -e "$R2_CACHE_ROOT" && "$RESUME" != "1" ]]; then
  echo "Refusing existing immutable R2 cache root: $R2_CACHE_ROOT" >&2
  exit 2
fi
if [[ -e "$REPORT_ROOT" ]]; then
  echo "Refusing existing immutable R2 report root: $REPORT_ROOT" >&2
  echo "Use a fresh BOXFUSION_R2_REPORT_TAG when resuming caches." >&2
  exit 2
fi
mkdir -p "$REPORT_ROOT"

"$PYTHON_BIN" tools/verify_frozen_anchor_manifest.py \
  --manifest "$FROZEN_MANIFEST"

echo "TR3D R2a real-depth/free-space observer"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  parent: $PARENT_CACHE_ROOT"
echo "  prefix manifest: $PREFIX_MANIFEST"
echo "  Top-K/pixel-stride/margin: $TOP_K/$PIXEL_STRIDE/$MARGIN"
echo "  GPU use: none (CPU observer)"
echo "  observer_only=true; mutation_enabled=false; applied_count=0"

resume_args=()
if [[ "$RESUME" == "1" ]]; then
  resume_args+=(--resume)
fi
"$PYTHON_BIN" tools/run_tr3d_r2_observer.py \
  --parent-cache-root "$PARENT_CACHE_ROOT" \
  --r2-cache-root "$R2_CACHE_ROOT" \
  --prefix-manifest "$PREFIX_MANIFEST" \
  --frames-root "$FRAMES_ROOT" \
  --scene-list "$SCENE_LIST" \
  --prefix-id "$PREFIX_ID" \
  --expected-parent-checkpoint-sha256 "$PARENT_CHECKPOINT_SHA" \
  --expected-parent-config-sha256 "$PARENT_CONFIG_SHA" \
  --top-k "$TOP_K" \
  --pixel-stride "$PIXEL_STRIDE" \
  --depth-scale 1000 \
  --margin "$MARGIN" \
  --min-depth 0.10 \
  --max-depth 8.0 \
  --image-height 480 \
  --image-width 640 \
  --report "$REPORT_ROOT/export_report.json" \
  "${resume_args[@]}"

"$PYTHON_BIN" tools/audit_tr3d_r2_observer.py \
  --frozen-manifest "$FROZEN_MANIFEST" \
  --parent-cache-root "$PARENT_CACHE_ROOT" \
  --r2-cache-root "$R2_CACHE_ROOT" \
  --r2-export-report "$REPORT_ROOT/export_report.json" \
  --prefix-manifest "$PREFIX_MANIFEST" \
  --frames-root "$FRAMES_ROOT" \
  --scene-list "$SCENE_LIST" \
  --prefix-id "$PREFIX_ID" \
  --expected-parent-checkpoint-sha256 "$PARENT_CHECKPOINT_SHA" \
  --expected-parent-config-sha256 "$PARENT_CONFIG_SHA" \
  --gt-root "$GT_ROOT" \
  --scans-root "$SCANS_ROOT" \
  --report "$REPORT_ROOT/depth_audit.json" \
  > "$REPORT_ROOT/audit_stdout.json"

"$PYTHON_BIN" tools/summarize_tr3d_r2_audit.py \
  "$REPORT_ROOT/depth_audit.json" | tee "$REPORT_ROOT/summary.txt"
echo "R2a observer complete: $REPORT_ROOT"
