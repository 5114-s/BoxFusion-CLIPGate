#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

GPU_ID="${1:-0}"
RUN_TAG="${2:-${BOXFUSION_R2B_RUN_TAG:-}}"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU id must be a non-negative integer" >&2
  exit 2
fi
if [[ -z "$RUN_TAG" || ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$ ]]; then
  echo "Usage: $0 <gpu-id> <unique-run-tag>" >&2
  exit 2
fi

PYTHON_BIN="${BOXFUSION_R2B_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
PARENT_CACHE_ROOT="${BOXFUSION_R2B_PARENT_CACHE_ROOT:-$ROOT/cache/tr3d_prefix_boxfusion_causal_p100_fixed10_v2}"
R2A_TAG="${BOXFUSION_R2B_R2A_TAG:-r2a_depth_fixed10_v3}"
R2A_CACHE_ROOT="${BOXFUSION_R2B_R2A_CACHE_ROOT:-$ROOT/cache/tr3d_r2a/$R2A_TAG}"
R2A_EXPORT_REPORT="${BOXFUSION_R2B_R2A_EXPORT_REPORT:-$ROOT/reports/tr3d_r2a/$R2A_TAG/export_report.json}"
R2B_CACHE_ROOT="${BOXFUSION_R2B_CACHE_ROOT:-$ROOT/cache/tr3d_r2b/$RUN_TAG}"
REPORT_ROOT="${BOXFUSION_R2B_REPORT_ROOT:-$ROOT/reports/tr3d_r2b/$RUN_TAG}"
PREFIX_MANIFEST="${BOXFUSION_R2B_PREFIX_MANIFEST:-$ROOT/data/tr3d_prefix_val10_boxfusion_causal_p100_v2/manifests/trajectory_prefix_val10_boxfusion_causal_p100_v2.jsonl}"
FRAMES_ROOT="${BOXFUSION_R2B_FRAMES_ROOT:-/data/ZhaoX/BoxFusion/upstream_clean/scannet_readme_frames}"
SCENE_LIST="${BOXFUSION_R2B_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
FROZEN_MANIFEST="${BOXFUSION_R2B_FROZEN_MANIFEST:-$ROOT/manifests/frozen_g0_selective_boxer_full100.json}"
GT_ROOT="${BOXFUSION_R2B_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCANS_ROOT="${BOXFUSION_R2B_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
OFFICIAL_BOXER_ROOT="${BOXFUSION_R2B_OFFICIAL_BOXER_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer}"
BOXER_COMMIT="${BOXFUSION_R2B_BOXER_COMMIT:-1f86542dc342a4b1d474c87c97c5d1d6566d9148}"
DINO_SHA="${BOXFUSION_R2B_DINO_SHA256:-4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea}"
PARENT_CHECKPOINT_SHA="${BOXFUSION_R2B_PARENT_CHECKPOINT_SHA256:-a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448}"
PARENT_CONFIG_SHA="${BOXFUSION_R2B_PARENT_CONFIG_SHA256:-709b66d9e244ef4385dfa9bbc89895ad06c78534f9d14bb7149b687fd58da785}"
PREFIX_ID="${BOXFUSION_R2B_PREFIX_ID:-p100}"
PRECISION="${BOXFUSION_R2B_PRECISION:-bfloat16}"
RESUME="${BOXFUSION_R2B_RESUME:-0}"

for path in "$PYTHON_BIN" "$R2A_EXPORT_REPORT" "$PREFIX_MANIFEST" "$SCENE_LIST" "$FROZEN_MANIFEST"; do
  [[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
for path in "$PARENT_CACHE_ROOT" "$R2A_CACHE_ROOT" "$FRAMES_ROOT" "$OFFICIAL_BOXER_ROOT"; do
  [[ -d "$path" ]] || { echo "Missing required directory: $path" >&2; exit 2; }
done
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || { echo "BOXFUSION_R2B_RESUME must be 0 or 1" >&2; exit 2; }
scene_count="$(awk 'NF && $1 !~ /^#/ {count += 1} END {print count + 0}' "$SCENE_LIST")"
[[ "$scene_count" == "10" ]] || { echo "R2b observer10 requires exactly 10 scenes" >&2; exit 2; }
if [[ -e "$R2B_CACHE_ROOT" && "$RESUME" != "1" ]]; then
  echo "Refusing existing immutable R2b cache root: $R2B_CACHE_ROOT" >&2
  exit 2
fi
if [[ -e "$REPORT_ROOT" ]]; then
  echo "Refusing existing immutable R2b report root: $REPORT_ROOT" >&2
  exit 2
fi
mkdir -p "$REPORT_ROOT"

"$PYTHON_BIN" tools/verify_frozen_anchor_manifest.py --manifest "$FROZEN_MANIFEST"

echo "TR3D R2b Boxer-DINOv3 multi-view feature observer"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  GPU: physical $GPU_ID; precision=$PRECISION"
echo "  R2a parent: $R2A_CACHE_ROOT"
echo "  R2b cache: $R2B_CACHE_ROOT"
echo "  support-mask pooling; CLIP semantics unchanged"
echo "  observer_only=true; mutation_enabled=false; applied_count=0"

resume_args=()
if [[ "$RESUME" == "1" ]]; then
  resume_args+=(--resume)
fi
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" tools/run_tr3d_r2b_feature_observer.py \
  --parent-cache-root "$PARENT_CACHE_ROOT" \
  --r2a-cache-root "$R2A_CACHE_ROOT" \
  --r2a-export-report "$R2A_EXPORT_REPORT" \
  --r2b-cache-root "$R2B_CACHE_ROOT" \
  --prefix-manifest "$PREFIX_MANIFEST" \
  --frames-root "$FRAMES_ROOT" \
  --scene-list "$SCENE_LIST" \
  --prefix-id "$PREFIX_ID" \
  --expected-parent-checkpoint-sha256 "$PARENT_CHECKPOINT_SHA" \
  --expected-parent-config-sha256 "$PARENT_CONFIG_SHA" \
  --official-boxer-root "$OFFICIAL_BOXER_ROOT" \
  --expected-boxer-commit "$BOXER_COMMIT" \
  --expected-dino-sha256 "$DINO_SHA" \
  --input-height 960 \
  --input-width 960 \
  --precision "$PRECISION" \
  --device cuda \
  --min-support-points 2 \
  --min-feature-cells 1 \
  --feature-storage float16 \
  --report "$REPORT_ROOT/export_report.json" \
  "${resume_args[@]}"

"$PYTHON_BIN" tools/verify_frozen_anchor_manifest.py --manifest "$FROZEN_MANIFEST"

"$PYTHON_BIN" tools/audit_tr3d_r2b_feature_observer.py \
  --frozen-manifest "$FROZEN_MANIFEST" \
  --parent-cache-root "$PARENT_CACHE_ROOT" \
  --r2a-cache-root "$R2A_CACHE_ROOT" \
  --r2a-export-report "$R2A_EXPORT_REPORT" \
  --r2b-cache-root "$R2B_CACHE_ROOT" \
  --r2b-export-report "$REPORT_ROOT/export_report.json" \
  --prefix-manifest "$PREFIX_MANIFEST" \
  --frames-root "$FRAMES_ROOT" \
  --scene-list "$SCENE_LIST" \
  --prefix-id "$PREFIX_ID" \
  --expected-parent-checkpoint-sha256 "$PARENT_CHECKPOINT_SHA" \
  --expected-parent-config-sha256 "$PARENT_CONFIG_SHA" \
  --official-boxer-root "$OFFICIAL_BOXER_ROOT" \
  --gt-root "$GT_ROOT" \
  --scans-root "$SCANS_ROOT" \
  --report "$REPORT_ROOT/feature_audit.json" \
  > "$REPORT_ROOT/audit_stdout.json"

"$PYTHON_BIN" tools/summarize_tr3d_r2b_audit.py \
  "$REPORT_ROOT/feature_audit.json" | tee "$REPORT_ROOT/summary.txt"
echo "R2b observer/audit complete: $REPORT_ROOT"
