#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

RUN_TAG="${1:-${BOXFUSION_R3_RUN_TAG:-}}"
[[ -n "$RUN_TAG" && "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$ ]] || {
  echo "Usage: $0 <unique-run-tag>" >&2
  exit 2
}
RESUME="${BOXFUSION_R3_RESUME:-0}"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || {
  echo "BOXFUSION_R3_RESUME must be 0 or 1" >&2
  exit 2
}

PYTHON_BIN="${BOXFUSION_R3_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
FROZEN_MANIFEST="$ROOT/manifests/frozen_g0_selective_boxer_full100.json"
PARENT_CACHE_ROOT="$ROOT/cache/tr3d_prefix_boxfusion_causal_p100_full100_v3"
PREFIX_MANIFEST="$ROOT/data/tr3d_prefix_val100_boxfusion_causal_p100_v3/manifests/trajectory_prefix_val100_boxfusion_causal_p100_v3.jsonl"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
DEVELOPMENT_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
SCANS_ROOT="/extra/ZhaoX/scannet_data/scans"
GT_ROOT="/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data"
R3_CACHE_ROOT="$ROOT/cache/tr3d_r3/$RUN_TAG"
REPORT_ROOT="$ROOT/reports/tr3d_r3/$RUN_TAG"
CHECKPOINT_SHA="a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448"
CONFIG_SHA="709b66d9e244ef4385dfa9bbc89895ad06c78534f9d14bb7149b687fd58da785"

for path in \
  "$PYTHON_BIN" "$FROZEN_MANIFEST" "$PREFIX_MANIFEST" "$SCENE_LIST" \
  "$DEVELOPMENT_LIST" "$SCANS_ROOT" "$GT_ROOT"; do
  [[ -e "$path" ]] || { echo "Missing R3 input: $path" >&2; exit 2; }
done
if [[ ! -d "$PARENT_CACHE_ROOT" ]]; then
  echo "Missing strict full100 TR3D parent cache: $PARENT_CACHE_ROOT" >&2
  echo "After restoring the NVIDIA device, run:" >&2
  echo "  BOXFUSION_TR3D_RESUME=0 bash scripts/run_tr3d_strict_val100_parent.sh 0,1" >&2
  exit 2
fi
"$PYTHON_BIN" tools/validate_tr3d_residual_cache.py \
  --cache-root "$PARENT_CACHE_ROOT" \
  --scene-list "$SCENE_LIST" \
  --prefix-id p100 \
  --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --config-sha256 "$CONFIG_SHA"

resume_args=()
if [[ "$RESUME" == "1" ]]; then
  [[ -d "$R3_CACHE_ROOT" ]] || { echo "Resume cache is absent: $R3_CACHE_ROOT" >&2; exit 2; }
  resume_args+=(--resume)
else
  [[ ! -e "$R3_CACHE_ROOT" ]] || { echo "R3 cache already exists: $R3_CACHE_ROOT" >&2; exit 2; }
fi
[[ ! -e "$REPORT_ROOT" ]] || { echo "R3 report root already exists: $REPORT_ROOT" >&2; exit 2; }

echo "R3 anchor-near full100 observer"
echo "  frozen rule: per anchor highest TR3D score, replace iff TR3D score > anchor score"
echo "  R2a/R2b: disabled (explicit zero/false sentinels)"
echo "  output: observer sidecars only; G0 predictions remain immutable"

"$PYTHON_BIN" tools/run_tr3d_r3_near_observer.py \
  --frozen-manifest "$FROZEN_MANIFEST" \
  --parent-cache-root "$PARENT_CACHE_ROOT" \
  --prefix-manifest "$PREFIX_MANIFEST" \
  --scene-list "$SCENE_LIST" \
  --scans-root "$SCANS_ROOT" \
  --r3-cache-root "$R3_CACHE_ROOT" \
  --prefix-id p100 \
  --expected-parent-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --expected-parent-config-sha256 "$CONFIG_SHA" \
  --report "$REPORT_ROOT/export_report.json" \
  "${resume_args[@]}"

"$PYTHON_BIN" tools/audit_tr3d_r3_near_correction.py \
  --frozen-manifest "$FROZEN_MANIFEST" \
  --parent-cache-root "$PARENT_CACHE_ROOT" \
  --prefix-manifest "$PREFIX_MANIFEST" \
  --r3-cache-root "$R3_CACHE_ROOT" \
  --r3-export-report "$REPORT_ROOT/export_report.json" \
  --scene-list "$SCENE_LIST" \
  --development-scene-list "$DEVELOPMENT_LIST" \
  --prefix-id p100 \
  --expected-parent-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --expected-parent-config-sha256 "$CONFIG_SHA" \
  --gt-root "$GT_ROOT" \
  --scans-root "$SCANS_ROOT" \
  --report "$REPORT_ROOT/counterfactual_audit.json" \
  > "$REPORT_ROOT/audit_stdout.json"

"$PYTHON_BIN" tools/summarize_tr3d_r3_audit.py \
  "$REPORT_ROOT/counterfactual_audit.json" | tee "$REPORT_ROOT/summary.txt"
echo "R3 heldout90 audit complete: $REPORT_ROOT"
