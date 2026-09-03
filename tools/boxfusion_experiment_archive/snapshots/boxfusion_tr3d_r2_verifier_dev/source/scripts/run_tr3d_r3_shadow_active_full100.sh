#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

RUN_TAG="${1:-${BOXFUSION_R3_ACTIVE_RUN_TAG:-}}"
[[ -n "$RUN_TAG" && "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$ ]] || {
  echo "Usage: $0 <unique-run-tag>" >&2
  exit 2
}

PYTHON_BIN="${BOXFUSION_R3_ACTIVE_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
FROZEN_MANIFEST="$ROOT/manifests/frozen_g0_selective_boxer_full100.json"
PARENT_CACHE_ROOT="$ROOT/cache/tr3d_prefix_boxfusion_causal_p100_full100_v3"
PREFIX_MANIFEST="$ROOT/data/tr3d_prefix_val100_boxfusion_causal_p100_v3/manifests/trajectory_prefix_val100_boxfusion_causal_p100_v3.jsonl"
R3_CACHE_ROOT="$ROOT/cache/tr3d_r3/r3_near_full100_v1"
R3_EXPORT_REPORT="$ROOT/reports/tr3d_r3/r3_near_full100_v1/export_report.json"
COUNTERFACTUAL_REPORT="$ROOT/reports/tr3d_r3/r3_near_full100_v1/counterfactual_audit.json"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
SCANS_ROOT="/extra/ZhaoX/scannet_data/scans"
GT_ROOT="/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data"
ACTIVE_ROOT="$ROOT/results/tr3d_r3_shadow_active/$RUN_TAG"
REPORT_ROOT="$ROOT/reports/tr3d_r3_shadow_active/$RUN_TAG"
LOG_ROOT="$ROOT/logs/tr3d_r3_shadow_active/$RUN_TAG"
EVAL_ROOT="$ROOT/evaluation/tr3d_r3_shadow_active/$RUN_TAG"
MATERIALIZE_REPORT="$REPORT_ROOT/materialize_manifest.json"
PAIRED_REPORT="$REPORT_ROOT/paired_audit.json"
STANDARD_REPORT="$REPORT_ROOT/standard_eval_equivalence.json"
CHECKPOINT_SHA="a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448"
CONFIG_SHA="709b66d9e244ef4385dfa9bbc89895ad06c78534f9d14bb7149b687fd58da785"
FROZEN_SHA="327b0cfb07265db04db3af2f631e27e1165a65c9367a9db1d09a31299911342e"
R3_EXPORT_SHA="bb76491e4e7139be57038134a7f6b55e39e9aeb988f01ad3355209a430fcad4c"
COUNTERFACTUAL_SHA="c3973624ca3300eb67cc132aea717f3653f0bb4c8ccb2f12b878d0e0a84014fe"

for path in \
  "$PYTHON_BIN" "$FROZEN_MANIFEST" "$PREFIX_MANIFEST" "$R3_EXPORT_REPORT" \
  "$COUNTERFACTUAL_REPORT" "$SCENE_LIST" "$SCANS_ROOT" "$GT_ROOT" \
  "$PARENT_CACHE_ROOT" "$R3_CACHE_ROOT"; do
  [[ -e "$path" ]] || { echo "Missing shadow-active input: $path" >&2; exit 2; }
done

check_sha() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$observed" == "$expected" ]] || {
    echo "Pinned SHA mismatch: $path" >&2
    echo "  expected: $expected" >&2
    echo "  observed: $observed" >&2
    exit 2
  }
}
check_sha "$FROZEN_MANIFEST" "$FROZEN_SHA"
check_sha "$R3_EXPORT_REPORT" "$R3_EXPORT_SHA"
check_sha "$COUNTERFACTUAL_REPORT" "$COUNTERFACTUAL_SHA"

for path in "$ACTIVE_ROOT" "$REPORT_ROOT" "$LOG_ROOT" "$EVAL_ROOT"; do
  [[ ! -e "$path" ]] || {
    echo "Refusing existing shadow-active namespace: $path" >&2
    echo "Choose a new run tag; immutable outputs are never overwritten." >&2
    exit 2
  }
done
mkdir -p "$REPORT_ROOT" "$LOG_ROOT" "$EVAL_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "Another process holds $LOG_ROOT/run.lock" >&2; exit 2; }

echo "R3 shadow-active full100 engineering replay"
echo "  frozen G0: 40.2787 / 35.4508 / 15.2181"
echo "  expected counterfactual: 41.4870 / 36.8920 / 23.1078"
echo "  policy: geometry-only; label/score/order/count remain exact"
echo "  status: shadow replay only; formal active authorization remains false"

"$PYTHON_BIN" tools/materialize_tr3d_r3_shadow_active.py \
  --frozen-manifest "$FROZEN_MANIFEST" \
  --r3-export-report "$R3_EXPORT_REPORT" \
  --r3-cache-root "$R3_CACHE_ROOT" \
  --scene-list "$SCENE_LIST" \
  --scans-root "$SCANS_ROOT" \
  --output-root "$ACTIVE_ROOT" \
  --manifest "$MATERIALIZE_REPORT" \
  --prefix-id p100 \
  > "$LOG_ROOT/materialize_stdout.json"

"$PYTHON_BIN" tools/audit_tr3d_r3_shadow_active.py \
  --frozen-manifest "$FROZEN_MANIFEST" \
  --parent-cache-root "$PARENT_CACHE_ROOT" \
  --prefix-manifest "$PREFIX_MANIFEST" \
  --r3-cache-root "$R3_CACHE_ROOT" \
  --r3-export-report "$R3_EXPORT_REPORT" \
  --counterfactual-report "$COUNTERFACTUAL_REPORT" \
  --active-root "$ACTIVE_ROOT" \
  --scene-list "$SCENE_LIST" \
  --prefix-id p100 \
  --expected-parent-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --expected-parent-config-sha256 "$CONFIG_SHA" \
  --gt-root "$GT_ROOT" \
  --scans-root "$SCANS_ROOT" \
  --report "$PAIRED_REPORT" \
  > "$LOG_ROOT/paired_audit_stdout.json"

echo "Paired byte/AP audit passed; starting the unmodified ScanNet evaluator"
(
  cd "$ROOT/evaluation"
  PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
  "$PYTHON_BIN" eval_scannet.py \
    --dataset scannet \
    --data_path "$SCANS_ROOT" \
    --gt_root "$GT_ROOT" \
    --dump_dir "$EVAL_ROOT" \
    --num_point 40000 \
    --cluster_sampling seed_fps \
    --use_3d_nms \
    --use_cls_nms \
    --per_class_proposal \
    --num_workers 0 \
    --gpu 0 \
    --seed 0 \
    --scene_list "$SCENE_LIST" \
    --pred_root "$ACTIVE_ROOT"
) > "$LOG_ROOT/eval_stdout.log" 2>&1

grep -E '^eval (mAP|APrec|ARecall):' "$LOG_ROOT/eval_stdout.log"
"$PYTHON_BIN" tools/verify_tr3d_r3_standard_eval.py \
  --eval-log "$LOG_ROOT/eval_stdout.log" \
  --paired-report "$PAIRED_REPORT" \
  --report "$STANDARD_REPORT" \
  > "$LOG_ROOT/standard_eval_verify_stdout.json"

echo "R3 shadow-active full100 replay passed all three stages"
echo "  predictions: $ACTIVE_ROOT"
echo "  paired audit: $PAIRED_REPORT"
echo "  standard equivalence: $STANDARD_REPORT"
