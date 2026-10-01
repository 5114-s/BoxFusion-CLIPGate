#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

TAG="${1:-c3_online_shadow_fixed10_v1}"
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,95}$ ]] || {
  echo "usage: $0 [unique-run-tag]" >&2
  exit 2
}

PYTHON_BIN="${BOXFUSION_C3_ONLINE_SHADOW_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
SOURCE_RUN_TAG="${BOXFUSION_C3_ONLINE_SOURCE_RUN_TAG:-c3_online_identity_fixed10_v1}"
SCENE_LIST="${BOXFUSION_C3_ONLINE_SHADOW_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
LIST_SHA="$(sha256sum "$SCENE_LIST" | awk '{print substr($1,1,12)}')"
SOURCE_NAMESPACE="${BOXFUSION_C3_ONLINE_SOURCE_NAMESPACE:-$SOURCE_RUN_TAG/$(basename "$SCENE_LIST" .txt)-$LIST_SHA}"
ANCHOR_ROOT="${BOXFUSION_C3_ONLINE_SHADOW_ANCHOR_ROOT:-$ROOT/results/b6_g0_tr3d_terminal/$SOURCE_NAMESPACE}"
DIAGNOSTICS_ROOT="${BOXFUSION_C3_ONLINE_SHADOW_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/b6_g0_tr3d_terminal/$SOURCE_NAMESPACE/online/tr3d_c3_online_identity}"
PARENT_ROOT="${BOXFUSION_C3_ONLINE_SHADOW_PARENT_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/cache/tr3d_prefix_boxfusion_causal_p100_full100_v3}"
SCANS_ROOT="${BOXFUSION_C3_ONLINE_SHADOW_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
GT_ROOT="${BOXFUSION_C3_ONLINE_SHADOW_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
PRED_ROOT="${BOXFUSION_C3_ONLINE_SHADOW_PRED_ROOT:-$ROOT/results/tr3d_c3_online_shadow/$TAG}"
REPORT_ROOT="${BOXFUSION_C3_ONLINE_SHADOW_REPORT_ROOT:-$ROOT/reports/tr3d_c3_online_shadow/$TAG}"
LOG_ROOT="${BOXFUSION_C3_ONLINE_SHADOW_LOG_ROOT:-$ROOT/logs/tr3d_c3_online_shadow/$TAG}"
EVAL_ROOT="${BOXFUSION_C3_ONLINE_SHADOW_EVAL_ROOT:-$ROOT/eval_outputs/tr3d_c3_online_shadow/$TAG}"
MANIFEST="$REPORT_ROOT/materialize_manifest.json"
AUDIT="$REPORT_ROOT/identity_audit.json"
EVAL_REPORT="$REPORT_ROOT/paired_eval.json"
EVALUATOR="$ROOT/evaluation/eval_scannet.py"

for path in "$PYTHON_BIN" "$SCENE_LIST" "$EVALUATOR"; do
  [[ -f "$path" ]] || { echo "Missing online C3 shadow input: $path" >&2; exit 2; }
done
for path in "$ANCHOR_ROOT" "$DIAGNOSTICS_ROOT" "$PARENT_ROOT" "$SCANS_ROOT" "$GT_ROOT"; do
  [[ -d "$path" ]] || { echo "Missing online C3 shadow directory: $path" >&2; exit 2; }
done
SCENE_COUNT="$(awk 'NF && $1 !~ /^#/ {count += 1} END {print count + 0}' "$SCENE_LIST")"
case "$SCENE_COUNT" in
  1|10|100) ;;
  *) echo "Online C3 shadow supports exactly 1, 10, or 100 scenes" >&2; exit 2 ;;
esac
for path in "$PRED_ROOT" "$REPORT_ROOT" "$LOG_ROOT" "$EVAL_ROOT"; do
  [[ ! -e "$path" ]] || {
    echo "Refusing existing immutable online C3 shadow namespace: $path" >&2
    echo "Choose a new run tag." >&2
    exit 2
  }
done

mkdir -p "$REPORT_ROOT" "$LOG_ROOT/mplconfig" "$EVAL_ROOT/baseline" "$EVAL_ROOT/shadow"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "Another online C3 shadow driver holds $LOG_ROOT/run.lock" >&2; exit 2; }

echo "Online C3 append-only AP shadow"
echo "  scenes: $SCENE_COUNT from $SCENE_LIST"
echo "  anchor: $ANCHOR_ROOT"
echo "  diagnostics: $DIAGNOSTICS_ROOT"
echo "  route: source_rank<=5 AND online_yoloe_mask2_depth"
echo "  score policy: frozen C1 rank, every candidate below every anchor"
echo "  authority: shadow-only; live mutation and formal activation remain false"

COMMON_ENV=(env -u PYTHONPATH TMPDIR="${BOXFUSION_C3_ONLINE_SHADOW_TMPDIR:-/dev/shm}" PYTHONDONTWRITEBYTECODE=1)
"${COMMON_ENV[@]}" "$PYTHON_BIN" tools/materialize_tr3d_c3_online_shadow.py \
  --scene-list "$SCENE_LIST" \
  --identity-diagnostics-root "$DIAGNOSTICS_ROOT" \
  --parent-cache-root "$PARENT_ROOT" \
  --anchor-prediction-root "$ANCHOR_ROOT" \
  --output-root "$PRED_ROOT" --prefix-id p100 --manifest "$MANIFEST" \
  > "$LOG_ROOT/materialize_stdout.json"

"${COMMON_ENV[@]}" "$PYTHON_BIN" tools/audit_tr3d_c3_online_shadow.py \
  --manifest "$MANIFEST" --scene-list "$SCENE_LIST" \
  --identity-diagnostics-root "$DIAGNOSTICS_ROOT" \
  --parent-cache-root "$PARENT_ROOT" \
  --anchor-prediction-root "$ANCHOR_ROOT" --output-root "$PRED_ROOT" \
  --prefix-id p100 --report "$AUDIT" > "$LOG_ROOT/identity_audit_stdout.json"

run_eval() {
  local pred_root="$1"
  local dump_root="$2"
  local output_log="$3"
  (
    cd "$ROOT/evaluation"
    env -u PYTHONPATH TMPDIR="${BOXFUSION_C3_ONLINE_SHADOW_TMPDIR:-/dev/shm}" \
      PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
      "$PYTHON_BIN" eval_scannet.py \
        --dataset scannet --data_path "$SCANS_ROOT" --gt_root "$GT_ROOT" \
        --dump_dir "$dump_root" --num_point 40000 --cluster_sampling seed_fps \
        --use_3d_nms --use_cls_nms --per_class_proposal --num_workers 0 \
        --gpu 0 --seed 0 --scene_list "$SCENE_LIST" --pred_root "$pred_root"
  ) > "$output_log" 2>&1
}

echo "GT-free identity audit passed; running paired unmodified ScanNet evaluator"
run_eval "$ANCHOR_ROOT" "$EVAL_ROOT/baseline" "$LOG_ROOT/eval_baseline.log"
run_eval "$PRED_ROOT" "$EVAL_ROOT/shadow" "$LOG_ROOT/eval_shadow.log"

"${COMMON_ENV[@]}" "$PYTHON_BIN" tools/compare_tr3d_c3_online_shadow_eval.py \
  --manifest "$MANIFEST" --audit "$AUDIT" \
  --baseline-log "$LOG_ROOT/eval_baseline.log" \
  --shadow-log "$LOG_ROOT/eval_shadow.log" --evaluator "$EVALUATOR" \
  --report "$EVAL_REPORT" > "$LOG_ROOT/paired_eval_stdout.json"

chmod 0444 "$LOG_ROOT/materialize_stdout.json" \
  "$LOG_ROOT/identity_audit_stdout.json" "$LOG_ROOT/eval_baseline.log" \
  "$LOG_ROOT/eval_shadow.log" "$LOG_ROOT/paired_eval_stdout.json"

echo "=== terminal R3 anchor ==="
grep -E '^eval (mAP|APrec|ARecall):' "$LOG_ROOT/eval_baseline.log"
echo "=== terminal R3 + online C3 low-score shadow ==="
grep -E '^eval (mAP|APrec|ARecall):' "$LOG_ROOT/eval_shadow.log"
echo "Online C3 shadow evaluation completed"
echo "  report: $EVAL_REPORT"
