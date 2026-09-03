#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

TAG="${1:-c3_top5_mask2_active_full100_v1}"
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,95}$ ]] || {
  echo "usage: $0 [unique-run-tag]" >&2
  exit 2
}

PYTHON_BIN="${BOXFUSION_C3_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
SCENE_LIST="${BOXFUSION_C3_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
C2_RUN="${BOXFUSION_C3_C2_RUN:-$ROOT/artifacts/tr3d_c2_maskrgbd/c2_c1top10_full100_v1}"
C2_REPORT="${BOXFUSION_C3_C2_EXPORT_REPORT:-$C2_RUN/reports/export_report.json}"
C2_CACHE_ROOT="${BOXFUSION_C3_C2_CACHE_ROOT:-$C2_RUN/cache}"
PARENT_ROOT="${BOXFUSION_C3_PARENT_CACHE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/cache/tr3d_prefix_boxfusion_causal_p100_full100_v3}"
ANCHOR_ROOT="${BOXFUSION_C3_ANCHOR_ROOT:-/extra/ZhaoX/codex_artifacts/boxfusion_r3_20260805/boxfusion_tr3d_terminal_paired_full100/results/b6_g0_tr3d_terminal/g0_tr3d_terminal_paired_full100_v1/scannetv2_val-4b18fc586f7a}"
SHADOW_REPORT="${BOXFUSION_C3_SHADOW_REPORT:-$ROOT/artifacts/tr3d_c3_shadow/c3_top5_mask2_full100_v1/report.json}"
SCANS_ROOT="${BOXFUSION_C3_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
GT_ROOT="${BOXFUSION_C3_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
PRED_ROOT="${BOXFUSION_C3_PRED_ROOT:-$ROOT/results/tr3d_c3_active/$TAG}"
REPORT_ROOT="${BOXFUSION_C3_REPORT_ROOT:-$ROOT/reports/tr3d_c3_active/$TAG}"
LOG_ROOT="${BOXFUSION_C3_LOG_ROOT:-$ROOT/logs/tr3d_c3_active/$TAG}"
EVAL_ROOT="${BOXFUSION_C3_EVAL_ROOT:-$ROOT/eval_outputs/tr3d_c3_active/$TAG}"
MANIFEST="$REPORT_ROOT/materialize_manifest.json"
IDENTITY_REPORT="$REPORT_ROOT/identity_audit.json"
STANDARD_REPORT="$REPORT_ROOT/standard_eval_verification.json"
EVAL_SCRIPT="$ROOT/evaluation/eval_scannet.py"
MIN_FREE_KB="${BOXFUSION_C3_MIN_FREE_KB:-1048576}"

for path in "$PYTHON_BIN" "$SCENE_LIST" "$C2_REPORT" "$SHADOW_REPORT" "$EVAL_SCRIPT"; do
  [[ -f "$path" ]] || { echo "Missing required C3 input: $path" >&2; exit 2; }
done
for path in "$C2_CACHE_ROOT" "$PARENT_ROOT" "$ANCHOR_ROOT" "$SCANS_ROOT" "$GT_ROOT"; do
  [[ -d "$path" ]] || { echo "Missing required C3 directory: $path" >&2; exit 2; }
done
scene_count="$(awk 'NF && $1 !~ /^#/ {count += 1} END {print count + 0}' "$SCENE_LIST")"
[[ "$scene_count" == "100" ]] || { echo "C3 full100 requires exactly 100 scenes" >&2; exit 2; }
for path in "$PRED_ROOT" "$REPORT_ROOT" "$LOG_ROOT" "$EVAL_ROOT"; do
  [[ ! -e "$path" ]] || {
    echo "Refusing existing immutable C3 namespace: $path" >&2
    echo "Choose a new run tag." >&2
    exit 2
  }
done
available_kb="$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')"
[[ "$available_kb" =~ ^[0-9]+$ ]] || { echo "Could not determine free disk space" >&2; exit 2; }
(( available_kb >= MIN_FREE_KB )) || {
  echo "Refusing C3 run: only ${available_kb} KiB free; require ${MIN_FREE_KB} KiB" >&2
  exit 2
}

mkdir -p "$REPORT_ROOT" "$LOG_ROOT/mplconfig" "$EVAL_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "Another C3 driver holds $LOG_ROOT/run.lock" >&2; exit 2; }

echo "C3 Top-5 Mask-RGBD append-only engineering replay"
echo "  scenes: 100 from $SCENE_LIST"
echo "  anchor: $ANCHOR_ROOT"
echo "  route: source_rank<=5 AND mask2_depth"
echo "  score policy: global C1 rank; every candidate below every anchor"
echo "  contract: no GT/CLIP in materializer; anchor rows byte-equivalent after load"
echo "  status: shadow-only engineering replay; formal activation remains false"

env -u PYTHONPATH TMPDIR="${BOXFUSION_C3_TMPDIR:-/dev/shm}" PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON_BIN" tools/materialize_tr3d_c3_active.py \
    --scene-list "$SCENE_LIST" --c2-export-report "$C2_REPORT" \
    --c2-cache-root "$C2_CACHE_ROOT" --parent-cache-root "$PARENT_ROOT" \
    --active-prediction-root "$ANCHOR_ROOT" --output-root "$PRED_ROOT" \
    --prefix-id p100 --manifest "$MANIFEST" \
    > "$LOG_ROOT/materialize_stdout.json"

env -u PYTHONPATH TMPDIR="${BOXFUSION_C3_TMPDIR:-/dev/shm}" PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON_BIN" tools/audit_tr3d_c3_active.py \
    --manifest "$MANIFEST" --scene-list "$SCENE_LIST" \
    --c2-export-report "$C2_REPORT" --c2-cache-root "$C2_CACHE_ROOT" \
    --parent-cache-root "$PARENT_ROOT" --active-prediction-root "$ANCHOR_ROOT" \
    --output-root "$PRED_ROOT" --prefix-id p100 --report "$IDENTITY_REPORT" \
    > "$LOG_ROOT/identity_audit_stdout.json"

echo "GT-free identity audit passed; starting the unmodified ScanNet evaluator"
(
  cd "$ROOT/evaluation"
  env -u PYTHONPATH TMPDIR="${BOXFUSION_C3_TMPDIR:-/dev/shm}" \
    PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
    "$PYTHON_BIN" eval_scannet.py \
      --dataset scannet --data_path "$SCANS_ROOT" --gt_root "$GT_ROOT" \
      --dump_dir "$EVAL_ROOT" --num_point 40000 --cluster_sampling seed_fps \
      --use_3d_nms --use_cls_nms --per_class_proposal --num_workers 0 \
      --gpu 0 --seed 0 --scene_list "$SCENE_LIST" --pred_root "$PRED_ROOT"
) > "$LOG_ROOT/eval_stdout.log" 2>&1

grep -E '^eval (mAP|APrec|ARecall):' "$LOG_ROOT/eval_stdout.log"
env -u PYTHONPATH TMPDIR="${BOXFUSION_C3_TMPDIR:-/dev/shm}" PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON_BIN" tools/verify_tr3d_c3_standard_eval.py \
    --materialize-manifest "$MANIFEST" --identity-audit "$IDENTITY_REPORT" \
    --shadow-report "$SHADOW_REPORT" --eval-log "$LOG_ROOT/eval_stdout.log" \
    --eval-script "$EVAL_SCRIPT" --report "$STANDARD_REPORT" \
    > "$LOG_ROOT/standard_eval_verify_stdout.json"

chmod 0444 "$LOG_ROOT/materialize_stdout.json" \
  "$LOG_ROOT/identity_audit_stdout.json" "$LOG_ROOT/eval_stdout.log" \
  "$LOG_ROOT/standard_eval_verify_stdout.json"

echo "C3 full100 replay passed materialization, identity, and official evaluation"
echo "  predictions: $PRED_ROOT"
echo "  identity audit: $IDENTITY_REPORT"
echo "  standard verification: $STANDARD_REPORT"
